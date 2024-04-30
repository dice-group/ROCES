import os, json
import numpy as np, copy
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from .dataset import BaseDataset, DatasetROCES, HeadAndRelationBatchLoader
from .data import Data
from roces.synthesizer import ConceptSynthesizer
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
import time
import random
from argparse import Namespace
torch.multiprocessing.set_sharing_strategy('file_system')

base_path = os.path.dirname(os.path.realpath(__file__)).split('utils')[0]

class Experiment:
    
    def __init__(self, vocab, num_examples, kwargs):
        self.decay_rate = kwargs.decay_rate
        self.clip_value = kwargs.grad_clip_value
        self.num_workers = kwargs.num_workers
        self.batch_size = kwargs.batch_size
        self.kb = kwargs.path_to_triples.split("/")[-3]
        self.kb_embedding_data = Data(kwargs)
        self.load_pretrained = kwargs.load_pretrained
        complete_args = vars(kwargs)
        complete_args.update({"num_entities": len(self.kb_embedding_data.entities),\
                              "num_relations": len(self.kb_embedding_data.relations)})
        complete_args = Namespace(**complete_args)
        self.synthesizer = ConceptSynthesizer(vocab, num_examples, complete_args)
        self.kwargs = complete_args
        
    
    def before_pad(self, arg):
        arg_temp = []
        for atm in arg:
            if atm == 'PAD':
                break
            arg_temp.append(atm)
        return arg_temp
    
    
    def compute_accuracy(self, prediction, target):
        def soft(arg1, arg2):
            arg1_ = arg1
            arg2_ = arg2
            if isinstance(arg1_, str):
                arg1_ = set(self.before_pad(BaseDataset.decompose(arg1_)))
            else:
                arg1_ = set(self.before_pad(arg1_))
            if isinstance(arg2_, str):
                arg2_ = set(self.before_pad(BaseDataset.decompose(arg2_)))
            else:
                arg2_ = set(self.before_pad(arg2_))
            return 100*float(len(arg1_.intersection(arg2_)))/len(arg1_.union(arg2_))

        def hard(arg1, arg2):
            arg1_ = arg1
            arg2_ = arg2
            if isinstance(arg1_, str):
                arg1_ = self.before_pad(BaseDataset.decompose(arg1_))
            else:
                arg1_ = self.before_pad(arg1_)
            if isinstance(arg2_, str):
                arg2_ = self.before_pad(BaseDataset.decompose(arg2_))
            else:
                arg2_ = self.before_pad(arg2_)
            return 100*float(sum(map(lambda x,y: x==y, arg1_, arg2_)))/max(len(arg1_), len(arg2_))
        soft_acc = sum(map(soft, prediction, target))/len(target)
        hard_acc = sum(map(hard, prediction, target))/len(target)
        return soft_acc, hard_acc
          

    def get_optimizer(self, synthesizer, embedding_model, optimizer='Adam'):
        if optimizer == 'Adam':
            return torch.optim.Adam(list(synthesizer.parameters())+list(embedding_model.parameters()), lr=self.kwargs.learning_rate)
        elif optimizer == 'SGD':
            return torch.optim.SGD(list(synthesizer.parameters())+list(embedding_model.parameters()), lr=self.kwargs.learning_rate)
        elif optimizer == 'RMSprop':
            return torch.optim.RMSprop(list(synthesizer.parameters())+list(embedding_model.parameters()), lr=self.kwargs.learning_rate)
        else:
            raise ValueError
            print('Unsupported optimizer')
    
    def show_num_learnable_params(self, synthesizer, embedding_model):
        print("*"*20+"Trainable model size"+"*"*20)
        size = sum([p.numel() for p in synthesizer.parameters()])
        size_ = sum([p.numel() for p in embedding_model.parameters()])
        print(f"Synthesizer ({synthesizer.name} with {synthesizer.num_inds} inducing points): {size}")
        print(f"Embedding Model ({embedding_model.name} with {synthesizer.embedding_dim} embedding dimensions): {size_}")
        print("*"*20+"Trainable model size"+"*"*20)
        print()
        return size, size_
    
    def get_data_idxs(self, data):
        data_idxs = [(self.kb_embedding_data.entity2idx.loc[t[0]].values[0],
                      self.kb_embedding_data.relation2idx.loc[t[1]].values[0],
                      self.kb_embedding_data.entity2idx.loc[t[2]].values[0]) for t in data]
        return data_idxs
    
    @staticmethod
    def get_er_vocab(data):
        # head entity and relation
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab
        
    def collate_batch(self, batch):
        pos_emb_list = []
        neg_emb_list = []
        target_tokens_list = []
        target_labels = []
        for pos_emb, neg_emb, label in batch:
            if pos_emb.ndim != 2:
                pos_emb = pos_emb.reshape(1, -1)
            if neg_emb.ndim != 2:
                neg_emb = neg_emb.reshape(1, -1)
            pos_emb_list.append(pos_emb)
            neg_emb_list.append(neg_emb)
            target_labels.append(label)
        pos_emb_list[0] = F.pad(pos_emb_list[0], (0, 0, 0, self.synthesizer.model.num_examples - pos_emb_list[0].shape[0]), "constant", 0)
        pos_emb_list = pad_sequence(pos_emb_list, batch_first=True, padding_value=0)
        neg_emb_list[0] = F.pad(neg_emb_list[0], (0, 0, 0, self.synthesizer.model.num_examples - neg_emb_list[0].shape[0]), "constant", 0)
        neg_emb_list = pad_sequence(neg_emb_list, batch_first=True, padding_value=0)
        target_labels = pad_sequence(target_labels, batch_first=True, padding_value=-100)
        return pos_emb_list, neg_emb_list, target_labels
            
        
    def map_to_token(self, idx_array):
        return self.synthesizer.model.inv_vocab[idx_array]
    
    def train_step(self, dataloader, synthesizer, embedding_model, opt, head_to_relation_batch, emb_batch_iterator, train_on_gpu):
        soft_acc, hard_acc = [], []
        train_losses = []
        for x1, x2, labels in tqdm(dataloader):
            ## Compute KG embedding loss
            head_batch = head_to_relation_batch[emb_batch_iterator%len(head_to_relation_batch)]
            e1_idx, r_idx, emb_targets = head_batch
            if train_on_gpu:
                emb_targets = emb_targets.cuda()
                r_idx = r_idx.cuda()
                e1_idx = e1_idx.cuda()
            if emb_batch_iterator and emb_batch_iterator%len(head_to_relation_batch) == 0:
                random.shuffle(head_to_relation_batch)
            emb_loss = embedding_model.forward_head_and_loss(e1_idx, r_idx, emb_targets)
            target_sequence = self.map_to_token(labels)
            if train_on_gpu:
                x1, x2, labels = x1.cuda(), x2.cuda(), labels.cuda()
            pred_sequence, scores = synthesizer(x1, x2)
            synt_loss = synthesizer.loss(scores, labels)
            loss = 0.5 * (emb_loss + synt_loss)
            s_acc, h_acc = self.compute_accuracy(pred_sequence, target_sequence)
            soft_acc.append(s_acc); hard_acc.append(h_acc)
            train_losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            clip_grad_norm_(synthesizer.parameters(), self.clip_value)
            clip_grad_norm_(embedding_model.parameters(), self.clip_value)
            
            opt.step()
            if self.decay_rate:
                self.scheduler.step()
        if np.random.rand() > 0.7:
            print("Example prediction: ", pred_sequence[np.random.choice(range(x1.shape[0]))])
            print()
        return np.mean(train_losses), np.mean(soft_acc), np.mean(hard_acc)
                
    def validation_step(self, dataloader, synthesizer, train_on_gpu):
        val_losses = []
        soft_acc, hard_acc = [], []
        for x1, x2, labels in tqdm(dataloader):
            target_sequence = self.map_to_token(labels)
            if train_on_gpu:
                x1, x2, labels = x1.cuda(), x2.cuda(), labels.cuda()
            pred_sequence, scores = synthesizer(x1, x2)
            synt_loss = synthesizer.loss(scores, labels)
            s_acc, h_acc = self.compute_accuracy(pred_sequence, target_sequence)
            soft_acc.append(s_acc); hard_acc.append(h_acc)
            val_losses.append(synt_loss.item())
        return np.mean(val_losses), np.mean(soft_acc), np.mean(hard_acc)
        
        
    
    def train(self, train_data, val_data, test_data, epochs=200, test=False, save_model = False, kb_emb_model="ConEx",\
              optimizer = 'Adam', record_runtime=False, final=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            print("Training on CPU, it may take long...")
        else:
            print("GPU available !")
        print()
        print("#"*100)
        print()
        print("{} ({} inducing points) starts training on {} data set \n".format(self.synthesizer.model.name, self.synthesizer.model.num_inds, self.kwargs.knowledge_base_path.split("/")[-2]))
        print("#"*100, "\n")
        desc1 = kb_emb_model+'_'+self.synthesizer.learner_name
        desc2 = self.synthesizer.learner_name+'_'+kb_emb_model+'_'+'Emb'
        if self.kwargs.sampling_strategy == 'uniform':
            print("*** sampling strategy: uniform ***")
            desc1 = desc1 + '_uniform'
            desc2 = desc2 + '_uniform'
        if self.load_pretrained:
            path1 = base_path+f"datasets/{self.kb}/Model_weights/"+desc1+f"_inducing_points{self.synthesizer.model.num_inds}.pt"
            path2 = base_path+f"datasets/{self.kb}/Model_weights/"+desc2+f"_inducing_points{self.synthesizer.model.num_inds}.pt"
            try:
                self.synthesizer.load_pretrained(path1, path2)
                print("\nUsing pretrained model...\n")
            except Exception as err:
                print(err, "\n")
                print("**** Could not load from pretrained, missing file ****\n")
        ## Make a copy of the model (initialization)
        synthesizer = copy.deepcopy(self.synthesizer.model)
        embedding_model = copy.deepcopy(self.synthesizer.embedding_model)
        
        ## Initialize data loader for embedding computation
        triple_data_idxs = self.get_data_idxs(self.kb_embedding_data.data_triples)
        head_to_relation_batch = list(DataLoader(
            HeadAndRelationBatchLoader(er_vocab=self.get_er_vocab(triple_data_idxs), num_e=len(self.kb_embedding_data.entities)),
            batch_size=2*self.batch_size, num_workers=self.num_workers, shuffle=True))
        
        ## Get combined model size
        size1, size2 = self.show_num_learnable_params(synthesizer, embedding_model)
        
        if final:
            desc1 = desc1+'_final'
            desc2 = desc2+'_final'
        if train_on_gpu:
            synthesizer.cuda()
            embedding_model.cuda()
                        
        opt = self.get_optimizer(synthesizer=synthesizer, embedding_model=embedding_model, optimizer=optimizer)
        if self.decay_rate:
            self.scheduler = ExponentialLR(opt, self.decay_rate)
        Train_loss = []
        Train_acc = defaultdict(list)
        if len(val_data):
            Val_loss = []
            Val_acc = defaultdict(list)
        best_score = 0.
        
        if record_runtime:
            t0 = time.time()
            
        train_dataset = DatasetROCES(train_data, self.kb_embedding_data, self.synthesizer.model.vocab, self.synthesizer.model.inv_vocab, self.kwargs)
        if len(val_data):
            val_dataset = DatasetROCES(val_data, self.kb_embedding_data, self.synthesizer.model.vocab, self.synthesizer.model.inv_vocab, self.kwargs)
        emb_batch_iterator = 0
        for e in range(epochs):
            train_dataset.load_embeddings(embedding_model)
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_batch, shuffle=True)
            if len(val_data):
                val_dataset.load_embeddings(embedding_model)
                val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_batch, shuffle=False)
            ##### Training step
            tr_loss, tr_soft_acc, tr_hard_acc = self.train_step(train_dataloader, synthesizer, embedding_model, opt, head_to_relation_batch, emb_batch_iterator, train_on_gpu)
            ##### Training step
            ##### Validation step
            if len(val_data):
                val_loss, val_soft_acc, val_hard_acc = self.validation_step(val_dataloader, synthesizer, train_on_gpu)
            ##### Validation step
            emb_batch_iterator += 1
            Train_loss.append(tr_loss)
            Train_acc['soft'].append(tr_soft_acc)
            Train_acc['hard'].append(tr_hard_acc)
            if len(val_data):
                Val_loss.append(val_loss)
                Val_acc['soft'].append(val_soft_acc)
                Val_acc['hard'].append(val_hard_acc)
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Train loss: {:.4f}...".format(tr_loss),
                      "Val loss: {:.4f}...".format(val_loss),
                      "Train soft acc: {:.2f}%...".format(tr_soft_acc),
                      "Val soft acc: {:.2f}%...".format(val_soft_acc),
                      "Train hard acc: {:.2f}%...".format(tr_hard_acc),
                      "Val hard acc: {:.2f}%...".format(val_hard_acc)
                     )
            else:
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Train loss: {:.4f}...".format(tr_loss),
                      "Train soft acc: {:.2f}%...".format(tr_soft_acc),
                      "Train hard acc: {:.2f}%...".format(tr_hard_acc),
                     )
            weights_cs = copy.deepcopy(synthesizer.state_dict())
            weights_tc = copy.deepcopy(embedding_model.state_dict())
            if len(val_data):
                if Val_acc['hard'] and Val_acc['hard'][-1] > best_score:
                    best_score = Val_acc['hard'][-1]
                    best_weights_cs = weights_cs
                    best_weights_tc = weights_tc
            else:
                if Train_acc['hard'] and Train_acc['hard'][-1] > best_score:
                    best_score = Train_acc['hard'][-1]
                    best_weights_cs = weights_cs
                    best_weights_tc = weights_tc
        synthesizer.load_state_dict(best_weights_cs)
        embedding_model.load_state_dict(best_weights_tc)
        if record_runtime:
            duration = time.time()-t0
            runtime_info = {"Concept synthesizer": synthesizer.name,
                           "Number of Epochs": epochs, "Runtime (s)": duration}
            if not os.path.exists(base_path+f"datasets/{self.kb}/Runtime"):
                os.mkdir(base_path+f"datasets/{self.kb}/Runtime")
            with open(base_path+f"datasets/{self.kb}/Runtime/"+"Runtime_"+desc1+f"_inducing_points{synthesizer.num_inds}.json", "w") as file:
                json.dump(runtime_info, file, indent=3)
                
        results_dict = {"Synthesizer size": size1, "Embedding model size": size2}
        if test:
            test_dataset = DatasetROCES(test_data, self.kb_embedding_data, self.synthesizer.model.vocab, self.synthesizer.model.inv_vocab, self.kwargs)
            test_dataset.load_embeddings(embedding_model)
            test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_batch, shuffle=False)
            print()
            print("#"*50)
            print("Testing the model ....................")
            print()
            synthesizer.eval()
            soft_acc, hard_acc = [], []
            for x1, x2, labels in test_dataloader:
                if train_on_gpu:
                    x1, x2 = x1.cuda(), x2.cuda()
                pred_sequence, _ = synthesizer(x1, x2)
                target_sequence = target_sequence = self.map_to_token(labels)
                s_acc, h_acc = self.compute_accuracy(pred_sequence, target_sequence)
                soft_acc.append(s_acc); hard_acc.append(h_acc)
            te_soft_acc, te_hard_acc = np.mean(soft_acc), np.mean(hard_acc)
            print("Test for {}:".format(synthesizer.name))
            print("Test soft accuracy: ", te_soft_acc)
            print("Test hard accuracy: ", te_hard_acc)
            results_dict.update({"Test soft acc":te_soft_acc, "Test hard acc": te_hard_acc})
        print("Train soft accuracy: {} ... Train hard accuracy: {}".format(max(Train_acc['soft']), max(Train_acc['hard'])))
        if len(val_data):
            print("**Val soft accuracy: {} ... Val hard accuracy: {}**".format(max(Val_acc['soft']), max(Val_acc['hard'])))
        print()
        results_dict.update({"Train max soft acc": max(Train_acc['soft']), "Train max hard acc": max(Train_acc['hard']), "Train min loss": min(Train_loss)})
        if len(val_data):
            results_dict.update({"Val max soft acc": max(Val_acc['soft']), "Val max hard acc": max(Val_acc['hard']), "Val min loss": min(Val_loss)})
        results_dict.update({'Vocab size': len(synthesizer.vocab)})
        if not os.path.exists(base_path+f"datasets/{self.kb}/Results"):
            os.mkdir(base_path+f"datasets/{self.kb}/Results")
        with open(base_path+f"datasets/{self.kb}/Results/"+"Train_Results_"+desc1+f"_inducing_points{synthesizer.num_inds}.json", "w") as file:
                json.dump(results_dict, file, indent=3)
        os.makedirs(base_path+f"datasets/{self.kb}/Model_weights/", exist_ok=True) # directory to save trained models
        self.kb_embedding_data.entity2idx.to_csv(base_path+f"datasets/{self.kb}/Model_weights/"+desc2+\
                                                 f"_inducing_points{synthesizer.num_inds}_entity_idx.csv")
        self.kb_embedding_data.relation2idx.to_csv(base_path+f"datasets/{self.kb}/Model_weights/"+desc2+\
                                                   f"_inducing_points{synthesizer.num_inds}_relation_idx.csv")
        if save_model:
            torch.save(synthesizer.state_dict(), base_path+f"datasets/{self.kb}/Model_weights/"+desc1+f"_inducing_points{synthesizer.num_inds}.pt")
            torch.save(embedding_model.state_dict(), base_path+f"datasets/{self.kb}/Model_weights/"+desc2+f"_inducing_points{synthesizer.num_inds}.pt")
            print("{} and {} saved".format(synthesizer.name, embedding_model.name))
            print()
        if len(val_data):
            plot_data = (np.array(Train_acc['soft']), np.array(Train_acc['hard']), Train_loss, np.array(Val_acc['soft']), np.array(Val_acc['hard']), Val_loss)
        else:
            plot_data = (np.array(Train_acc['soft']), np.array(Train_acc['hard']), Train_loss)
        return plot_data
            
            
    def train_all_nets(self, List_nets, train_data, val_data, test_data, epochs=200, test=False, save_model = False, kb_emb_model='ConEx', optimizer = 'Adam', record_runtime=False, final=False):
        if not os.path.exists(base_path+f"datasets/{self.kb}/Plot_data/"):
            os.mkdir(base_path+f"datasets/{self.kb}/Plot_data/")
                        
        for net in List_nets:
            self.synthesizer.learner_name = net
            desc = kb_emb_model+'_'+net
            if self.kwargs.sampling_strategy == 'uniform':
                desc = desc+'_uniform'
            self.synthesizer.refresh()
            if len(val_data):
                train_soft_acc, train_hard_acc, train_l, val_soft_acc, val_hard_acc, val_l = self.train(train_data, val_data, test_data, epochs, test, save_model, kb_emb_model, optimizer, record_runtime, final)
            else:
                train_soft_acc, train_hard_acc, train_l = self.train(train_data, val_data, test_data, epochs, test, save_model, kb_emb_model, optimizer, record_runtime, final)
            results = {"tr soft acc": list(train_soft_acc), "tr hard acc": list(train_hard_acc), "tr loss": list(train_l)}
            if len(val_data):
                results.update({"val soft acc": list(val_soft_acc), "val hard acc": list(val_hard_acc), "val loss": list(val_l)})
            if save_model:
                with open(base_path+f"datasets/{self.kb}/Plot_data/"+desc+f"_inducing_points{self.synthesizer.model.num_inds}.json", "w") as plot_file:
                    json.dump(results, plot_file, indent=3)
                print()
                print(results)
            else:
                print(results)

            
