import torch
import random
import numpy as np

class BaseDataset:
    """Supervised Machine Learning approach for learning class expressions in ALCHIQ(D) from examples"""
    
    def __init__(self, vocab, inv_vocab, kwargs):
        self.max_length = kwargs.max_length
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        self.kwargs = kwargs
        
    @staticmethod
    def decompose(concept_name: str) -> list:
        """ Decomposes a class expression into a sequence of tokens (atoms) """
        def is_number(char):
            """ Checks if a character can be converted into a number """
            try:
                int(char)
                return True
            except:
                return False
        specials = ['⊔', '⊓', '∃', '∀', '¬', '⊤', '⊥', ' ', '(', ')',\
                    '⁻', '≤', '≥', '{', '}', ':', '[', ']']
        list_ordered_pieces = []
        i = 0
        while i < len(concept_name):
            concept = ''
            while i < len(concept_name) and not concept_name[i] in specials:
                if concept_name[i] == '.' and not is_number(concept_name[i-1]):
                    break
                concept += concept_name[i]
                i += 1
            if concept and i < len(concept_name):
                list_ordered_pieces.extend([concept, concept_name[i]])
            elif concept:
                list_ordered_pieces.append(concept)
            elif i < len(concept_name):
                list_ordered_pieces.append(concept_name[i])
            i += 1
        return list_ordered_pieces
    
    
    def get_labels(self, target):
        target = self.decompose(target)
        labels = [self.vocab[atm] for atm in target]
        return labels, len(target)

class DatasetROCES(BaseDataset, torch.utils.data.Dataset):
    def __init__(self, data, triples_data, vocab, inv_vocab, kwargs):
        super(DatasetROCES, self).__init__(vocab, inv_vocab, kwargs)
        self.k = 5
        self.triples_data = triples_data
        self.data_raw = data
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        
    def load_embeddings(self, embedding_model):
        embeddings, _ = embedding_model.get_embeddings()
        self.embeddings = embeddings.cpu()
        

    def __len__(self):
        return len(self.data_raw)
    
    def __getitem__(self, idx):
        key, value = self.data_raw[idx]
        pos = value['positive examples']
        neg = value['negative examples']
        if self.kwargs.sampling_strategy != 'uniform':
            prob_pos_set = 1.0/(1+np.array(range(min(self.k, len(pos)), len(pos)+1, self.k)))#[1.0/(p+1) for p in range(min(5, len(pos)), len(pos)+1)]
            prob_pos_set = prob_pos_set/prob_pos_set.sum()
            prob_neg_set = 1.0/(1+np.array(range(min(self.k, len(neg)), len(neg)+1, self.k)))
            prob_neg_set = prob_neg_set/prob_neg_set.sum()
            k_pos = np.random.choice(range(min(self.k, len(pos)), len(pos)+1, self.k), replace=False, p=prob_pos_set)
            k_neg = np.random.choice(range(min(self.k, len(neg)), len(neg)+1, self.k), replace=False, p=prob_neg_set)
        else:
            k_pos = np.random.choice(range(min(self.k, len(pos)), len(pos)+1, self.k), replace=False)
            k_neg = np.random.choice(range(min(self.k, len(neg)), len(neg)+1, self.k), replace=False)
        selected_pos = random.sample(pos, k_pos)
        selected_neg = random.sample(neg, k_neg)
        datapoint_pos = self.embeddings[self.triples_data.entity2idx.loc[selected_pos].values.squeeze()]
        datapoint_neg = self.embeddings[self.triples_data.entity2idx.loc[selected_neg].values.squeeze()]
        labels, length = self.get_labels(key)
        return datapoint_pos, datapoint_neg, torch.cat([torch.tensor(labels), self.vocab['PAD']*torch.ones(self.max_length-length)]).long()
    
    
    
class DatasetNoLabel(torch.utils.data.Dataset):
    
    """This class is similar to DatasetROCES except that labels (class expression strings) are not needed here. This is useful for learning problems whose atoms are not present in the trained models. Still NCES instances are able to synthesize quality solutions as they do not rely on labels."""
    
    def __init__(self, data, triples_data, k, random_sample=False):
        super().__init__()
        self.triples_data = triples_data
        self.data_raw = data
        self.k = k
        self.random_sample = random_sample
        
    def load_embeddings(self, embedding_model):
        embeddings, _ = embedding_model.get_embeddings()
        self.embeddings = embeddings.cpu()
        
    def set_k(self, k):
        self.k = k
        

    def __len__(self):
        return len(self.data_raw)
    
    def __getitem__(self, idx):
        _, value = self.data_raw[idx]
        pos = value['positive examples']
        neg = value['negative examples']
        if self.random_sample:
            pos = random.sample(pos, min(self.k, len(pos)))
            neg = random.sample(neg, min(self.k, len(neg)))
        datapoint_pos = self.embeddings[self.triples_data.entity2idx.loc[pos[:self.k]].values.squeeze()]
        datapoint_neg = self.embeddings[self.triples_data.entity2idx.loc[neg[:self.k]].values.squeeze()]
        return datapoint_pos, datapoint_neg

class DatasetInference(BaseDataset, torch.utils.data.Dataset):
    def __init__(self, data, triples_data, k, vocab, inv_vocab, kwargs, random_sample=False):
        super(DatasetInference, self).__init__(vocab, inv_vocab, kwargs)
        self.triples_data = triples_data
        self.data_raw = data
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        self.k = k
        self.random_sample = random_sample
        
    def load_embeddings(self, embedding_model):
        embeddings, _ = embedding_model.get_embeddings()
        self.embeddings = embeddings.cpu()
        

    def __len__(self):
        return len(self.data_raw)
    
    def __getitem__(self, idx):
        key, value = self.data_raw[idx]
        pos = value['positive examples']
        neg = value['negative examples']
        if self.random_sample:
            selected_pos = random.sample(pos, min(len(pos),self.k))
            selected_neg = random.sample(neg, min(len(neg),self.k))
            datapoint_pos = self.embeddings[self.triples_data.entity2idx.loc[selected_pos].values.squeeze()]
            datapoint_neg = self.embeddings[self.triples_data.entity2idx.loc[selected_neg].values.squeeze()]
        else:
            datapoint_pos = self.embeddings[self.triples_data.entity2idx.loc[pos[:self.k]].values.squeeze()]
            datapoint_neg = self.embeddings[self.triples_data.entity2idx.loc[neg[:self.k]].values.squeeze()]
        labels, length = self.get_labels(key)
        return datapoint_pos, datapoint_neg, torch.cat([torch.tensor(labels), self.vocab['PAD']*torch.ones(self.max_length-length)]).long()
        
        
class HeadAndRelationBatchLoader(torch.utils.data.Dataset):
    def __init__(self, er_vocab, num_e):
        self.num_e = num_e
        head_rel_idx = torch.Tensor(list(er_vocab.keys())).long()
        self.head_idx = head_rel_idx[:, 0]
        self.rel_idx = head_rel_idx[:, 1]
        self.tail_idx = list(er_vocab.values())
        assert len(self.head_idx) == len(self.rel_idx) == len(self.tail_idx)

    def __len__(self):
        return len(self.tail_idx)

    def __getitem__(self, idx):
        y_vec = torch.zeros(self.num_e)
        y_vec[self.tail_idx[idx]] = 1  # given head and rel, set 1's for all tails.
        return self.head_idx[idx], self.rel_idx[idx], y_vec