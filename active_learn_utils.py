import os, random
from utils.simple_solution import SimpleSolution
from utils.evaluator import Evaluator
from utils.data import Data
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.render import DLSyntaxObjectRenderer
from roces import BaseConceptSynthesis
from roces.synthesizer import ConceptSynthesizer
from owlapy.parser import DLSyntaxParser
from utils.dataset import DatasetNoLabel
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from tqdm import tqdm
import json
import torch
import numpy as np, time
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything():
    seed = 42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print('-----Seed Set!-----')
    

def build_roces_vocabulary(data_train, data_test, kb, args):
    def add_data_values(path):
        print("\n*** Finding relevant data values ***")
        values = set()
        for ce, lp in data_train+data_test:
            if '[' in ce:
                for val in re.findall("\[(.*?)\]", ce):
                    values.add(val.split(' ')[-1])
        print("*** Done! ***\n")
        print("Added values: ", values)
        print()
        return list(values)
    renderer = DLSyntaxObjectRenderer()
    individuals = [ind.get_iri().as_str().split("/")[-1] for ind in kb.individuals()]
    atomic_concepts = list(kb.ontology().classes_in_signature())
    atomic_concept_names = [renderer.render(a) for a in atomic_concepts]
    role_names = [rel.get_iri().get_remainder() for rel in kb.ontology().object_properties_in_signature()] + \
                 [rel.get_iri().get_remainder() for rel in kb.ontology().data_properties_in_signature()]
    vocab = atomic_concept_names + role_names + ['⊔', '⊓', '∃', '∀', '¬', '⊤', '⊥', '.', ' ', '(', ')',\
                                                '⁻', '≤', '≥', 'True', 'False', '{', '}', ':', '[', ']',
                                                'double', 'integer', 'date', 'xsd']
    quantified_restriction_values = [str(i) for i in range(1,12)]
    data_values = add_data_values(args.knowledge_base_path)
    vocab = vocab + data_values + quantified_restriction_values
    vocab = sorted(set(vocab)) + ['PAD']
    print("Vocabulary size: ", len(vocab))
    num_examples = min(args.num_examples, kb.individuals_count()//2)
    return vocab, num_examples


def before_pad(arg):
    arg_temp = []
    for atm in arg:
        if atm == 'PAD':
            break
        arg_temp.append(atm)
    return arg_temp

num_examples = 1000
def collate_batch(batch):
    pos_emb_list = []
    neg_emb_list = []
    target_labels = []
    for pos_emb, neg_emb in batch:
        if pos_emb.ndim != 2:
            pos_emb = pos_emb.reshape(1, -1)
        if neg_emb.ndim != 2:
            neg_emb = neg_emb.reshape(1, -1)
        pos_emb_list.append(pos_emb)
        neg_emb_list.append(neg_emb)
    pos_emb_list[0] = F.pad(pos_emb_list[0], (0, 0, 0, num_examples - pos_emb_list[0].shape[0]), "constant", 0)
    pos_emb_list = pad_sequence(pos_emb_list, batch_first=True, padding_value=0)
    neg_emb_list[0] = F.pad(neg_emb_list[0], (0, 0, 0, num_examples - neg_emb_list[0].shape[0]), "constant", 0)
    neg_emb_list = pad_sequence(neg_emb_list, batch_first=True, padding_value=0)
    return pos_emb_list, neg_emb_list

def predict(kb_name, positives, negatives, models, embedding_models, args):
    args.path_to_triples = f"datasets/{kb_name}/Triples/"
    global num_examples
    num_examples = models[0].num_examples
    vocab = models[0].vocab
    inv_vocab = models[0].inv_vocab
    kb_embedding_data = Data(args)
    k = max(len(positives), len(negatives))
    Scores = []
    test_dataset = DatasetNoLabel([("dummy_key", {"positive examples": positives, "negative examples": negatives})], kb_embedding_data, k) #data, triples_data, k
    for i, (model, embedding_model) in enumerate(zip(models, embedding_models)):
        model = model.eval()
        model.to(device)
        scores = []
        test_dataset.load_embeddings(embedding_model.eval())
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_batch, shuffle=False)
        for x1, x2 in tqdm(test_dataloader):
            x1 = x1.to(device)
            x2 = x2.to(device)
            _, sc = model(x1, x2)
            scores.append(sc.detach().cpu()) 
        scores = torch.cat(scores, 0)
        if i == 0:
            cum_scores = scores
        else:
            cum_scores = cum_scores + scores
    avg_scores = cum_scores / len(models)
    pred_sequence = model.inv_vocab[avg_scores.argmax(1)]
    return pred_sequence[0]


def initialize_synthesizer(vocab, num_examples, num_inds, args):
    args.num_inds = num_inds
    roces = ConceptSynthesizer(vocab, num_examples, args)
    roces.refresh()
    return roces.model, roces.embedding_model

def load_models(kb_name, vocab, num_examples, num_inds, args):
    args.knowledge_base_path = "datasets/" + f"{kb_name}/{kb_name}.owl"
    embs = torch.load(f"datasets/{kb_name}/Model_weights/SetTransformer_{args.kb_emb_model}_Emb_inducing_points32.pt", map_location=torch.device("cpu"))
    setattr(args, 'num_entities', embs['emb_ent_real.weight'].shape[0])
    setattr(args, 'num_relations', embs['emb_rel_real.weight'].shape[0])
    models, embedding_models = [], []
    for inds in num_inds:
        model, embedding_model = initialize_synthesizer(vocab, num_examples, inds, args)
        if args.sampling_strategy != 'uniform':
            model.load_state_dict(torch.load(f"datasets/{kb_name}/Model_weights/{args.kb_emb_model}_SetTransformer_inducing_points{inds}.pt", map_location=torch.device("cpu")))
            embedding_model.load_state_dict(torch.load(f"datasets/{kb_name}/Model_weights/SetTransformer_{args.kb_emb_model}_Emb_inducing_points{inds}.pt", map_location=torch.device("cpu")))
        else:
            model.load_state_dict(torch.load(f"datasets/{kb_name}/Model_weights/{args.kb_emb_model}_SetTransformer_uniform_inducing_points{inds}.pt", map_location=torch.device("cpu")))
            embedding_model.load_state_dict(torch.load(f"datasets/{kb_name}/Model_weights/SetTransformer_{args.kb_emb_model}_Emb_uniform_inducing_points{inds}.pt", map_location=torch.device("cpu")))
        models.append(model)
        embedding_models.append(embedding_model)
    return models, embedding_models

def prepare_utilities_for_roces(kb_name, args):
    kb = KnowledgeBase(path=f"datasets/{kb_name}/{kb_name}.owl")
    with open(f"datasets/{kb_name}/Test_data/Data.json", "r") as file:
        test_data = json.load(file)
    with open(f"datasets/{kb_name}/Train_data/Data.json", "r") as file:
        train_data = json.load(file)
    vocab, num_examples = build_roces_vocabulary(train_data, test_data, kb, args)
    namespace = list(kb.individuals())[0].get_iri().get_namespace()
    print("KB namespace: ", namespace)
    print()
    simpleSolution = SimpleSolution(kb)
    evaluator = Evaluator(kb)
    dl_parser = DLSyntaxParser(namespace = namespace)
    all_individuals = set(kb.individuals())
    return kb, simpleSolution, evaluator, dl_parser, all_individuals, vocab, num_examples


def predict_with_roces(kb_name, dl_parser, positives, negatives, models, embedding_models, simpleSolution, args):
    pred = predict(kb_name, positives, negatives, models, embedding_models, args)
    prediction = None
    try:
        end_idx = np.where(pred == 'PAD')[0][0] # remove padding token
    except IndexError:
        end_idx = -1
    pred = pred[:end_idx]
    try:
        prediction = dl_parser.parse("".join(pred.tolist()))
    except Exception as err:
        try:
            pred = simpleSolution.predict(pred.sum())
            prediction = dl_parser.parse(pred)
        except Exception:
            print(f"Could not understand expression {pred}")
    if prediction is None:
        prediction = dl_parser.parse('⊤')
    return prediction

def query_oracle(prediction, oracle, kb, positives, negatives, all_individuals, pos_diff, neg_diff, k_max):
    true_positive_examples = set([ind.get_iri().as_str().split("/")[-1] for ind in kb.individuals(oracle)])
    true_negative_examples = all_individuals-true_positive_examples
    try:
        predicted_positives = set([ind.get_iri().as_str().split("/")[-1] for ind in kb.individuals(prediction)])
    except:
        predicted_positives = all_individuals
    covered_positives = predicted_positives.intersection(true_positive_examples)
    candidate_negatives = (true_negative_examples.intersection(all_individuals.difference(predicted_positives))).difference(set(negatives))
    candidate_positives = covered_positives.difference(set(positives))
    new_positives = positives + random.sample(list(candidate_positives), min(k_max, len(candidate_positives)))
    new_negatives = negatives + random.sample(list(candidate_negatives), min(k_max, len(candidate_negatives)))
    if not candidate_positives:
        random.shuffle(new_positives)
        if len(new_positives) >= 2 and random.random() > 0.8:
            new_positives = new_positives[:-1]
        
    if not candidate_negatives:
        random.shuffle(new_negatives)
        if len(new_negatives) >= 2 and random.random() > 0.8:
            new_negatives = new_negatives[:-1]
        
    return new_positives, new_negatives
    
def evaluate_prediction(kb, prediction, oracle, dl_parser, evaluator, simpleSolution, all_individuals):
    positive_examples = set(kb.individuals(oracle))
    negative_examples = all_individuals-positive_examples
    try:
        _, f1 = evaluator.evaluate(prediction, positive_examples, negative_examples)
    except Exception as err:
        print(f"Parsing error on ", prediction)
        prediction = dl_parser.parse('⊤')
        _, f1 = evaluator.evaluate(prediction, positive_examples, negative_examples)
    try:
        prediction_str = simpleSolution.renderer.render(prediction)
    except:
        prediction_str = "Unknown"
    return prediction_str, f1

def start_active_learning(kb_name, kb, models, embedding_models, oracle, positives, negatives, simpleSolution, evaluator, dl_parser, all_individuals, vocab, args, max_iter=10, k_max=5):
    i = 0
    best_prediction = None
    best_score = None
    previous_score = None
    all_predictions = []
    F1 = []
    #kb, simpleSolution, evaluator, dl_parser, all_individuals, vocab = prepare_utilities_for_roces(kb_name, args)
    all_individuals_str = set([ind.get_iri().as_str().split("/")[-1] for ind in all_individuals])
    oracle = dl_parser.parse(oracle)
    while i < max_iter:
        if i == 0:
            prediction = predict_with_roces(kb_name, dl_parser, positives, negatives, models, embedding_models, simpleSolution, args)
            prediction_str, f1 = evaluate_prediction(kb, prediction, oracle, dl_parser, evaluator, simpleSolution, all_individuals)
            best_prediction = prediction_str
            all_predictions.append(prediction_str)
            F1.append(f1)
            new_positives, new_negatives = positives, negatives
            pos_diff = len(new_positives) - len(positives)
            neg_diff = len(new_negatives) - len(negatives)
            previous_score = f1
            best_score = f1
        else:
            copy_pos = new_positives
            copy_neg = new_negatives
            new_positives, new_negatives = query_oracle(prediction, oracle, kb, new_positives, new_negatives, all_individuals_str, pos_diff, neg_diff, k_max)
            pos_diff = len(new_positives) - len(copy_pos)
            neg_diff = len(new_negatives) - len(copy_neg)
            print(f"On {kb_name.upper()}", f"k_max {k_max}", "new positives ==>", len(new_positives))
            print(f"On {kb_name.upper()}", f"k_max {k_max}", "new negatives ==>", len(new_negatives))
            prediction = predict_with_roces(kb_name, dl_parser, positives, negatives, models, embedding_models, simpleSolution, args)
            previous_score = f1
            prediction_str, f1 = evaluate_prediction(kb, prediction, oracle, dl_parser, evaluator, simpleSolution, all_individuals)
            all_predictions.append(prediction_str)
            F1.append(f1)
            if f1 > previous_score:
                best_prediction = prediction_str
        if f1 > best_score:
            print("Improved performance")
            best_score = f1
        i += 1
        print()
        if f1 == 100:
            print("Found perfect solution! Early stopping...")
            break
    return best_prediction, all_predictions, F1