import json
from roces.synthesizer import ConceptSynthesizer
from utils.data import Data
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.parser import DLSyntaxParser
from owlapy.render import DLSyntaxObjectRenderer
from argparse import Namespace
import argparse
from utils.dataset import DatasetNoLabel
from utils.helper_funcs import get_lp_size, before_pad
from utils.quality_computation import compute_quality
from time import time
from tqdm import tqdm
from collections import defaultdict
import torch
import random
import re
random.seed(42)

def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ['t', 'true', 'y', 'yes', '1']:
        return True
    elif v.lower() in ['f', 'false', 'n', 'no', '0']:
        return False
    else:
        raise ValueError('Invalid boolean value.')
        
parser = argparse.ArgumentParser()
parser.add_argument('--sampling_strategy', type=str, default='original', help='The strategy to sample the number of examples k')
parser.add_argument('--approach', type=str, nargs='+', default=['roces'], help='The selected approach, ROCES or NCES2')
parser.add_argument('--save', type=str2bool, default=True, help='Whether to save results')
args = parser.parse_args()

def collate_batch(batch):
    pos_emb_list = []
    neg_emb_list = []
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

def get_dataloader(data, synthesizer, embedding_data, input_size, batch_size=64):
    global num_examples
    num_examples = synthesizer.model.num_examples
    dataset = DatasetNoLabel(data, embedding_data, input_size)
    dataset.load_embeddings(synthesizer.embedding_model)
    dataloader = DataLoader(dataset, collate_fn=collate_batch, batch_size=batch_size, shuffle=False)
    return dataloader


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

def initialize_synthesizer(vocab, num_examples, num_inds, args):
    args.num_inds = num_inds
    roces = ConceptSynthesizer(vocab, num_examples, args)
    roces.refresh()
    return roces

def predict(synthesizers, test_dataloaders):
    preds = []
    models = [synt.model for synt in synthesizers] if isinstance(synthesizers, list) else synthesizers.model
    if isinstance(models, list):
        assert isinstance(test_dataloaders, list), "each model should have its own dataloader"
        for i, (model, dataloader) in enumerate(zip(models, test_dataloaders)):
            model = model.eval()
            scores = []
            for x1, x2 in dataloader:
                _, sc = model(x1, x2)
                scores.append(sc.detach()) 
            scores = torch.cat(scores, 0)
            if i == 0:
                Scores = scores
            else:
                Scores = Scores + scores
        Scores = Scores / len(models)
        pred_sequence = model.inv_vocab[Scores.argmax(1)]
        return pred_sequence
    else:
        models = models.eval()
        for x1, x2 in test_dataloaders:
            pred_sequence, _ = models(x1, x2)
            preds.append(pred_sequence)
        return np.concatenate(preds, 0)
    
def evaluate_nces2_and_roces(kbs, kb_emb_model="ConEx", approach="roces", save=True):
    results = defaultdict(lambda: defaultdict(list))
    assert approach in ['nces2', 'roces'], 'This approach is not supported, use <nces2> or <roces> instead'
    print(f"Evaluating {approach.upper()} with increasing example set sizes...\n")
    for kb in kbs:
        print(f"#### On {kb.upper()} ...")
        KB = KnowledgeBase(path=f"datasets/{kb}/{kb}.owl")
        namespace = list(KB.individuals())[0].get_iri().get_namespace()
        all_individuals = set(KB.individuals())
        with open(f'datasets/{kb}/Test_data/Data.json') as lp:
            test_data = json.load(lp)
        with open(f'datasets/{kb}/Train_data/Data.json') as lp:
            train_data = json.load(lp)
        max_lp_size = max([max(len(lp["positive examples"]), len(lp["negative examples"])) for _, lp in test_data])
        print("***Max number of individuals in learning problems: ", max_lp_size, "***")
        targets = [ce[0] for ce in test_data]
        with open("config.json") as config:
            nces_args = json.load(config)
        nces_args = Namespace(**nces_args)
        nces_args.knowledge_base_path = f"datasets/{kb}/{kb}.owl"
        nces_args.path_to_triples = f"datasets/{kb}/Triples/"
        nces_args.kb = kb
        kb_embedding_data = Data(nces_args)
        nces_args.num_entities = len(kb_embedding_data.entities)
        nces_args.num_relations = len(kb_embedding_data.relations)
        vocab, num_examples = build_roces_vocabulary(train_data, test_data, KB, nces_args)
        all_synthesizers = []
        for num_inds in tqdm([32, 64, 128]):
            print(f"Loaded pretrained model (num_inds = {num_inds})...")
            nces_args.num_inds = num_inds
            synthesizer = initialize_synthesizer(vocab, num_examples, num_inds, nces_args)
            if approach == 'nces2':
                synthesizer.load_pretrained(f"nces2/trained_models/{kb}/Model_weights/{kb_emb_model}_SetTransformer_inducing_points{num_inds}.pt", 
                                    f"nces2/trained_models/{kb}/Model_weights/SetTransformer_{kb_emb_model}_Emb_inducing_points{num_inds}.pt")
            elif args.sampling_strategy != 'uniform':
                synthesizer.load_pretrained(f"datasets/{kb}/Model_weights/{kb_emb_model}_SetTransformer_inducing_points{num_inds}.pt", 
                                f"datasets/{kb}/Model_weights/SetTransformer_{kb_emb_model}_Emb_inducing_points{num_inds}.pt")
            else:
                synthesizer.load_pretrained(f"datasets/{kb}/Model_weights/{kb_emb_model}_SetTransformer_uniform_inducing_points{num_inds}.pt", 
                                f"datasets/{kb}/Model_weights/SetTransformer_{kb_emb_model}_Emb_uniform_inducing_points{num_inds}.pt")
            print("Done.")
            all_synthesizers.append(synthesizer)
        ## Ensemble prediction
        print("Ensemble model prediction...")
        examples_sizes = get_lp_size(max_lp_size)
        for input_size in examples_sizes:
            if input_size > all_synthesizers[0].model.num_examples:
                break
            dataloaders = []
            for synt in all_synthesizers:
                dataloader = get_dataloader(test_data, synt, kb_embedding_data, input_size)
                dataloaders.append(dataloader)
            t0 = time()
            predictions = predict(all_synthesizers, dataloaders)
            t1 = time()
            duration = t1-t0
            acc, f1 = compute_quality(KB, namespace, all_individuals, predictions, targets)
            results[kb]["acc"].append(acc)
            results[kb]["f1"].append(f1)
            results[kb]["runtime"].append(duration/len(predictions))
        results[kb]["examples sizes"].extend(examples_sizes)
        print("Done.")
    print(results)
    if save:
        print('Saving...')
        if approach == 'nces2':
            file_name = f"datasets/NCES2_{kb_emb_model}_incremental.json"
        elif args.sampling_strategy != 'uniform':
            file_name = f"datasets/ROCES_{kb_emb_model}_incremental.json"
        else:
            file_name = f"datasets/ROCES_{kb_emb_model}_uniform_incremental.json"
        with open(file_name, "w") as file:
            json.dump(results, file)
        print('Done.')

for approach in args.approach:
    evaluate_nces2_and_roces(["semantic_bible", "mutagenesis", "carcinogenesis", "vicodi"], approach=approach, save=args.save)