import matplotlib.pyplot as plt
import torch, pandas as pd, numpy as np
import os, json
import argparse
import random
from sklearn.model_selection import train_test_split
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.render import DLSyntaxObjectRenderer
import re

seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


from utils.experiment import Experiment
import json

def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ['t', 'true', 'y', 'yes', '1']:
        return True
    elif v.lower() in ['f', 'false', 'n', 'no', '0']:
        return False
    else:
        raise ValueError('Invalid boolean value.')
        
def build_roces_vocabulary(data_train, data_test, data_val, kb, args):
    def add_data_values(path):
        print("\n*** Finding relevant data values ***")
        values = set()
        for ce, lp in data_train+data_test+data_val:
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
        
parser = argparse.ArgumentParser()
parser.add_argument('--kbs', type=str, nargs='+', default=['carcinogenesis'], choices=['carcinogenesis', 'mutagenesis', 'vicodi', 'semantic_bible'],
                    help='Knowledge base name. Check the folder datasets to see all available knowledge bases')
parser.add_argument('--models', type=str, nargs='+', default=['SetTransformer'], help='Neural models')
parser.add_argument('--sampling_strategy', type=str, default='original', choices=['uniform', 'original'], help='The sampling strategy for sampling example subset sizes')
parser.add_argument('--all_strategies', type=str, nargs='+', default=['original'], choices=['uniform', 'original'], help='Sampling strategies as a list')
parser.add_argument('--kb_emb_model', type=str, default='ConEx', help='Embedding model name')
parser.add_argument('--load_pretrained', type=str2bool, default=False, help='Whether to load pretrained models')
parser.add_argument('--learner_name', type=str, default="SetTransformer", choices=['LSTM', 'GRU', 'SetTransformer'], help='Neural model')
parser.add_argument('--knowledge_base_path', type=str, default="", help='Path to KB owl file')
parser.add_argument('--path_to_triples', type=str, default="", help='Path to KG (result of the conversion of KB to KG)')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--embedding_dim', type=int, default=50, help='Number of embedding dimensions')
parser.add_argument('--num_workers', type=int, default=12, help='Number of workers to use to load training data')
parser.add_argument('--proj_dim', type=int, default=128, help='The projection dimension for examples')
parser.add_argument('--num_inds', type=int, default=32, help='Number of induced instances')
parser.add_argument('--all_num_inds', type=int, nargs='+', default=[32, 64, 128], help='Number of induced instances provided as a list')
parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
parser.add_argument('--num_seeds', type=int, default=1, help='Number of seed components in the output')
parser.add_argument('--num_examples', type=int, default=1000, help='Total number of examples for concept learning')
parser.add_argument('--ln', type=str2bool, default=False, help='Whether to use layer normalization')
parser.add_argument('--decay_rate', type=float, default=0.0, help='Decay rate for the optimizer')
parser.add_argument('--grad_clip_value', type=float, default=5.0, help='Gradient norm clip value')
parser.add_argument('--opt', type=str, default='Adam', help='Name of the optimizer to use')
parser.add_argument('--max_length', type=int, default=48, help='Maximum length of class expressions')
parser.add_argument('--drop_prob', type=float, default=0.1, help='Dropout rate in neural networks')
parser.add_argument('--input_dropout', type=float, default=0.0, help='Input dropout probability for embedding computation')
parser.add_argument('--feature_map_dropout', type=float, default=0.1, help='Feature map dropout probability')
parser.add_argument('--hidden_dropout', type=float, default=0.1, help='Hidden dropout probability during embedding computation')
parser.add_argument('--kernel_size', type=int, default=4, help='Kernel size in ConEx')
parser.add_argument('--num_of_output_channels', type=int, default=8, help='Number of output channels in ConEx')
parser.add_argument('--gamma', type=float, default=1.0, help='Margin in TransE embedding model')
parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=512, help='Training batch size')
parser.add_argument('--validate', type=str2bool, default=False, help='Whether to create a validation data split and validate')
parser.add_argument('--test', type=str2bool, default=True, help='Whether to evaluate the concept synthesizer on the test data during training')
parser.add_argument('--final', type=str2bool, default=False, help='Whether to train the concept synthesizer on test+train data')
parser.add_argument('--save_model', type=str2bool, default=True, help='Whether to save the model after training')
args = parser.parse_args()
    
print("Config: ", vars(args))

with open(f"config.json", "w") as config:
    json.dump(vars(args), config)
    
for sampling_strategy in args.all_strategies:
    args.sampling_strategy = sampling_strategy
    for kb in args.kbs:
        KB = KnowledgeBase(path=f"datasets/{kb}/{kb}.owl")
        data_path = f"datasets/{kb}/Train_data/Data.json"
        with open(data_path, "r") as file:
            data = json.load(file)
        if not args.validate:
            train_data = data
            val_data = []
        else:
            train_data, val_data = train_test_split(data, test_size=0.2, random_state = 42)
        test_data_path = f"datasets/{kb}/Test_data/Data.json"
        with open(test_data_path, "r") as file:
            test_data = json.load(file)
        vocab, num_examples = build_roces_vocabulary(train_data, test_data, val_data, KB, args)
        args.knowledge_base_path = f"datasets/{kb}/{kb}.owl"
        args.path_to_triples = f"datasets/{kb}/Triples/"
        for num_inds in args.all_num_inds:
            args.num_inds = num_inds
            experiment = Experiment(vocab, num_examples, args)
            final = args.final
            test = args.test
            if args.final:
                train_data = train_data + test_data
                test = False
            experiment.train_all_nets(args.models, train_data, val_data, test_data, epochs=args.epochs, test=test, save_model=args.save_model,
                                      kb_emb_model=args.kb_emb_model, optimizer=args.opt, record_runtime=True, final=final)
