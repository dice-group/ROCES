from argparse import Namespace
import argparse
from helper_run_roces import *
import torch, os, numpy as np, random
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    
def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ['t', 'true', 'y', 'yes', '1']:
        return True
    elif v.lower() in ['f', 'false', 'n', 'no', '0']:
        return False
    else:
        raise ValueError('Invalid boolean value.')

if __name__ == '__main__':
    with open("config.json") as config:
        nces_args = json.load(config)
    nces_args = Namespace(**nces_args)

    parser = argparse.ArgumentParser()

    parser.add_argument('--kbs', type=str, nargs='+', default=['carcinogenesis'], choices=['carcinogenesis', 'mutagenesis', 'semantic_bible', 'vicodi'], help='Knowledge base name')
    parser.add_argument('--kb_emb_model', type=str, default="ConEx", help='KB embedding model')
    parser.add_argument('--repeat_pred', type=str2bool, default=False, help='Whether to use the repeated sampling technique')
    parser.add_argument('--save', type=str2bool, default=False, help='Whether to save the evaluation results')
    parser.add_argument('--verbose', type=str2bool, default=False, help='Whether to print the target and predicted class expressions')
    parser.add_argument('--sampling_strategy', type=str, default='original', help='The strategy to sample the number of examples k')
    args = parser.parse_args()
    
    nces_args.kb_emb_model = args.kb_emb_model
    nces_args.sampling_strategy = args.sampling_strategy
    for kb in args.kbs:
        print("*"*25 + " Evaluating ROCES " + "*"*25)
        evaluate_ensemble(kb_name=kb, args=nces_args, repeat_pred=args.repeat_pred, save_results=args.save, verbose=args.verbose)
        print("*"*25 + " Evaluating ROCES " + "*"*25+"\n")
        
    if args.repeat_pred:
        print("\n\n...")
        for kb in args.kbs:
            print("*"*25 + " Evaluating ROCES+ " + "*"*25)
            evaluate_ensemble(kb_name=kb, args=nces_args, repeat_pred=args.repeat_pred, save_results=args.save, verbose=args.verbose)
            print("*"*25 + " Evaluating ROCES+ " + "*"*25+"\n")
            