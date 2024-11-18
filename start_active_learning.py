import json, argparse
from argparse import Namespace
from active_learn_utils import *
    
if __name__=="__main__":
    
    seed_everything()
    
    with open("./config.json") as config:
        nces_args = json.load(config)
        nces_args = Namespace(**nces_args)
    nces_args.kb_emb_model = "ConEx"
    nces_args.sampling_strategy = "original"
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--kbs', type=str, nargs='+', default=['carcinogenesis', 'mutagenesis', 'semantic_bible', 'vicodi'], choices=['carcinogenesis', 'mutagenesis', 'vicodi', 'semantic_bible'], help='Knowledge base name. Check the folder `datasets` to see all available knowledge bases')
    parser.add_argument('--max_iter', type=int, default=20, help='Maximum number of iterations in an active learning task')
    parser.add_argument('--k_max', type=int, nargs='+', default=[3, 5, 7, 9, 11, 13], help='Maximum number of new individuals to label')
    parser.add_argument('--start_size', type=int, default=3, help='Initial number of examples')
    args = parser.parse_args()
    results = {k: {kb: {"f1": [], "predictions": [], "best_pred": []} for kb in args.kbs} for k in args.k_max}
    for kb_name in args.kbs:
        print('#'*100)
        print(f'Starting active learning on KB {kb_name.upper()}...')
        print('#'*100)
        
        kb, simpleSolution, evaluator, dl_parser, all_individuals, vocab, num_examples = prepare_utilities_for_roces(kb_name, nces_args)
        ensemble_models = "+".join(["SetTransformer_I32", "SetTransformer_I64", "SetTransformer_I128"])
        num_inds = [int(model_name.split("I")[-1]) for model_name in ensemble_models.split("+")]
        models, embedding_models = load_models(kb_name, vocab, num_examples, num_inds, nces_args)
        
        
        with open(f"./datasets/{kb_name}/Test_data/Data.json") as file:
            test_data = json.load(file)
        for k in args.k_max:
            for lp,examples in test_data:
                oracle = lp
                print(f"\nTarget class expression ===> {oracle} <=== \n")
                full_positives, full_negatives = examples["positive examples"], examples["negative examples"]
                positives = random.sample(full_positives, min(len(full_positives), args.start_size))
                negatives = random.sample(full_negatives, min(len(full_negatives), args.start_size))
                best_prediction, all_predictions, F1 = start_active_learning(kb_name, kb, models, embedding_models, oracle,
                                                                             positives, negatives, simpleSolution, evaluator,
                                                                             dl_parser, all_individuals, vocab, nces_args,
                                                                             max_iter=args.max_iter, k_max=k)
                results[k][kb_name]["f1"].append(F1)
                results[k][kb_name]["predictions"].append(all_predictions)
                results[k][kb_name]["best_pred"].append(best_prediction)
    with open("Results_active_learning.json", "w") as f:
        json.dump(results, f)
        
    print("\nDone!")