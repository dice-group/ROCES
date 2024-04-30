import argparse
import json, time
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import os, sys
import urllib.parse
currentpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentpath.split("dllearner")[0])
from binders import DLLearnerBinder
from ontolearn.knowledge_base import KnowledgeBase
from utils.helper_funcs import get_lp_size
from utils.quality_computation import compute_quality

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kb', type=str, default='semantic_bible', choices=['carcinogenesis', 'mutagenesis', 'semantic_bible', 'vicodi'], help='Knowledge base name')
    args = parser.parse_args()
    kb = args.kb
    results = defaultdict(lambda: defaultdict(list))
    kb_path = currentpath.split("dllearner")[0]+f'datasets/{kb}/{kb}.owl'
    dl_learner_binary_path = currentpath.split("dllearner")[0]+'dllearner-1.4.0/'
    print(f"#### On {kb.upper()} ...")
    KB = KnowledgeBase(path=f"../datasets/{kb}/{kb}.owl")
    namespace = list(KB.individuals())[0].get_iri().get_namespace()
    kb_prefix = namespace[:namespace.rfind("/")+1]
    all_individuals = set(KB.individuals())
    with open(f'../datasets/{kb}/Test_data/Data.json') as lp:
        test_data = json.load(lp)
    max_lp_size = max([max(len(lp["positive examples"]), len(lp["negative examples"])) for _, lp in test_data])
    examples_sizes = get_lp_size(max_lp_size)
    print("***Max number of individuals in learning problems: ", max_lp_size, "***")
    targets = [ce[0] for ce in test_data]
    all_individuals = set(KB.individuals())
    for num_examples in tqdm(get_lp_size(max_lp_size), desc="Next example's set size"):
        durations = []
        predictions = []
        for str_target_concept, examples in tqdm(test_data):
            model = DLLearnerBinder(binary_path=dl_learner_binary_path, kb_path=kb_path, model='celoe')
            print('Target concept:', str_target_concept)
            p = [urllib.parse.quote(kb_prefix+ind) for ind in examples['positive examples'][:num_examples]] # encode with urllib as required by dllearner ontology manager
            n = [urllib.parse.quote(kb_prefix+ind) for ind in examples['negative examples'][:num_examples]]
            t0 = time.time()
            try:
                best_pred = model.fit(pos=p, neg=n, max_runtime=300).best_hypotheses()["Prediction"] # Start learning
            except Exception as err:
                print(err)
                best_pred = None
            t1 = time.time()
            duration = t1-t0
            durations.append(duration)
            prediction = best_pred if best_pred is not None else '⊤'
            predictions.append(prediction)
        acc, f1 = compute_quality(KB, namespace, all_individuals, predictions, targets)
        results[kb]["acc"].append(acc)
        results[kb]["f1"].append(f1)
        results[kb]["runtime"].append(np.mean(durations))
    results[kb]["examples sizes"].extend(examples_sizes)
    file_name = f"../datasets/CELOE_{kb}_incremental.json"
    with open(file_name, "w") as file:
        json.dump(results, file)