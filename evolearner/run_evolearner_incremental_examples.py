import json
import os, sys
currentpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentpath.split("evolearner")[0])
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import EvoLearner
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.model import OWLNamedIndividual, IRI
import argparse
from tqdm import tqdm
from collections import defaultdict
import time
import numpy as np
from utils.helper_funcs import get_lp_size
from utils.quality_computation import compute_quality


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kb', type=str, default='semantic_bible', choices=['carcinogenesis', 'mutagenesis', 'semantic_bible', 'vicodi'], help='Knowledge base name')
    args = parser.parse_args()
    kb = args.kb
    results = defaultdict(lambda: defaultdict(list))
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
            model = EvoLearner(knowledge_base=KB, max_runtime=300)
            p = [kb_prefix+ind for ind in examples['positive examples'][:num_examples]]
            n = [kb_prefix+ind for ind in examples['negative examples'][:num_examples]]
            print('\tTarget concept: ', str_target_concept)
            typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
            typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
            lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
            t0 = time.time()
            model.fit(lp, verbose=False)
            t1 = time.time()
            duration = t1-t0
            durations.append(duration)
            for desc in model.best_hypotheses(1):
                predictions.append(desc.concept)
        acc, f1 = compute_quality(KB, namespace, all_individuals, predictions, targets)
        results[kb]["acc"].append(acc)
        results[kb]["f1"].append(f1)
        results[kb]["runtime"].append(np.mean(durations))
    results[kb]["examples sizes"].extend(examples_sizes)
    file_name = f"../datasets/EvoLearner_{kb}_incremental.json"
    with open(file_name, "w") as file:
        json.dump(results, file)