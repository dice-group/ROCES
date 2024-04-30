from .simple_solution import SimpleSolution
from .evaluator import Evaluator
from .helper_funcs import before_pad
from owlapy.parser import DLSyntaxParser
import numpy as np

def compute_quality(KB, namespace, all_individuals, predictions, targets):
    simpleSolution = SimpleSolution(KB)
    evaluator = Evaluator(KB)
    dl_parser = DLSyntaxParser(namespace = namespace)
    Acc = []
    F1 = []
    for i, pb_str in enumerate(targets):
        if isinstance(predictions[0], np.ndarray):
            pb_str = "".join(before_pad(pb_str))
            try:
                end_idx = np.where(predictions[i] == 'PAD')[0][0] # remove padding token
            except IndexError:
                end_idx = -1
            pred = predictions[i][:end_idx]
            try:
                prediction = dl_parser.parse("".join(pred.tolist()))
            except Exception:
                try:
                    pred = simpleSolution.predict(predictions[i].sum())
                    prediction = dl_parser.parse(pred)
                except Exception:
                    print(f"Could not understand expression {pred}")
        elif isinstance(predictions[0], str):
            prediction = dl_parser.parse(predictions[i])
        else:
            prediction = predictions[i]
        target_expression = dl_parser.parse(pb_str) # The target class expression
        positive_examples = set(KB.individuals(target_expression))
        negative_examples = all_individuals-positive_examples
        acc, f1 = evaluator.evaluate(prediction, positive_examples, negative_examples)
        Acc.append(acc)
        F1.append(f1)
    return np.mean(Acc), np.mean(F1)