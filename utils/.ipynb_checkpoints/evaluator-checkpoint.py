from .metrics import Accuracy, F1

class Evaluator:
    def __init__(self, kb):
        self.kb = kb
        
    def evaluate(self, prediction, pos_examples, neg_examples):
        try:
            instances = set(self.kb.individuals(prediction))
        except Exception:
            print("Parsing the prediction failed")
            instances = set(self.kb.individuals())
        f1 = F1.score(pos_examples, neg_examples, instances)
        acc = Accuracy.score(pos_examples, neg_examples, instances)
        return 100*acc, 100*f1