import torch

from copy import deepcopy
from numpy import isnan


class ConfusionMatrix():
    """
    FloatTensor of size (n_classes, n_classes). Dimension 1 contains predicted classes,
    dimension 2 contains target classes.
    Computes class precision, recall and f-scores.
    """

    def __init__(self, n_classes, ignore_index=None, class_dict=None):
        self.n_classes = n_classes
        self.matrix = torch.zeros((n_classes, n_classes), dtype=torch.float)
        # ignore_index is used for ignoring the padding token
        self.ignore_index = ignore_index
        self.filter = torch.LongTensor([i for i in range(n_classes) if i is not ignore_index])
        self.class_dict = class_dict

    def __repr__(self):
        return repr(self.matrix.int())

    def __str__(self):
        return str(self.matrix.int())

    def reset(self):
        self.matrix = torch.zeros((self.n_classes, self.n_classes), dtype=torch.float)

    def copy(self):
        return deepcopy(self)

    def add(self, predictions, targets):
        if predictions.numel() != targets.numel():
            raise Exception("The dimensions of matrices do not match.")
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        for i in range(predictions.numel()):
            self.matrix[predictions[i], targets[i]] += 1

    def accuracy(self):
        matrix = self.matrix.index_select(0, self.filter).index_select(1, self.filter)
        res = matrix.diag().sum() / matrix.sum()
        return res

    def precision(self, mean=False):
        matrix = self.matrix.index_select(0, self.filter).index_select(1, self.filter)
        res = matrix.diag() / matrix.sum(dim=0)
        if mean:
            return mean_without_nan(res)
        return res

    def recall(self, mean=False):
        matrix = self.matrix.index_select(0, self.filter).index_select(1, self.filter)
        res = matrix.diag() / matrix.sum(dim=1)
        if mean:
            return mean_without_nan(res)
        return res

    def f_score(self, b2=1, mean=False):
        res = (1 + b2) * self.precision() * self.recall() / (b2 * self.precision() + self.recall())
        # Set NaNs to zero
        res[res.ne(res)] = 0
        if mean:
            return mean_without_nan(res)
        return res

    def class_frequency(self, count_predictions=False):
        """
        Computes the frequency of all the classes by either the number of targets or predictions
        """
        matrix = self.matrix.index_select(0, self.filter).index_select(1, self.filter)
        if count_predictions:
            return matrix.sum(dim=1) / matrix.sum()
        else:
            return matrix.sum(dim=0) / matrix.sum()

    def print_class_stats(self, class_dict=None, fscore_b2=1):
        if not class_dict:
            class_dict = self.class_dict
        precision = self.precision()
        recall = self.recall()
        f_score = self.f_score(b2=fscore_b2)
        headline = f"Class\tPrec.\tRecall\tF-score (b2={fscore_b2})"
        print(headline)
        print(len(headline)*"-")
        for i, j in enumerate(item.item() for item in self.filter):
            print("{:s}\t{:.4f}\t{:.4f}\t{:.4f}".format(class_dict[j], precision[i], recall[i], f_score[i]))
        print(len(headline)*"-")
        print("Mean\t{:.4f}\t{:.4f}\t{:.4f}".format(mean_without_nan(precision),
                                                    mean_without_nan(recall),
                                                    mean_without_nan(f_score)))

    def matrix_to_csv(self, filename, class_dict=None):
        if not class_dict:
            class_dict = self.class_dict
        with open(filename, "w") as csv:
            print(",".join([""] + [class_dict[i] for i in range(self.n_classes)]), file=csv)
            for i in range(self.n_classes):
                print(",".join([class_dict[i]] + [str(int(self.matrix[i, j])) for j in range(self.n_classes)]),
                      file=csv)


def mean_without_nan(vector):
    non_nan = vector.eq(vector)
    vector[vector.ne(vector)] = 0
    return vector.sum() / non_nan.sum()