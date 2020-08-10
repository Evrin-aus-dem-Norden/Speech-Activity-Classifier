import numpy as np
from sklearn.base import TransformerMixin
from sklearn.mixture import BayesianGaussianMixture


class Relabeler(TransformerMixin):
    """
    Class for relabeling clusters according to targets.
    """
    def __init__(self, config):
        self.config = config
        self.re_dict = dict()

    def fit(self, y):
        print(f"20 first predictions are\n {y[:20]}")
        print("Enter values for relabeling")
        for c in range(self.config.n_classes):
            self.re_dict[c] = int(input(f"{c}: "))

        return self

    def transform(self, y):
        re_y = np.zeros_like(y)
        for key in self.re_dict:
            re_y[y == key] = self.re_dict[key]

        return re_y


class RelabeledBayesianGaussianMixture(BayesianGaussianMixture):
    """
    BayesianGaussianMixture with built-in clusters relabeling.
    """
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.relabeler = None

    def transform(self, X):
        return X

    def fit_predict(self, X, y):
        prediction = super().fit_predict(X)
        self.relabeler = Relabeler(self.config).fit(prediction)
        return self.relabeler.transform(prediction)

    def predict(self, X):
        return self.relabeler.transform(super().predict(X))

