import scipy.sparse as sp

import numpy as np
import cupy as cp

from cuml.naive_bayes import MultinomialNB as rapidsMNB
from sklearn.naive_bayes import MultinomialNB as sklearnMNB


class NBModel(object):
    """docstring for NBModel."""

    def __init__(self, librairy="rapids"):
        self.librairy = librairy
        if self.librairy == "rapids":
            self.model = rapidsMNB()
        elif self.librairy == "sklearn":
            self.model = sklearnMNB()
        else:
            raise ValueError("librairy parameters must be either rapids or sklearn")
        self.size_train_dataset = 0

    def preprocessing(self, data, labels):
        if self.librairy == "rapids":
            X = sp.vstack(data, format="csr")
            y = cp.asarray(labels, dtype=cp.int32)
        elif self.librairy == "sklearn":
            X = np.abs(sp.vstack(data, format="csr"))
            y = labels
        return X, y

    def online_train(self, X, y, n):

        if self.size_train_dataset == 0 and self.librairy == "sklearn":
            self.model.partial_fit(X, y, classes=np.unique(y))
        else:
            self.model.partial_fit(X, y)
        self.size_train_dataset += n
