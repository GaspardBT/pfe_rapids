from cuml.naive_bayes import MultinomialNB


class NBModel(object):
    """docstring for NBModel."""

    def __init__(self):
        self.model = MultinomialNB()
        self.size_train_dataset = 0

    def online_train(self, X, y, n):
        self.size_train_dataset += n
        self.model.partial_fit(X, y)
