from cuml.naive_bayes import MultinomialNB


class NBModel(object):
    """docstring for NBModel."""

    def __init__(self):
        self.model = MultinomialNB()

    def online_train(self, X, y):
        self.model.partial_fit(X, y)
