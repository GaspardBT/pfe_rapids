import pickle

import cudf, cuml
from cuml.neighbors import KNeighborsClassifier, NearestNeighbors

print("cuML version", cuml.__version__)


def model_trainer(k=3):
    train = cudf.read_csv("./data/digit-recognizer/train.csv")

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train.iloc[:, 1:], train.iloc[:, 0])

    pickle.dump(knn, open("/img_app/models/model.pkl", "wb"))


if __name__ == '__main__':
    model_trainer()
