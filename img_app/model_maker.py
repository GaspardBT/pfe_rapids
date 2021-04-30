import pickle

import cudf
import cuml
from cuml.neighbors import KNeighborsClassifier

print("cuML version", cuml.__version__)


def model_trainer(dataset_path, k=3):
    train = cudf.read_csv(dataset_path)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train.iloc[:, 1:], train.iloc[:, 0])

    pickle.dump(knn, open("./models/model.pkl", "wb"))


if __name__ == "__main__":
    dataset_path = "../data/digit-recognizer/train.csv"
    model_trainer(dataset_path, k=3)
