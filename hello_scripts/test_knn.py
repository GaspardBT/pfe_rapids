import time

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


import cuml
from cuml.neighbors import KNeighborsClassifier as cumlkNN
from sklearn.neighbors import KNeighborsClassifier as sklearnkNN

print("cuML version", cuml.__version__)


def main(dataset_path, librairy):
    train = pd.read_csv(dataset_path)
    print("train shape =", train.shape)

    # GRID SEARCH USING CROSS VALIDATION
    for k in range(3, 6):
        time_start_1 = time.time()
        print("k =", k)
        oof = np.zeros(len(train))
        skf = KFold(n_splits=5, shuffle=True, random_state=42)
        for i, (idxT, idxV) in enumerate(
            skf.split(train.iloc[:, 1:], train.iloc[:, 0])
        ):
            time_start_2 = time.time()
            if librairy == "rapids":
                knn = cumlkNN(n_neighbors=k)
            elif librairy == "sklearn":
                knn = sklearnkNN(n_neighbors=k)
            knn.fit(train.iloc[idxT, 1:], train.iloc[idxT, 0])
            oof[idxV] = knn.predict(train.iloc[idxV, 1:])
            acc = (oof[idxV] == np.array(train.iloc[idxV, 0])).sum() / len(idxV)
            print("fold =", i, " acc =", acc, " in ", time.time() - time_start_2)
        acc = (oof == train.iloc[:, 0]).sum() / len(train)
        print("OOF with k =", k, "ACC =", acc, " in ", time.time() - time_start_1)


if __name__ == "__main__":
    dataset_path = "./data/digit-recognizer/train.csv"
    main(dataset_path, librairy="rapids")
    main(dataset_path, librairy="sklearn")
