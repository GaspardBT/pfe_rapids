import time

import cudf, cuml
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, KFold
from cuml.neighbors import KNeighborsClassifier, NearestNeighbors

print('cuML version', cuml.__version__)

train = cudf.read_csv('./data/digit-recognizer/train.csv')
print('train shape =', train.shape)

# CREATE 20% VALIDATION SET
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:, 1:], train.iloc[:, 0], test_size=0.2, random_state=42)

# GRID SEARCH FOR OPTIMAL K
accs = []
for k in range(3, 22):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    # Better to use knn.predict() but cuML v0.11.0 has bug
    # y_hat = knn.predict(X_test)
    y_hat_p = knn.predict_proba(X_test)
    acc = (y_hat_p.to_pandas().values.argmax(axis=1) == y_test.to_array()).sum() / y_test.shape[0]
    # print(k,acc)
    print(k, ', ', end='')
    accs.append(acc)

# GRID SEARCH USING CROSS VALIDATION
for k in range(3, 6):
    time_start_1 = time.time()
    print('k =', k)
    oof = np.zeros(len(train))
    skf = KFold(n_splits=5, shuffle=True, random_state=42)
    for i, (idxT, idxV) in enumerate(skf.split(train.iloc[:, 1:], train.iloc[:, 0])):
        time_start_2 = time.time()
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train.iloc[idxT, 1:], train.iloc[idxT, 0])
        # Better to use knn.predict() but cuML v0.11.0 has bug
        # y_hat = knn.predict(train.iloc[idxV,1:])
        y_hat_p = knn.predict_proba(train.iloc[idxV, 1:])
        oof[idxV] = y_hat_p.to_pandas().values.argmax(axis=1)
        acc = (oof[idxV] == train.iloc[idxV, 0].to_array()).sum() / len(idxV)
        print(' fold =', i, ' acc =', acc, ' in ', time.time() - time_start_2)
    acc = (oof == train.iloc[:, 0].to_array()).sum() / len(train)
    print(' OOF with k =', k, 'ACC =', acc, ' in ', time.time() - time_start_1)
