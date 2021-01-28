import time
import cupy as cp
import cupyx

from cuml.naive_bayes import MultinomialNB

from sklearn.datasets import load_svmlight_files


def load_svmlight_batched(n=1):
    svmformat_file = "datastore/url_svmlight/"
    files_names = []
    for i in range(n):
        files_names.append(svmformat_file + "Day" + str(i) + ".svm")
    data = load_svmlight_files(files=files_names)
    data_peer = []
    for i in range(n):
        X = data[2 * i]
        y = data[2 * i + 1]
        X = cupyx.scipy.sparse.csr_matrix(X, dtype=cp.float32)
        y = cp.asarray(y, dtype=cp.int32)
        data_peer.append((X, y))
    return data_peer


def main(n=1):
    time1 = time.time()
    data = load_svmlight_batched(n)
    time2= time.time()
    print("time load: ",time2-time1)
    print(len(data))
    model = MultinomialNB()
    total = 0
    time1 = time.time()

    for X, y in data[:-1]:
        model.partial_fit(X, y)
        total += len(y)

    time2 = time.time()
    print("time train: ", time2 - time1)
    # Compute accuracy on training set
    X_test, y_train = data[-1]
    a = model.score(X_test, y_train)
    print(a)
    print(total)


if __name__ == '__main__':
    main(120)
