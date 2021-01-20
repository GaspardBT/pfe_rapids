from PIL import Image
import numpy as np
import cv2
import csv
import pickle

import cudf
from cuml.neighbors import KNeighborsClassifier

def load_image(request):
    f = request.files["file"].read()
    npimg = np.fromstring(f, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
    return img


def append_to_csv(filename, csvrow):
    with open(filename, "a") as fd:
        writer = csv.writer(fd)
        writer.writerow(csvrow)


def model_trainer(datapath, modelpath, k=3):
    train = cudf.read_csv(datapath)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train.iloc[:, 1:], train.iloc[:, 0])

    pickle.dump(knn, open(modelpath, "wb"))
