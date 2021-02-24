import os
from dotenv import load_dotenv
import numpy as np
import time
import pickle
from sklearn.datasets import load_svmlight_files

from msgHandler import MsgHandler


def load_svmlight_batched(n=120):
    # svmformat_file = "/opt/dataset/maliciousurl/url_svmlight/"
    svmformat_file = "/var/datastore/url_svmlight/"
    # svmformat_file = (
    #     "/home/gaspard/Documents/work_3A/pfe_rapids/full_stream/datastore/url_svmlight/"
    # )
    files_names = []
    for i in range(n):
        files_names.append(svmformat_file + "Day" + str(i) + ".svm")
    data = load_svmlight_files(files=files_names)
    data_raw = []
    for i in range(n):
        X = data[2 * i]
        y = data[2 * i + 1]
        for j in range(X.shape[0]):
            data_raw.append((X[j], y[j]))

    return data_raw


def urlFeaturesFaker(
    handler, url_features_topic_name, n_urls=None, interval=10, batch_size=5
):
    print("Starts loading the data")
    data_raw = load_svmlight_batched(n=5)
    size_data_raw = len(data_raw)
    if n_urls and n_urls < size_data_raw:
        pass
    else:
        n_urls = size_data_raw
    print("Starts producing fake urls")
    print(size_data_raw)
    # split data in bacth
    batchs = [data_raw[i : i + batch_size] for i in range(0, len(data_raw), batch_size)]

    for batch in batchs:
        for data in batch:
            serialized_data = pickle.dumps(data)
            handler.sendMsg(
                topic_name=url_features_topic_name, key=None, msg=serialized_data,
            )


def work(handler, url_features_topic_name):
    try:
        urlFeaturesFaker(handler, url_features_topic_name)
    finally:
        handler.closeAll()


if __name__ == "__main__":
    # Chargement des variables d'environnement
    load_dotenv()

    debug = os.getenv("DEBUG")
    kafka_group = os.getenv("TRAINER_KAFKA_GROUP")

    broker_list = os.getenv("BROKER_LIST")
    url_features_topic_name = "url_features"  # os.getenv("URL_FEATURES_TOPIC")

    consumers_attr = []

    handler = MsgHandler(broker_list, consumers_attr=consumers_attr, producer=True)

    print("Starts Working")
    work(handler, url_features_topic_name)
