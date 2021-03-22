import os
from dotenv import load_dotenv
import numpy as np
from time import time, sleep
import pickle
from tqdm import tqdm


from sklearn.datasets import load_svmlight_files

from msgHandler import MsgHandler


def load_svmlight_batched(n=120):
    svmformat_file = "/opt/dataset/maliciousurl/url_svmlight/"
    # svmformat_file = "/var/datastore/url_svmlight/"
    # svmformat_file = "/home/gaspard/Documents/CS_3A/work_3A/pfe_rapids/url_class/datastore/url_svmlight/"
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


def work(handler, url_features_topic_name, interval=10, batch_size=5, n_urls_days=5):

    try:
        print("Starts loading the data")
        data_raw = load_svmlight_batched(n=n_urls_days)
        size_data_raw = len(data_raw)
        if n_urls_days and n_urls_days < size_data_raw:
            pass
        else:
            n_urls_days = size_data_raw
        print("Starts producing fake urls")
        print(size_data_raw)
        # split data in bacth
        batchs = [
            data_raw[i : i + batch_size] for i in range(0, len(data_raw), batch_size)
        ]

        for batch in tqdm(batchs):
            time_start = time()
            for data in batch:
                serialized_data = pickle.dumps(data)
                handler.sendMsg(
                    topic_name=url_features_topic_name, key=None, msg=serialized_data,
                )
            time_finish = time()

            time_sleep = max(0, interval - (time_finish - time_start) * 1000)
            if time_sleep == 0:
                print("send took to long: ", (time_finish - time_start) * 1000)
            sleep(time_sleep)
    finally:
        handler.closeAll()


if __name__ == "__main__":
    # Chargement des variables d'environnement
    load_dotenv()

    debug = os.getenv("DEBUG")
    kafka_group = os.getenv("TRAINER_KAFKA_GROUP")

    broker_list = os.getenv("BROKER_LIST")

    interval = int(os.getenv("INTERVAL"))
    batch_size = int(os.getenv("BATCH_SIZE"))
    n_urls_days = int(os.getenv("N_URLS_DAYS"))

    url_features_topic_name = os.getenv("URL_FEATURES_TOPIC")

    consumers_attr = []

    handler = MsgHandler(broker_list, consumers_attr=consumers_attr, producer=True)

    print("Starts Working")
    work(
        handler=handler,
        url_features_topic_name=url_features_topic_name,
        interval=interval,
        batch_size=batch_size,
        n_urls_days=n_urls_days,
    )
    print("End Working")
