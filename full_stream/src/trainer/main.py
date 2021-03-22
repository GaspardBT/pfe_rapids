import os
from dotenv import load_dotenv
import pickle
from time import time
from datetime import datetime

import numpy as np
import cupy as cp

import scipy.sparse as sp

from msgHandler import MsgHandler

from model import NBModel


def work(handler, stats_topic_name, url_features_topic_name, nbmodel):
    try:
        while True:
            records = handler.getNextMsgBatch(topic_name=url_features_topic_name)
            if records is not None:
                for _, consumerRecords in records.items():
                    time_start = time()
                    data = []
                    labels = []
                    for record in consumerRecords:
                        inputs = pickle.loads(record.value)
                        X, y = inputs
                        data.append(X)
                        labels.append(y)
                    n = len(data)
                    time_load = time()
                    X = sp.vstack(data, format="csr")
                    y = cp.asarray(labels, dtype=cp.int32)
                    try:
                        nbmodel.online_train(X, y, n)
                    except Exception as e:
                        # msg to logger
                        print("error: ", e)
                    time_train = time()
                    metrics = {
                        "timestamp": str(datetime.now()),
                        "time_load": time_load - time_start,
                        "time_train": time_train - time_load,
                        "size_batch": n,
                        # "model": pickle.dumps(nbmodel),
                    }
                    handler.sendMsg(
                        topic_name=stats_topic_name,
                        key=None,
                        msg=handler.serializer(metrics),
                    )
    finally:
        handler.closeAll()


if __name__ == "__main__":
    # Chargement des variables d'environnement
    load_dotenv()

    debug = os.getenv("DEBUG")

    broker_list = os.getenv("BROKER_LIST")
    kafka_group = os.getenv("TRAINER_KAFKA_GROUP")
    url_features_topic_name = os.getenv("URL_FEATURES_TOPIC")
    stats_topic_name = os.getenv("STATS_TOPIC")

    consumers_attr = [
        {"topic": url_features_topic_name, "kafka_group": kafka_group + "_url"}
    ]

    handler = MsgHandler(broker_list, consumers_attr=consumers_attr, producer=True)
    predictors = {}

    print("Souscription aux topics")
    handler.subscribeTopics()

    print("Init Model")
    nbmodel = NBModel()
    print("Début de la lecture des messages")
    work(
        handler=handler,
        stats_topic_name=stats_topic_name,
        url_features_topic_name=url_features_topic_name,
        nbmodel=nbmodel,
    )
