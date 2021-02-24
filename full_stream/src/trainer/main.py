import os
from dotenv import load_dotenv
import pickle

from msgHandler import MsgHandler

# from model import NBModel


def urlHandler(handler, url_features_topic_name, nbmodel):
    records = handler.getNextMsgBatch(topic_name=url_features_topic_name)
    if records is not None:
        for _, consumerRecords in records.items():
            for record in consumerRecords:
                data = pickle.loads(record.value)
                try:
                    print(data)
                except Exception as e:
                    # msg to logger
                    print(e)


def work(handler, url_features_topic_name, nbmodel):
    try:
        while True:
            urlHandler(handler, url_features_topic_name, nbmodel)
    finally:
        handler.closeAll()


if __name__ == "__main__":
    # Chargement des variables d'environnement
    load_dotenv()

    debug = os.getenv("DEBUG")

    broker_list = os.getenv("BROKER_LIST")
    kafka_group = os.getenv("TRAINER_KAFKA_GROUP")
    url_features_topic_name = os.getenv("URL_FEATURES_TOPIC")

    consumers_attr = [
        {"topic": url_features_topic_name, "kafka_group": kafka_group + "_url"}
    ]

    handler = MsgHandler(broker_list, consumers_attr=consumers_attr, producer=False)
    predictors = {}

    print("Souscription aux topics")
    handler.subscribeTopics()

    print("Init Model")
    nbmodel = 4  # NBModel()
    print("Début de la lecture des messages")
    work(handler, url_features_topic_name, nbmodel)
