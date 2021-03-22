import os
from dotenv import load_dotenv
import pandas as pd
from msgHandler import MsgHandler


def work(handler, stats_topic_name, file_path):
    metrics = []
    try:
        while True:
            records = handler.getNextMsgBatch(topic_name=stats_topic_name)
            if records is not None:
                for _, consumerRecords in records.items():
                    for record in consumerRecords:
                        msg = handler.msg_from_record(record)
                        metrics.append(msg)
            pd.DataFrame(metrics).to_csv(file_path, encoding="utf-8", index=False)
    finally:
        handler.closeAll()


if __name__ == "__main__":
    # Chargement des variables d'environnement
    load_dotenv()

    debug = os.getenv("DEBUG")

    broker_list = os.getenv("BROKER_LIST")
    kafka_group = os.getenv("METRIC_KAFKA_GROUP")
    stats_topic_name = os.getenv("STATS_TOPIC")

    consumers_attr = [{"topic": stats_topic_name, "kafka_group": kafka_group + "_url"}]

    handler = MsgHandler(broker_list, consumers_attr=consumers_attr, producer=False)
    predictors = {}

    print("Souscription aux topics")
    handler.subscribeTopics()

    interval = int(os.getenv("INTERVAL"))
    batch_size = int(os.getenv("BATCH_SIZE"))
    n_urls_days = int(os.getenv("N_URLS_DAYS"))

    file_path = (
        "./metrics_"
        + str(n_urls_days)
        + "_"
        + str(interval)
        + "_"
        + str(batch_size)
        + ".csv"
    )
    print("Début de la lecture des messages")
    work(handler=handler, stats_topic_name=stats_topic_name, file_path=file_path)
