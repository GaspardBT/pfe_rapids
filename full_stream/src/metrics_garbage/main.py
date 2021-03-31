import os
import sys
from datetime import datetime

from dotenv import load_dotenv
import pandas as pd
from msgHandler import MsgHandler


def work(handler, stats_topic_name, file_path):
    metrics = []
    state = 0
    try:
        while True:
            records = handler.getNextMsgBatch(topic_name=stats_topic_name)
            if records is not None:
                for _, consumerRecords in records.items():
                    for record in consumerRecords:
                        msg = handler.msg_from_record(record)
                        metrics.append(msg)
                if len(records.items()) == 0 and state:
                    print("empty" + str(datetime.now()))
                    state = 0
                elif len(records.items()) and state == 0:
                    state = 1
    except KeyboardInterrupt:
        print("Interrupted")
        pd.DataFrame(metrics).to_csv(file_path, encoding="utf-8", index=False)

        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
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

    interval = os.getenv("INTERVAL")
    batch_size = os.getenv("BATCH_SIZE")
    n_urls_days = os.getenv("N_URLS_DAYS")
    model_librairy = os.getenv("MODEL_LIBRAIRY")
    n_processes = os.getenv("N_PROCESSES")

    file_path = (
        "./metrics_"
        + model_librairy
        + "_"
        + n_processes
        + "_"
        + n_urls_days
        + "_"
        + interval
        + "_"
        + batch_size
        + ".csv"
    )
    print(file_path)
    print("Début de la lecture des messages")
    work(handler=handler, stats_topic_name=stats_topic_name, file_path=file_path)
