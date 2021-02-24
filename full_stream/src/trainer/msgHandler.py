from kafka import KafkaConsumer, KafkaProducer, TopicPartition
from kafka.consumer.fetcher import ConsumerRecord

import json

from typing import Dict

__all__ = ["MsgHandler"]


class MsgHandler:
    def __init__(self, broker_list, consumers_attr=[], producer=False):
        """
        link to all consumer [{"topic":topice_name, "kafka_group":kafka_group}]
        """
        self.kafka_api_version = (2, 3, 0)
        self.consumers = {}
        self.producer = None
        for consumer_attr in consumers_attr:

            cascadeConsumerProperties = {
                "bootstrap_servers": broker_list,
                "auto_offset_reset": "earliest",
                "group_id": consumer_attr["kafka_group"],
            }

            # init Consumer
            self.consumers[consumer_attr["topic"]] = KafkaConsumer(
                **cascadeConsumerProperties, api_version=self.kafka_api_version
            )
        if producer:
            producerProperties = {"bootstrap_servers": broker_list}

            # init Producer
            self.producer = KafkaProducer(
                **producerProperties, api_version=self.kafka_api_version
            )

    def subscribeTopics(self):
        for topice_name, consumer in self.consumers.items():
            consumer.subscribe(topice_name)

    def closeAll(self) -> None:
        for topice_name, consumer in self.consumers.items():
            consumer.close()
        if self.producer:
            self.producer.close()

    def getNextMsgBatch(
        self, topic_name: str, timeout: int = 1000
    ) -> Dict[TopicPartition, ConsumerRecord]:
        return self.consumers[topic_name].poll(timeout)

    def sendMsg(self, topic_name: str, key: int, msg: dict) -> None:
        self.producer.send(topic_name, key=key, value=msg)

    @staticmethod
    def msg_from_record(record: ConsumerRecord) -> dict:
        return json.loads(record.value.decode("utf-8"))

    @staticmethod
    def key_from_record(record: ConsumerRecord) -> int:
        return int(record.key.decode("utf-8"))

    @staticmethod
    def serializer(valueToSerialize):
        return json.dumps(valueToSerialize).encode("utf-8")
