Go to the Apache Software Foundation download center, find the latest version and download it:
wget https://downloads.apache.org/kafka/2.6.0/kafka_2.13-2.6.0.tgz

Decompress the archive:
tar xvzf kafka_2.13-2.6.0.tgz

Create an environment variable:
export KAFKA_HOME=/yourPath/kafka_2.13-2.6.0

Check Your Install
If the ZooKeeper daemon is not active and running, you can use a convenience script packaged with the Kafka distribution to get a quick-and-dirty single-node ZooKeeper instance:
$KAFKA_HOME/bin/zookeeper-server-start.sh $KAFKA_HOME/config/zookeeper.properties &

Start the Kafka broker:
$KAFKA_HOME/bin/kafka-server-start.sh $KAFKA_HOME/config/server.properties &

Let's create a topic named "test" with a single partition and only one replica:
$KAFKA_HOME/bin/kafka-topics.sh --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic test

Send some messages: Kafka comes with a command-line client that will take input from a file or from standard input and send it out as messages to the Kafka cluster. By default, each line will be sent as a separate record with a String-typed value and a null key. Run the producer and then type a few messages into the console to send to the Kafka server:
$KAFKA_HOME/bin/kafka-console-producer.sh --bootstrap-server localhost:9092 --topic test
>Hello world!
>"Paths are made by walking" (Franz Kafka)

Either end the producer process with a Ctrl+C or open a new terminal.

Start a consumer: Kafka also has a command-line consumer that will dump out messages to standard output. In another terminal type:
$KAFKA_HOME/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic test --from-beginning

If you see the 2 lines input previously, congratulations, your setup seems fine!

Let's shutdown properly before moving to the next activity:
$KAFKA_HOME/bin/kafka-topics.sh --delete --bootstrap-server localhost:9092 --topic test
$KAFKA_HOME/bin/kafka-server-stop.sh
