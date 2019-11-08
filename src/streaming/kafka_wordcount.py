"""
kafka wordcount
run zookeeper: ./bin/zkServer.sh start
run kafka: bin/kafka-server-start.sh config/server.properties
创建topic: ./bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic wordsendertest
查看topic: ./bin/kafka-topics.sh --list --zookeeper localhost:2181
producer生产数据：./bin/kafka-console-producer.sh --broker-list localhost:9092 --topic wordsendertest
查看数据(打开另一个终端)：./bin/kafka-console-consumer.sh --bootstrap-server  127.0.0.1:9092 --topic wordsendertest --from-beginning

run: bin/spark-submit --jars \
      jars/spark-streaming-kafka-0-8-assembly_2.11-2.4.0.jar \
      /Users/t/python/spark-learning/src/streaming/kafka_wordcount.py \
      localhost:2181 wordsendertest
"""


import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: kafka_wordcount.py <zk> <topic>", file=sys.stderr)
        sys.exit(-1)

    sc = SparkContext(appName="StreamingKafkaWordCount")
    ssc = StreamingContext(sc, 1)

    zkQuorum, topic = sys.argv[1:]
    kvs = KafkaUtils.createStream(ssc, zkQuorum, "spark-streaming-consumer", {topic: 1})
    lines = kvs.map(lambda x: x[1])
    counts = lines.flatMap(lambda line: line.split(" ")) \
        .map(lambda word: (word, 1)) \
        .reduceByKey(lambda a, b: a + b)
    counts.pprint()

    ssc.start()
    ssc.awaitTermination()