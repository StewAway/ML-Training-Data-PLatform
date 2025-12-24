import argparse, json, sys, time, random, signal
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from kafka import KafkaProducer, KafkaAdminClient, KafkaProducer
from kafka.admin import NewTopic
from kafka.errors import UnknownTopicOrPartitionError, KafkaError
import events_pb2

@dataclass
class UserEvent:
    user_id: int
    item_id: int
    timestamp: str # ISO-8601 string
    action: str # view / click / add to cart

ACTIONS = ["view", "click", "add_to_cart"]
ACTIONS_WEIGHTS = [0.80, 0.15, 0.05]

def ensure_topic(admin, topic, partitions, replication_factor):
    try:
        admin.create_topics([NewTopic(name=topic, num_partitions=partitions, replication_factor=replication_factor)])
    except UnknownTopicOrPartitionError:
        print("[ensure_topic] topic already exists")
    except KafkaError as e:
        print(f"[ensure_topic] failed to create topic: {e}")
        sys.exit(1)

def build_producer(bootstrap_server):
    return KafkaProducer(
        bootstrap_servers=[bootstrap_server],
        retries=10,
        acks='all',
        key_serializer=lambda k: str(k).encode("utf-8")
    )

def generate_event(user_max, item_max):
    event = events_pb2.UserEvent()
    event.user_id = random.randint(1, user_max)
    event.item_id = random.randint(1, item_max)
    event.timestamp = datetime.now(timezone.utc).isoformat()
    event.action = random.choices(ACTIONS, ACTIONS_WEIGHTS)[0]
    return event

def send_event(producer, topic, event):
    event_bytes = event.SerializeToString()
    producer.send(topic, key=event.user_id, value=event_bytes)

def main():
    parser = argparse.ArgumentParser(description="Generates user events to be ingested into Kafka")
    parser.add_argument("--num-events", type=int, default=1000, help="Number of events to generate (-1 = forever)")
    parser.add_argument("--topic", type=str, default="user_events", help="Kafka topic to send events to")
    parser.add_argument("--user-max", type=int, default=100_000, help="Maximum number of users")
    parser.add_argument("--item-max", type=int, default=100_000, help="Maximum number of items")
    parser.add_argument("--bootstrap-server", type=str, default="localhost:9092", help="Kafka bootstrap server")
    parser.add_argument("--sleep-time", type=float, default=0.1, help="Seconds to sleep between events")
    parser.add_argument("--partitions", type=int, default=3, help="Topic partitions (if topic must be created)")
    parser.add_argument("--replication-factor", type=int, default=1, help="Topic replication factor")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (set for reproducible runs)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        print(f"[main] using seed={args.seed}")
    

    # 1) Ensure topic exist before ingesting
    admin = KafkaAdminClient(bootstrap_servers=[args.bootstrap_server])
    ensure_topic(admin, args.topic, args.partitions, args.replication_factor)
    print("[main] Topic ensured. Waiting 5 seconds for metadata to sync...")
    time.sleep(5)

    # 2) Build producer and ingest events
    producer = build_producer(args.bootstrap_server)

    running = True

    def stop(*_):
        nonlocal running
        running = False
    
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    print(f"[producer] bootstrap={args.bootstrap_server} topic={args.topic} num_events={args.num_events}")

    forever = (args.num_events == -1)
    sent = 0
    while running and (forever or sent < args.num_events):
        event = generate_event(args.user_max, args.item_max)
        print(f"[producer] event={event}")
        send_event(producer, args.topic, event)
        time.sleep(args.sleep_time)
        sent += 1
    print("[producer] flusing/closing...")
    producer.flush(timeout=10)
    producer.close(timeout=10)
    admin.close()
    print(f"[done] sent_total={sent}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Error] {e}", file=sys.stderr)
        sys.exit(1)