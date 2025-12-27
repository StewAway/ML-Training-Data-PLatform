import time
import os
import subprocess
import boto3
import argparse

def wait_for_data(bucket, key):
    s3 = boto3.client("s3", endpoint_url="http://minio:9000", 
                      aws_access_key_id="admin", aws_secret_access_key="password123")
    print(f"[*] Training worker {os.environ.get('RANK')} waiting for data at {key}...")

    while True:
        try:
            response = s3.list_objects_v2(Bucket=bucket, Prefix=key)
            if 'Contents' in response and len(response['Contents']) > 0:
                print("[!] Data detected! Starting Distributed Training...")
                break
        except Exception:
            pass
            time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="distributed", choices=["distributed", "single"], help="Run in 'distributed' (DDP) or 'single' (standard) mode")
    args = parser.parse_args()
    wait_for_data("training-lake", 'events/')
    
    if args.mode == "distributed":
        # Launch DDP Script with Rank/World Size
        print(">>> Launching Distributed Training...")
        subprocess.run([
            "python3", "distributed_train.py", 
            "--rank", os.environ['RANK'], 
            "--world-size", os.environ['WORLD_SIZE']
        ])
    else:
        # Launch Standard Single-Process Script
        print(">>> Launching Single-Node Benchmark...")
        subprocess.run(["python3", "standard_train.py"])
