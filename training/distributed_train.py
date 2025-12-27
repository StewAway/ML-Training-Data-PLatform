import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
from io import BytesIO
import boto3

# --- 1. Model Definition ---
class CTRModel(torch.nn.Module):
    def __init__(self, feature_count):
        super(CTRModel, self).__init__()
        self.linear = torch.nn.Linear(feature_count, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

# --- 2. DDP Setup (The "Huddle") ---
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'training-node-0')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    os.environ['GLOO_SOCKET_FAMILY'] = 'AF_INET'
    
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

# --- 3. Data Loading (MinIO S3) ---
def load_data_from_s3():
    s3 = boto3.client('s3', endpoint_url='http://minio:9000',
                      aws_access_key_id='admin', aws_secret_access_key='password123')
    
    objects = s3.list_objects_v2(Bucket='training-lake', Prefix='events/')
    parquet_files = [o['Key'] for o in objects.get('Contents', []) if o['Key'].endswith('.parquet')]
    
    if not parquet_files:
        print("[WARN] No parquet files found. Returning empty dataset.")
        return TensorDataset(torch.randn(0, 2), torch.randn(0))

    obj = s3.get_object(Bucket='training-lake', Key=parquet_files[0])
    df = pd.read_parquet(BytesIO(obj['Body'].read()))
    
    print(f"[*] Loaded {len(df)} rows of data from MinIO.")

    
    df['user_feat'] = df['user_id'].apply(lambda x: abs(hash(x)) % 10000).astype('float32')
    
    df['item_feat'] = df['item_id'].fillna("0").apply(lambda x: abs(hash(x)) % 10000).astype('float32')
    
    df['label'] = df['action'].apply(lambda x: 1.0 if x == 'click' else 0.0).astype('float32')

    features_data = df[['user_feat', 'item_feat']].values
    labels_data = df['label'].values

    features = torch.tensor(features_data)
    labels = torch.tensor(labels_data)
    
    return TensorDataset(features, labels)

# --- 4. Main Training Loop ---
def train(rank, world_size):
    setup(rank, world_size)
    
    dataset = load_data_from_s3()
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    model = CTRModel(feature_count=2)
    ddp_model = DDP(model)
    
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    
    start_time = time.time()
    total_samples = 0

    print(f"Rank {rank}: Starting training on {len(dataset)} rows...")

    for epoch in range(5):
        sampler.set_epoch(epoch)
        
        for data, target in loader:
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target.view(-1, 1))
            loss.backward()
            optimizer.step()
            
            total_samples += len(data)

    end_time = time.time()
    
    if rank == 0:
        duration = end_time - start_time
        # Throughput = (Samples per worker * N workers) / Time
        throughput = (total_samples * world_size) / duration
        
        print(f"\n--- DDP Benchmark Results ({world_size} Nodes) ---")
        print(f"Total Time: {duration:.2f} seconds")
        print(f"Cluster Speed: {throughput:.2f} samples/sec")
        print(f"----------------------------------------------\n")
        
        torch.save(model.state_dict(), "ctr_model.pth")
        print("Model saved to ctr_model.pth")

    dist.destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # These args are passed automatically by your 'train_wrapper.py'
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()
    
    train(args.rank, args.world_size)