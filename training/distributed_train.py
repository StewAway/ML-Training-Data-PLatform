import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
import pandas as pd
import boto3
from io import BytesIO
from datetime import datetime


class CTRModel(nn.Module):
    def __init__(self, feature_count):
        self.net = nn.Sequential(
            nn.Linear(feature_count, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)
    
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "training master"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def train(rank, world_size):
    setup(rank, world_size)

    s3 = boto3.client('s3', endpoint_url='http://minio:9000', aws_access_key_id='admin', aws_secret_access_key='password123')

    current_month = datetime.now().month
    current_year = datetime.now().year

    # Fetch the Parquet file from the Lake
    obj = s3.get_object(Bucket='training-lake', Key='events/data.parquet')
    df = pd.read_parquet(BytesIO(obj['Body'].read()))
    
    # Only train on datas that fit the current month of the year
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[(df['timestamp'].dt.month == current_month) & (df['timestamp'].dt.year == current_year)]

    if rank == 0:
        print(f"Training on {len(df)} rows from Month: {current_month}")


    # Load distributed datas
    features = torch.tensor(df[['user_id', 'item_id']].values, dtype=torch.float32)
    labels = torch.tensor(df['label'].values, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(features, labels)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    model = CTRModel(feature_count=2)
    ddp_model = DDP(model)

    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    # Training Loop
    for epoch in range(10):
        sampler.set_epoch(epoch) 
        for data, target in loader:
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = loss_fn(output, target)
            loss.backward() 
            optimizer.step()

    if rank == 0:
        torch.save(ddp_model.state_dict(), "ctr_model.pth")
        print("Distributed Training Complete.")

    dist.destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world-size", type=int)
    args = parser.parse_args()
    train(args.rank, args.world_size)

