import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from io import BytesIO
import boto3

# --- Same Model Architecture ---
class CTRModel(nn.Module):
    def __init__(self, feature_count):
        super(CTRModel, self).__init__()
        self.linear = nn.Linear(feature_count, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

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

def train():
    print("[*] Starting SINGLE-PROCESS Benchmark...")
    
    dataset = load_data_from_s3()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = CTRModel(feature_count=2)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    start_time = time.time()
    total_samples = 0

    for epoch in range(5):
        for data, target in loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_samples += len(data)

    end_time = time.time()
    duration = end_time - start_time
    throughput = total_samples / duration

    print(f"\n--- 🐢 Single Node Baseline Results ---")
    print(f"Total Time:      {duration:.2f} seconds")
    print(f"Throughput:      {throughput:.2f} samples/sec")
    print(f"---------------------------------------\n")

if __name__ == "__main__":
    train()