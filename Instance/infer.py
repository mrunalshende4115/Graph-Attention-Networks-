import json
import joblib
import boto3
import torch
import numpy as np
import torch.nn as nn

from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import GATConv

BUCKET = "mrunal-ids2025-data"

s3 = boto3.client("s3")

sns = boto3.client(
    "sns",
    region_name="us-east-1"
)

TOPIC_ARN = "arn:aws:sns:us-east-1:565713102632:IDSAlerts"

cloudwatch = boto3.client(
    "cloudwatch",
    region_name="us-east-1"
)


class GATNet(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels
    ):
        super().__init__()

        self.gat1 = GATConv(
            in_channels,
            hidden_channels,
            heads=4
        )

        self.gat2 = GATConv(
            hidden_channels * 4,
            out_channels,
            heads=1
        )

    def forward(
        self,
        x,
        edge_index
    ):

        x = torch.relu(
            self.gat1(
                x,
                edge_index
            )
        )

        x = self.gat2(
            x,
            edge_index
        )

        return x


# Load artifacts

with open("features.json") as f:
    FEATURES = json.load(f)

with open("label_map.json") as f:
    LABEL_MAP = json.load(f)

scaler = joblib.load(
    "scaler.pkl"
)

model = GATNet(
    in_channels=len(FEATURES),
    hidden_channels=64,
    out_channels=len(LABEL_MAP)
)

model.load_state_dict(
    torch.load(
        "gat_model_balanced.pth",
        map_location="cpu"
    )
)

model.eval()

print("=" * 60)
print("MODEL LOADED")
print("=" * 60)

# Read inference-ready files

response = s3.list_objects_v2(
    Bucket=BUCKET,
    Prefix="inference-ready/"
)

if "Contents" not in response:
    print("No inference files found")
    exit()

objects = [
    obj
    for obj in response["Contents"]
    if obj["Key"].endswith(".json")
]

print(
    f"Total inference-ready files: {len(objects)}"
)

if len(objects) < 2:
    print(
        "Need at least 2 files to build graph"
    )
    exit()

# Sort by S3 last modified timestamp

objects = sorted(
    objects,
    key=lambda x: x["LastModified"]
)

# Take newest 25

json_files = [
    obj["Key"]
    for obj in objects[-25:]
]

print(
    f"Using latest {len(json_files)} files"
)

print("\nFiles being analyzed:")

for f in json_files:
    print(f)

rows = []
records = []
file_names = []

# Load records

for key in json_files:

    item = s3.get_object(
        Bucket=BUCKET,
        Key=key
    )

    record = json.loads(
        item["Body"].read()
    )

    row = []

    for feature in FEATURES:

        value = record.get(
            feature,
            0
        )

        try:
            value = float(value)

        except Exception:
            value = 0

        row.append(value)

    rows.append(row)
    records.append(record)
    file_names.append(key)

print(
    f"\nLoaded {len(rows)} records"
)

print("\nActual labels in batch:")

for r in records[:10]:
    print(
        r.get(
            "newlabel",
            "Unknown"
        )
    )

X = np.array(rows)

print(
    "Feature Matrix Shape:",
    X.shape
)

# Scale features

X = scaler.transform(X)

# Build KNN graph

nbrs = NearestNeighbors(
    n_neighbors=min(11, len(X))
)

nbrs.fit(X)

distances, indices = nbrs.kneighbors(X)

edge_index_list = []

for i in range(len(X)):

    for j in indices[i]:

        if i != j:

            edge_index_list.append(
                [i, j]
            )

edge_index = torch.tensor(
    edge_index_list,
    dtype=torch.long
).t().contiguous()

print(
    "Graph Nodes:",
    len(X)
)

print(
    "Graph Edges:",
    edge_index.shape[1]
)

# Inference

x = torch.tensor(
    X,
    dtype=torch.float
)

with torch.no_grad():

    logits = model(
        x,
        edge_index
    )

probs = torch.softmax(
    logits,
    dim=1
)

preds = probs.argmax(
    dim=1
).tolist()

print("\nConfidence Check")

for i in range(
    min(10, len(preds))
):

    pred = preds[i]

    conf = probs[
        i
    ].max().item()

    print(
        f"Node {i} -> "
        f"{LABEL_MAP[str(pred)]} "
        f"({conf:.4f})"
    )

print("\nPrediction Distribution")

unique, counts = np.unique(
    preds,
    return_counts=True
)

for u, c in zip(
    unique,
    counts
):

    print(
        f"{LABEL_MAP[str(u)]} -> {c}"
    )

print("\nDetailed Results")

for i, pred_class in enumerate(preds):

    actual_label = records[i].get(
        "newlabel",
        "Unknown"
    )

    prediction_label = LABEL_MAP[
        str(pred_class)
	]

    cloudwatch.put_metric_data(
        Namespace="IDS2025",
        MetricData=[
          {
            "MetricName": prediction_label,
            "Value": 1,
            "Unit": "Count"
          }
      ]
    )
    

    confidence = float(
        probs[i].max().item()
    )

# SNS Alert

    if prediction_label != "Normal":

    	sns.publish(
            TopicArn=TOPIC_ARN,
            Subject="IDS ATTACK DETECTED",
            Message=f"""
Attack Detected

File:
{file_names[i]}

Actual Label:
{actual_label}

Predicted Label:
{prediction_label}

Confidence:
{confidence:.4f}
"""
    )

    print(
        f"ALERT SENT: {prediction_label}"
    )

    prediction = {
        "source_file": file_names[i],
        "actual_label": actual_label,
        "prediction_class": int(pred_class),
        "prediction_label": prediction_label,
        "confidence": confidence,
        "graph_nodes": len(X),
        "graph_edges": int(
            edge_index.shape[1]
        )
    }

    output_key = (
        "predictions/"
        + file_names[i].split("/")[-1]
    )

    s3.put_object(
        Bucket=BUCKET,
        Key=output_key,
        Body=json.dumps(
            prediction,
            indent=2
        )
    )

    print("=" * 70)

    print(
        "FILE:",
        file_names[i]
    )

    print(
        "ACTUAL:",
        actual_label
    )

    print(
        "PREDICTED:",
        prediction_label
    )

    print(
        "CONFIDENCE:",
        round(confidence, 4)
    )

print("\nDone")
