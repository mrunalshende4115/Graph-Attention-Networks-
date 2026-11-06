import json
import joblib
import boto3
import torch
import numpy as np
import torch.nn as nn

from torch_geometric.nn import GATConv

BUCKET = "mrunal-ids2025-data"

s3 = boto3.client("s3")


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

response = s3.list_objects_v2(
    Bucket=BUCKET,
    Prefix="inference-ready/"
)

objects = [
    obj
    for obj in response["Contents"]
    if obj["Key"].endswith(".json")
]

objects = sorted(
    objects,
    key=lambda x: x["LastModified"]
)

latest_files = objects[-25:]

print(
    f"Testing {len(latest_files)} newest files"
)

correct = 0

for obj in latest_files:

    key = obj["Key"]

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
        except:
            value = 0

        row.append(value)

    X = np.array([row])

    X = scaler.transform(X)

    x = torch.tensor(
        X,
        dtype=torch.float
    )

    # SELF LOOP
    edge_index = torch.tensor(
        [[0],
         [0]],
        dtype=torch.long
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

    pred_class = probs.argmax(
        dim=1
    ).item()

    confidence = probs.max().item()

    predicted = LABEL_MAP[
        str(pred_class)
    ]

    actual = record.get(
        "newlabel",
        "Unknown"
    )

    if predicted == actual:
        correct += 1

    print(
        "-" * 60
    )

    print(
        f"FILE: {key.split('/')[-1]}"
    )

    print(
        f"ACTUAL: {actual}"
    )

    print(
        f"PREDICTED: {predicted}"
    )

    print(
        f"CONFIDENCE: {confidence:.4f}"
    )

print("\n" + "=" * 60)

print(
    f"CORRECT: {correct}/{len(latest_files)}"
)

accuracy = (
    correct /
    len(latest_files)
) * 100

print(
    f"ACCURACY: {accuracy:.2f}%"
)
