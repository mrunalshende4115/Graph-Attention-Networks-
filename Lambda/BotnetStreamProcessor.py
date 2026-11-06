import boto3
import csv
import io
import json
import uuid

s3 = boto3.client("s3")

BUCKET = "mrunal-ids2025-data"

DATA_PREFIX = "processed/output_ml_ready/"
OFFSET_FILE = "scripts/offset.json"

BATCH_SIZE = 10


def get_offset():

    try:

        obj = s3.get_object(
            Bucket=BUCKET,
            Key=OFFSET_FILE
        )

        offset_data = json.loads(
            obj["Body"].read()
        )

        return offset_data.get(
            "current_row",
            0
        )

    except:

        return 0


def save_offset(offset):

    s3.put_object(
        Bucket=BUCKET,
        Key=OFFSET_FILE,
        Body=json.dumps({
            "current_row": offset
        })
    )


def lambda_handler(event, context):

    current_offset = get_offset()

    print(
        "Current Offset:",
        current_offset
    )

    response = s3.list_objects_v2(
        Bucket=BUCKET,
        Prefix=DATA_PREFIX
    )

    files = response.get(
        "Contents",
        []
    )

    if not files:

        return {
            "statusCode": 404,
            "message": "No dataset found"
        }

    latest_file = max(
        files,
        key=lambda x: x["LastModified"]
    )

    dataset_key = latest_file["Key"]

    print(
        "Using Dataset:",
        dataset_key
    )

    obj = s3.get_object(
        Bucket=BUCKET,
        Key=dataset_key
    )

    csv_content = (
        obj["Body"]
        .read()
        .decode("utf-8")
    )

    reader = list(
        csv.DictReader(
            io.StringIO(csv_content)
        )
    )

    total_rows = len(reader)

    print(
        "Total Rows:",
        total_rows
    )

    start = current_offset

    end = min(
        start + BATCH_SIZE,
        total_rows
    )

    rows = reader[start:end]

    streamed = 0

    for row in rows:

        key = (
            "streaming/flow_"
            + str(uuid.uuid4())
            + ".json"
        )

        s3.put_object(
            Bucket=BUCKET,
            Key=key,
            Body=json.dumps(row)
        )

        streamed += 1

    new_offset = end

    if new_offset >= total_rows:

        new_offset = 0

    save_offset(
        new_offset
    )

    return {
        "statusCode": 200,
        "streamed_rows": streamed,
        "previous_offset": start,
        "new_offset": new_offset
    }