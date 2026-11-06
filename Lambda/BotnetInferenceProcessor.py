import json
import boto3
import uuid
import math
from datetime import datetime

s3 = boto3.client("s3")

BUCKET = "mrunal-ids2025-data"

BATCH_SIZE = 25


def clean_record(record):

    cleaned = {}

    for key, value in record.items():

        if key in ["label", "newlabel"]:
            cleaned[key] = value
            continue

        if value is None or value == "":
            cleaned[key] = 0
            continue

        try:

            num = float(value)

            if math.isinf(num):
                num = 0

            num = max(
                min(num, 1000000),
                -1000000
            )

            cleaned[key] = num

        except Exception:
            cleaned[key] = 0

    return cleaned


def lambda_handler(event, context):

    try:

        response = s3.list_objects_v2(
            Bucket=BUCKET,
            Prefix="streaming/"
        )

        if "Contents" not in response:

            return {
                "statusCode": 200,
                "body": json.dumps({
                    "status": "No files found"
                })
            }

        all_files = response["Contents"]

        print(
            f"Total files found: {len(all_files)}"
        )

        files = all_files[:BATCH_SIZE]

        print(
            f"Processing batch size: {len(files)}"
        )

        processed = 0

        for file in files:

            key = file["Key"]

            if not key.endswith(".json"):
                continue

            try:

                print(
                    f"Processing: {key}"
                )

                obj = s3.get_object(
                    Bucket=BUCKET,
                    Key=key
                )

                content = (
                    obj["Body"]
                    .read()
                    .decode("utf-8")
                )

                record = json.loads(
                    content
                )

                cleaned_record = clean_record(
                    record
                )

                cleaned_file = (
                    "inference-ready/cleaned_"
                    + datetime.utcnow().strftime(
                        "%Y%m%d_%H%M%S"
                    )
                    + "_"
                    + str(uuid.uuid4())
                    + ".json"
                )

                s3.put_object(
                    Bucket=BUCKET,
                    Key=cleaned_file,
                    Body=json.dumps(
                        cleaned_record
                    )
                )

                archive_key = key.replace(
                    "streaming/",
                    "processed-streaming/"
                )

                s3.copy_object(
                    Bucket=BUCKET,
                    CopySource={
                        "Bucket": BUCKET,
                        "Key": key
                    },
                    Key=archive_key
                )

                s3.delete_object(
                    Bucket=BUCKET,
                    Key=key
                )

                processed += 1

            except Exception as e:

                print(
                    f"Error processing {key}: {str(e)}"
                )

        result = {
            "timestamp":
                datetime.utcnow().isoformat(),
            "files_processed":
                processed,
            "total_found":
                len(all_files),
            "batch_size":
                len(files),
            "status":
                "Cleaning completed"
        }

        print(result)

        return {
            "statusCode": 200,
            "body": json.dumps(result)
        }

    except Exception as e:

        print(
            f"Lambda failed: {str(e)}"
        )

        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e)
            })
        }