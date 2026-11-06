import json
import boto3

def lambda_handler(event, context):
    glue = boto3.client('glue')

    print("Event received:", json.dumps(event))

    # Start your Glue ETL job
    response = glue.start_job_run(
        JobName='ids2025_etl_job_new'
    )

    print("Glue job started:", response['JobRunId'])

    return {
        'statusCode': 200,
        'body': json.dumps('Glue ETL triggered successfully!')
    }
