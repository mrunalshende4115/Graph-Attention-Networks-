Graph Attention Networks for Detecting
Coordinated and Botnet-Style Attacks in AWS-Style
Virtual Cloud Networks Using VPC Flow Logs

Overview

This project implements a cloud-native Intrusion Detection System (IDS) using Graph Neural Networks (GNNs) and AWS services. The system captures AWS VPC Flow Logs, preprocesses them using AWS Lambda, performs attack classification using a Graph Attention Network (GAT) model deployed on Amazon EC2, stores predictions in Amazon S3, publishes attack alerts using Amazon SNS, and visualizes security metrics in Amazon CloudWatch.

The objective is to detect network attacks in near real-time and provide automated monitoring and alerting.

Project Features

Automated VPC Flow Log collection
Serverless preprocessing with AWS Lambda
Graph-based attack classification using GAT
Real-time prediction storage
Email notifications through SNS
CloudWatch monitoring dashboard
Cloud-native architecture
End-to-end AWS deployment

AWS Services Used

Amazon EventBridge  Event-driven workflow automation
Amazon S3	Data storage
AWS Lambda	Data preprocessing and cleaning
Amazon EC2	GAT inference server
Amazon SNS	Attack notifications
Amazon CloudWatch	Monitoring dashboard

AWS Resources Created

S3 Bucket: mrunal-ids2025-data
IDS2025.csv
Folder Structure
streaming/
processed-streaming/
inference-ready/
predictions/
models/
processed/
scripts/

Lambda Functions
BotnetInferenceProcessor
BotnetStreamProcessor
BotnetStreamProcessor

EC2 Instance: 
gat_project/
features.json           
infer.py.save   
single_flow_infer.py
gat_model.pth           
label_map.json  
test_load.py
gat_model_balanced.pth  
sample.json     
tmp
infer.py                
scaler.pkl

SNS Topic: IDSAlerts
Email


Artifacts Saved:
scaler.pkl
features.json
label_map.json
gat_model_balanced.pth

Notebook: GAT_Training.ipynb
