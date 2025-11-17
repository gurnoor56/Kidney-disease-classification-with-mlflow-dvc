# Kidney-disease-classification-with-mlflow-dvc


## WORKFLOWS

1.Update cofig.yaml
2.Update secrets.yaml[optional]
3.Update params.yaml
4.Update the entity
5.Update the configuration manager in src config
6.Update the components
7.Update the pipeline
8.Update the main.py
9.Update the dvc.yaml
10.app.py

# how to run
### STEPS:

clone the repository

```bash
git clone https://github.com/gurnoor56/Kidney-disease-classification-with-mlflow-dvc.git
```
### STEP 01:Create a conda environment after opening the repository
 
```bash
conda create -n cnncls python=3.9 -y
```

```bash
conda activate cnncls
```


### STEP 02:Install the Requirements
```bash
pip install -r requirements.txt
```






## MlFlow

[Documentation](https://mlfow.org/docs/latest/n=index.html)


##### cmd
- mlflow ui

## dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI = https://dagshub.com/gurnoor56/Kidney-disease-classification-with-mlflow-dvc.mlflow \
MLFLOW_TRACKING_USERNAME = gurnoor56 \
MLFLOW_TRACKING_PASSWORD = 26ed5d6606e6ca2401fee22c1f6245c29a4f805e \
pyhton script.py

Run this to export as env variable

```bash

set MLFLOW_TRACKING_URI = https://dagshub.com/gurnoor56/Kidney-disease-classification-with-mlflow-dvc.mlflow 

set MLFLOW_TRACKING_USERNAME = gurnoor56 

set MLFLOW_TRACKING_PASSWORD = 26ed5d6606e6ca2401fee22c1f6245c29a4f805e 

```

### DVC cmd

1.dvc init
2.dvc repro
3.dvc dag


## About MLflow & DVC

MLflow

 - Its Production Grade
 - Trace all of your expriements
 - Logging & taging your model


DVC 

 - Its very lite weight for POC only
 - lite weight expriements tracker
 - It can perform Orchestration (Creating Pipelines)



# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 566373416292.dkr.ecr.us-east-1.amazonaws.com/chicken

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app

