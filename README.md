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