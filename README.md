# ReAdvisor Response Generation Module Deliverables

This repository contains the code base for automaric review response generation developed in the ReAdvisor Project.

## Organisation

Included in this repo are all the scripts required for:
- data preprocessing
- model training
- automatic model evaluation and preparation for Prodigy
- model deployment (with Docker)

```
.
├── data_prep       # python/bash scripts for data preprocessing from re:spo DB to mBART input format
│   └── utils_pkg   # utility functions used in main data preprocessing scripts
├── evaluation      # python scripts for running evaluations described in the documentation
├── fastapi_app     # self-contained code for building and running docker container
│   └── app 
│     ├── egs       # examples of expected input/output formats for API
│     └── src       # NLP source code
│         └── mbart_hospo_respo     # source code for model training and inference
```

## Models

The actual model files model files are **not included** in this repo due to size limitations. They will be shipped separately.
Once you have the models, put them in the appropriate directories as follows (use symlinks).

| models dir    | repo dir                                      |
|---------------|-----------------------------------------------|
| `classifiers` | `evaluation/models`                           |
| `flair`       | `data_prep/models`                            |
| `spacy`       | `data_prep/models` & `fastapi_app/app/models` |
| `mbart`       | `fastapi_app/app/models`                      |


## Setup working environment

```
conda create --name respondelligent python=3.8.5
conda activate respondelligent
conda install cudatoolkit=your-cuda-version # if on GPU instance
pip install -r requirements.txt
```

## Steps
- For each step (e.g. preprocessing, training, evaluation), see the READMEs in the relevant directories or the project documentation.

---
