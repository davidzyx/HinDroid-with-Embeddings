# HinDroid-with-Embeddings

![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/davidzz/hindroid-xl)

## Overview

Malware detection for android applications is an expanding field with the introduction of the [Hindroid](https://www.cse.ust.hk/~yqsong/papers/2017-KDD-HINDROID.pdf) model (DOI:[10.1145/3097983.3098026](https://doi.org/10.1145/3097983.3098026)). It proposes a method that transforms the semantic relationships between Android applications and their decompiled source code to a Heterogeneous Information Network (HIN) and uses similarities from various meta-paths between apps to construct a model for malware classification. To further explore the field, we aim to extend the HinDroid model to improve the accuracy in specific subsets of the AMD dataset. Our effort will be focused on finding better representations for both apps as well as APIs and discovering methods to incorporate them as additional features in a new model. In the meantime, we plan to evaluate how the proposed model captures the features that are relevant to the classification task and compare to that of the HinDroid baseline. Our contributions can be utilized in systems where the analysis of malware and interpretable features are more important than mere detection.

## Usage
Docker image on Docker Hub:
[davidzyx/hindroid-xl](https://hub.docker.com/repository/docker/davidzz/hindroid)

On Datahub, create a pod using a custom image with 4 CPU and 32 GB of memory. If you use another configuration, use at least 6GB memory per CPU.
```bash
launch.sh -i davidzz/hindroid-xl -c 4 -m 24
```

Modify the config file located in `config/data-params.json`. If you want to run a test drive, use `config/test-params.json`. Put either `data` or `test` as the first argument.

The HinDroid baseline uses the driver file `run.py` with 3 targets: `ingest`, `process`, and `model`. Put each target space-separated as arguments in the call. To run the whole pipeline, use
```bash
python run.py data ingest process model
```

`process` target will save `.npz` files in `data/processed/` for generating various embeddings.

## Results

HinDroid:

| metapath | train_acc | test_acc | F1     | TP    | FP   | TN    | FN   |
|----------|-----------|----------|--------|-------|------|-------|------|
| AA       | 1.0000    | 0.9561   | 0.9562 | 158   | 10   | 147   | 4    |
| APA      | 1.0000    | 0.9373   | 0.9412 | 155   | 14   | 145   | 6    |
| ABA      | 0.9149    | 0.8558   | 0.8671 | 147   | 27   | 130   | 19   |
| APBPA    | 0.9040    | 0.8339   | 0.8408 | 140   | 32   | 126   | 22   |

Reduced:

| metapath | train_acc | test_acc | F1     | TP    | FP   | TN    | FN   |
|----------|-----------|----------|--------|-------|------|-------|------|
| AA       | 1.0000    | 0.9561   | 0.9562 | 158   | 10   | 147   | 4    |
| APA      | 1.0000    | 0.9373   | 0.9412 | 155   | 14   | 145   | 6    |
| ABA      | 0.9149    | 0.8558   | 0.8671 | 147   | 27   | 130   | 19   |
| APBPA    | 0.9040    | 0.8339   | 0.8408 | 140   | 32   | 126   | 22   |

### Class Labels Separation

![Labels](https://i.imgur.com/cdFOD6m.jpg)
