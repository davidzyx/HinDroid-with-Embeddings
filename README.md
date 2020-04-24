# HinDroid-with-Embeddings

Experiments on improving the HinDroid model

![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/davidzz/hindroid-xl)

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
