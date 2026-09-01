# Installing ClassyFlow

ClassyFlow is a Nextflow pipeline. Nextflow handles the running; you only have
to make sure the Python packages are available, either on your machine or in a
container.

- [Before you start](#before-you-start)
- [Option 1: your own machine, with Python](#option-1-your-own-machine-with-python)
- [Option 2: your own machine, with Docker](#option-2-your-own-machine-with-docker)
- [Option 3: a SLURM cluster](#option-3-a-slurm-cluster)
- [Option 4: Google Batch](#option-4-google-batch)

---

## Before you start

You need:

- **Nextflow 23.04 or newer** — [install instructions](https://www.nextflow.io/docs/latest/install.html)
- **Java 17 or newer**, which Nextflow requires

Then get the code:

```bash
git clone https://github.com/dimi-lab/ClassyFlow.git
cd ClassyFlow
```

---

## Option 1: your own machine, with Python

> **Use Python 3.10 or 3.11.** The install fails on 3.12, because one of the
> dependencies has no build for it. This is the most common installation
> problem. Check with `python3 --version` before you start.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Check it worked:

```bash
nextflow run main.nf --help
nextflow run main.nf -profile local     # runs the bundled example data
```

The virtual environment must be active whenever you run the pipeline, because
the analysis scripts in `bin/` run with whichever `python` is on your PATH.

Some version pins in `requirements.txt` are deliberate and carry a comment
explaining why. Please do not relax them without reading the comment.

---

## Option 2: your own machine, with Docker

This avoids the Python version question completely. The image also contains
Java and Nextflow.

### Build the image

From the repository root:

```bash
docker build -f container/Dockerfile -t classyflow:latest .
```

This takes a while — expect 30 to 60 minutes the first time.

### Run

```bash
nextflow run main.nf -profile docker
```

The `docker` profile is defined in `conf/docker.config`. If you tagged the image
as something other than `classyflow:latest`, or you are pulling it from a
registry, change the one line there:

```groovy
process.container = 'classyflow:latest'
```

### If Docker runs out of memory

The `docker` profile does not request specific CPU or memory, so processes use
whatever Docker Desktop is configured to allow. If a step is killed, raise the
memory limit in Docker's settings, or lower the work the pipeline does — see
`target_splitting_size` and the model grid settings in
[parameters.md](parameters.md).

---

## Option 3: a SLURM cluster

```bash
nextflow run main.nf -profile slurm
```

Two things to adjust in `conf/slurm.config` before the first run:

1. **Queue names.** The file uses `med-n16-64g` and `sm-n2-8g-pre`, which are
   specific to the authors' cluster. Replace them with partitions that exist on
   yours, or delete the `queue` lines to use the default partition.
2. **Python.** This profile does not use a container, so the packages in
   `requirements.txt` must be available on the compute nodes — through a module,
   a shared virtual environment, or a conda environment you activate first.

The file also sets per-step CPU, memory and time requests. The heaviest steps
are the train/holdout split, feature elimination and the model search.

---

## Option 4: Google Batch

ClassyFlow runs on Google Cloud through the Batch service.

### 1. Prepare the project

- Enable the **Batch**, **Compute Engine**, **Cloud Storage**, **Artifact
  Registry** and **Cloud Logging** APIs.
- Create a **Cloud Storage bucket** for Nextflow's working files.
- Create an **Artifact Registry** repository for the container image.
- Give the service account these roles: *Batch Agent Reporter*, *Batch Admin*,
  *Logs Writer*, *Storage Admin*. Whoever builds the image also needs
  *Cloud Build Admin*.

Google's own [Nextflow on Batch guide](https://cloud.google.com/batch/docs/nextflow)
is a good check that the account works before you bring ClassyFlow into it.

### 2. Set the environment variables

`conf/gcp.config` reads its settings from the environment. Copy the template
and fill it in:

```bash
cp .envrc.template .envrc
# edit .envrc, then:
source .envrc
```

| Variable | What it is |
| --- | --- |
| `GCP_PROJECT_ID` | Your project ID. |
| `GCP_CONTAINER` | Full image path, e.g. `us-central1-docker.pkg.dev/PROJECT/REPO/classyflow:latest`. |
| `GCP_WORKDIR` | Bucket path for Nextflow's working files. Pass it with `-w` (see below). |
| `GCP_ARTIFACT_REPOSITORY` | Repository name. `container/cloud-build.sh` refuses to run without it. |
| `GCP_SERVICE_ACCOUNT` | Optional. A named service account instead of the default. |
| `GCP_NETWORK`, `GCP_SUBNETWORK` | Optional. For a custom VPC. |
| `GCP_USE_PRIVATE_ADDRESS` | `true` (default) gives the VMs no public IP. |

### 3. Build and push the image

With Cloud Build, from the repository root:

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
./container/cloud-build.sh
```

The script needs `GCP_ARTIFACT_REPOSITORY` and `GCP_CONTAINER` set, and submits
`container/build-config.yaml` as an asynchronous build. Watch it in the Cloud
Build console.

Or build locally and push it yourself:

```bash
docker build -f container/Dockerfile -t "$GCP_CONTAINER" .
docker push "$GCP_CONTAINER"
```

### 4. Run

```bash
nextflow run main.nf -profile gcp -w "$GCP_WORKDIR"
```

> **Pass `-w` explicitly.** `nextflow.config` sets `workDir = "work"`, a local
> path. Without `-w` pointing at your bucket, the run will not work on Batch.

Notes:

- The profile uses **Spot VMs** to keep costs down, and retries a step twice if
  it is preempted. For a run you cannot afford to lose, set `spot = false` in
  `conf/gcp.config`.
- Per-step machine sizes are set in `conf/gcp.config`. The largest request is
  16 CPU and 120 GB for the train/holdout split; if that is more than your quota
  allows, lower it and reduce `target_splitting_size` so less data is held in
  memory at once.
- Region is `us-central1` throughout. Change it in `conf/gcp.config` and in
  `container/cloud-build.sh` together.

---

## Next steps

- [README](../README.md) — how to prepare your data and read the results
- [parameters.md](parameters.md) — every setting and its default
