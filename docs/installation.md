 # ClassyFlow Installation Instructions

## Table of Contents
- [Requirements](#Requirements)
- [Option 1: Local Installation](#option-1-local-installation)
- [Option 2: Docker Container](#option-2-docker-container)
- [Option 3: Google Batch](#option-3-google-batch)


## Requirements
 
- [Nextflow](https://www.nextflow.io/) ≥ 23.04.0
- [Python](https://www.python.org/) ≥ 3.10
- [Docker](https://www.docker.com/) (optional)

### Python Dependencies
```
fpdf 1.7.2
numpy 1.23.5
matplotlib 3.8.0
dataframe-image 0.2.3
pandas 2.2.0
seaborn 0.13.1
xgboost 1.6.2
scipy 1.12.0
scikit-learn 1.4.0
```

## Option 1: Local Installation

```bash
# Clone the repository
git clone https://github.com/dimi-lab/ClassyFlow.git
cd ClassyFlow

# Create a virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install Python packages
pip install -r requirements.txt

# Test installation
nextflow run main.nf --help
```

## Option 2: Docker Container

### Build Docker image

1. Go to the repository root and run:

   ```sh
   docker build -f deployment/Dockerfile . -t classyflow:latest
   ```

   *Note that this may take ~30-60 minutes depending on your computer speed.*

### NextFlow config for local Docker

In `conf/docker.config`, update the `docker` profile to use the appropriate container name. 

If you used `classyflow:latest`:

   ```
   docker {
     process.container = 'classyflow:latest'
     docker.enabled = true
   }
   ```

The default CPU allocation is 8 CPUs, and Memory is 24 GB. (See the top of each `module` file). If your Docker runtime doesn't have this many resources, simply reduce. Eg:

   ```
     cpus 2
     memory '6 GB'
   ```

### Run Docker image

Go to the repository root and run:

   ```
   nextflow run main.nf -c nextflow.config -profile docker
   ```




## Option 3: Google Batch

Run ClassyFlow on Google Cloud using Google Batch service.  

### Perform Setup and Test Steps

Run a test Nextflow job to verify your service account permissions for Google Batch and GCP resources.  This will ensure your `nextflow.config` file is setup correctly to use Google Batch as a backend provider.  The test job will run a simple `hello` process to verify the setup.  


1. Clone this repo to a local directory eg on a GCE instance (VM).
2. Install Nextflow. Follow [Google's instructions to get version 23.04.1](https://cloud.google.com/batch/docs/nextflow#before-you-begin). Put the resulting `nextflow` binary on your path, or in the repository root directory.
3. Verify Logging, Cloud Storage, Google Batch and GCE APIs are enabled on your GCP Project.
4. Make sure you have a bucket created for Nextflow job files.
5. Run a test job (hello NF) to verify your [GCP service account permissions](https://cloud.google.com/batch/docs/nextflow) and `nextflow.config` file is setup correctly to use Google Batch as a backend provider.

   - You need at least these for your default GCE service account: Logs Writer, Storage Admin, Batch Admin, Batch Agent Reporter.
   - Your account (or whichever creates the build) also needs Cloud Build Admin.

6. Follow the instructions in the [User Guide](user-guide.md) to set up your ClassyFlow run
7. Setup your `conf/gcp.config` and `main.nf` files to use the appropriate Google Batch resources (see below)
8. Run ClassyFlow using the provided command `./nextflow run main.nf -profile gcp` from the ClassyFlow repository root directory.

### Build Docker Image with Cloud Build 

Use GCP Cloud Build to build a ClassyFlow container and push it to GCP Artifact Registry.

1. Create an Artifact Registry in your GCP project named `images`, note the region and project ID.

2. Auth to GCP Artifact Registry (using your region):

   `gcloud auth configure-docker us-central1-docker.pkg.dev`

3. Update cloud-build.yaml with your region & project name.

4. Submit a Cloud Build job; from the root repository directory:

   ````sh
   gcloud builds submit --config cloud-build.yaml --machine-type=E2_HIGHCPU_8
   ````

### Nextflow Config for Google Batch

Update the `conf/gcp.config` file with the correct region and project id and bucket.

Your `main.nf` file should have config information at the task level.  General syntax shown below.

```
process myTask {
    cpus 8
    memory '40 GB'

    """
    your_command --here
    """
}

process anotherTask {
    machineType 'n1-highmem-8'

    """
    your_command --here
    """
}
```

NOTES for Enterprise Customers:  
- Specify a named (project) GCP service account in the `conf/gcp.config` file
- Specify a named (project) GCP network and subnet in the `conf/gcp.config` file
- Specify 'no external IP' for the GCP VM instances in the `conf/gcp.config` file
- Specify both CPU and MEMORY requirements for **each process** in the main.nf file to request a specific machine type