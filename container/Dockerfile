FROM ubuntu:24.04

# Add the deadsnakes PPA to get Python 3.11
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update

RUN apt-get install -y --no-install-recommends \
    apt-utils \
    chromium-driver \
    curl \
    git \
    openjdk-21-jre \
    pandoc \
    pkg-config \
    vim

# Set system default python to 3.11 (10 == priority, not sure this matters much)
RUN apt-get remove -y python3.12 && \
    apt-get autoremove -y && \
    apt-get install -y --no-install-recommends python3.11-full python3.11-dev python3.11-distutils python3.11-venv && \
    python3.11 -m ensurepip && \
    python3.11 -m pip install --upgrade pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --remove python /usr/bin/python2

# Python Packages
ADD requirements.txt .
RUN python -m pip install -r requirements.txt

# Install Nextflow
RUN curl -s https://get.nextflow.io | bash && \
    mv nextflow /usr/local/bin

# Cleanup to reduce image size
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

COPY bin/ /app/bin
COPY conf/ /app/conf
COPY data/ /app/data
COPY assets/ /app/assets
COPY main.nf /app/main.nf
COPY modules/ /app/modules
COPY nextflow.config /app/nextflow.config
