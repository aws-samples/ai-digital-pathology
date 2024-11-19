ARG REGION=us-west-2

# SageMaker PyTorch image for training
FROM 763104351884.dkr.ecr.$REGION.amazonaws.com/pytorch-training:2.4.0-gpu-py311-cu124-ubuntu22.04-sagemaker

ENV PATH="/opt/ml/code:${PATH}"

RUN apt-get update && apt-get install -y \
    openslide-tools \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir openslide-python

RUN conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
RUN conda install -c rapidsai -c conda-forge cucim
RUN conda install timm h5py
    
COPY /preprocessing-code /opt/ml/code
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

ENV SAGEMAKER_PROGRAM generate_features.py
