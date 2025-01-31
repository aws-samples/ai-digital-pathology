ARG REGION=us-east-1

# SageMaker PyTorch image for INFERENCE (See available images: https://github.com/aws/deep-learning-containers/blob/master/available_images.md)
FROM 763104351884.dkr.ecr.$REGION.amazonaws.com/pytorch-inference:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker

ENV PATH="/opt/ml/code:${PATH}"

RUN apt-get update && apt-get install -y \
    openslide-tools \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir openslide-python opencv-python pillow tqdm

RUN conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
RUN conda install -c rapidsai -c conda-forge cucim
RUN conda install timm h5py

# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
COPY /inference.py /opt/ml/model/code/inference.py
COPY /hoptimus_model_backbone.py /opt/ml/model/code/hoptimus_model_backbone.py

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/model/code

# this environment variable is used by the SageMaker PyTorch container to determine our program entry point
# for training and serving.
# For more information: https://github.com/aws/sagemaker-pytorch-container
ENV SAGEMAKER_PROGRAM inference.py

ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/model/code
ENV SAGEMAKER_HANDLER inference:predict_fn

ENV SAGEMAKER_MODEL_SERVER_TIMEOUT=3600
ENV SAGEMAKER_MODEL_SERVER_WORKERS=1
ENV TS_DECODE_INPUT_REQUEST=true
ENV TEMP=/tmp
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV SAGEMAKER_MAX_RETRY_DELAY=120
ENV SAGEMAKER_SERVING_TIME=3600

RUN pip freeze