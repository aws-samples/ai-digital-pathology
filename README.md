# Fine-tuning Vision Foundation-Models for Digital Pathology

This repository showcases how to fine-tune [H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0) on patch-level and whole-slide level tasks for digital pathology using AWS:
The different steps are:
 * Deployment of the **AWS Infrastructure**
 * Downloading the training datsets
 * Fine-tuning of patch-level models:
   * MHIST classification: This task uses the MHIST dataset for binary classification of colorectal polyps. The training script is located at `train/train_mhist.py`.
   * Lizard segmentation: This task uses the Lizard dataset for colonic nuclear instance segmentation. The training script is located at `train/train_lizard.py`.
 * Fine-tuning of WSI-level models:
   * Whole-Slide level feature extraction pipeline: This task uses a custom Docker image for tiling and feature extraction from whole-slide images. The Docker image is built and pushed using the `preprocessing/build_and_push.sh` script.
   * Whole-Slide level WSI prediction using Multiple Instance Learning : This task uses the features extracted from the TCGA-COAD dataset to predict MSI status at the whole-slide level. The training script is located at `train/train_msi_tcga.py`


The `train.ipynb` notebook in this repository demonstrates how to start the different training jobs for each task. It includes code for:
  * Setting up the SageMaker session and defining EFS inputs
  * Configuring and launching training jobs for each task
  * Specifying hyperparameters and metric definitions
  * Using custom Docker images for feature extraction

To use this notebook:
  * Open the `train.ipynb` notebook in your SageMaker instance
  * Modify the EFS file system ID, subnet, and security group IDs as needed
  * Run the cells for the desired tasks

  
# Initialization Steps: 

1. Deploy the required Infrastructure using the provided CloudFormation stack: ```aws cloudformation deploy --template-file infra/infra-stack.yml --stack-name EFS-SM --profile=pidemal+11-Admin --capabilities CAPABILITY_IAM```. As part of the deployment, the following infrastructure is being deployed: 
   * **Networking Infrastructure**: A VPC with a public and a private subnet. The Public Subnet has internet access, but the private subnet does not.
   * An **Elastic File System** with LifeCycle Configuration : The EFS is hosted on a the private subnet
   * A **SageMaker Notebook Instance** with a LifeCycle Configuration that mounts the EFS on the SageMaker Instance at start time. 
   * A **SageMaker Notebook Instance** Execution Role that has access to the EFS and full SageMaker permissions. 
  
2. Retrieve the `EFSFileSystemId` and the `SageMakerNotebookInstanceName` that was created in the previous time. You can retrieve these values in the console or by running the following command: 
```pathology-blogpost % aws cloudformation describe-stacks --stack-name EFS --query 'Stacks[0].Outputs' --profile=pidemal+11-Admin```

3. At this stage, you are ready to mount your EFS on your SageMaker notebook. For full details, you can refer to the following [blogpost](https://aws.amazon.com/blogs/machine-learning/mount-an-efs-file-system-to-an-amazon-sagemaker-notebook-with-lifecycle-configurations/)

The following command are an example of how to mount such an instance 
```bash
cd SageMaker
sudo mkdir efs
sudo mount -t nfs -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 fs-0f634af3c0c47b63d.efs.us-west-2.amazonaws.com:/ efs
https://aws.amazon.com/blogs/machine-learning/mount-an-efs-file-system-to-an-amazon-sagemaker-notebook-with-lifecycle-configurations/
```

4. At this point, we are ready to download the training datasets that we will use to train our model on these specific pathology downstream tasks.

   * Downstream task : Binary Classification of predominant histological pattern in WSI of colorectal polyps. For this task, we use the publicly available [MHIST dataset](https://bmirds.github.io/MHIST/). MHIST is a binary task which comprises of 3,152 hematoxylin and eosin (H&E)-stained Formalin Fixed Paraffin-Embedded (FFPE) fixed-size images (224 by 224 pixels) of colorectal polyps from the Department of Pathology and Laboratory Medicine at Dartmouth-Hitchcock Medical Center (DHMC).

    The tissue classes are: Hyperplastic Polyp (HP), Sessile Serrated Adenoma (SSA). This classification task focuses on the clinically-important binary distinction between HPs and SSAs, a challenging problem with considerable inter-pathologist variability. HPs are typically benign, while sessile serrated adenomas are precancerous lesions that can turn into cancer if left untreated and require sooner follow-up examinations. Histologically, HPs have a superficial serrated architecture and elongated crypts, whereas SSAs are characterized by broad-based crypts, often with complex structure and heavy serration. 

    Submit the data request form and you will be provided with the `annotations.csv` file and the `images.zip` file. A sample download script for MHIST can be found under `download_mhist.sh`. ***Note: The URLs of the files needs to be updated according to your data access grant***. This is a patch-level classification task.
   

   * Downstream task: Segmentation task of Colonic Nuclear Instance Segmentation. For this task we leverage the [Lizard dataset](https://arxiv.org/abs/2108.11195) developed by Warwick University available on Kaggle here. This is a patch level segmentation task. 
    ```wget -O archive.zip https://www.kaggle.com/api/v1/datasets/download/aadimator/conic-challenge-dataset```

   * Downstream task on **TCGA-COAD**. Using the GDC data transfer tool available [here](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool) and generate a manifest file with a all the WSI images from the TCGA data repository. Download the data using the [steps described in the TCGA documentation](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Preparing_for_Data_Download_and_Upload/)
  
        ```gdc-client download -m link/to/manifest-file.txt -d /mnt/efs/TCGA-COAD```

        This dataset is used for Slide-Level Tasks. The MSI status for the TCGA-COAD cohor is based on the findings of Liu et al available [here](https://github.com/KatherLab/cancer-metadata/blob/main/tcga/liu.xlsx)
        



5. Here, we will be leveraging the Foundational Model **H-Optimus-0** that was trained by [BioOptimus](https://www.bioptimus.com/news/bioptimus-launches-h-optimus-0-the-worlds-largest-open-source-ai-foundation-model-for-pathology). The model is weights are opensourced and can be downloaded to our EFS. 

```bash
cd SageMaker/mnt/efs
mkdir models && cd models
wget --no-check-certificate https://public-bioptimus-eu-west-3.s3.eu-west-3.amazonaws.com/h-optimus-v0/checkpoint.pth
```

 # MHIST Model Training steps

 Download the code repo:

 MHIST is a colorectal polyp classification public dataset that contains 3,152 images (224×224 pixels) presenting either hyperplastic polyp or sessile serrated adenoma at 5× magnification. This is a tile-level classification task. The dataset is defined and loaded in the `train/data_utils/MHIST.py` script. The training leverages a PyTorch Image defined in the `train/train_mhist.py` script. 

 The training can be started using the following code snippet: 

 ```python
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import FileSystemInput

# Initialize the SageMaker session
sagemaker_session = sagemaker.Session()

# Define the EFS file system input
efs_data_input = FileSystemInput(
    file_system_id='fs-XXXXXX', # MODIFY
    file_system_type='EFS',
    directory_path='/MHIST',
    file_system_access_mode='ro'
)

efs_model_input = FileSystemInput(
    file_system_id='fs-0b7a195df6775de4c',
    file_system_type='EFS',
    directory_path='/models',
    file_system_access_mode='ro'
)


# Configure the PyTorch estimator
estimator = PyTorch(
    source_dir='train',
    entry_point='train_mhist.py',
    role='arn:aws:iam::713881812217:role/EFS-SM-SageMakerRole-EyrRK8nNZo79',
    instance_count=1,
    instance_type='ml.p3.2xlarge',
    subnets=['subnet-XXXX'], # MODIFY
    security_group_ids=['sg-XXXXX'], # MODIFY
    framework_version='2.3',
    py_version='py311',
    hyperparameters={
        'epochs': 10,
        'batch-size': 32,
        'learning-rate': 0.001
    }
)

# Start the training job
estimator.fit({'training': efs_data_input, 'models': efs_model_input})
```

Once, the model is trained, you can look at the model metrics on the test dataset in the Training Logs. You can deploy the model to a SageMaker Endpoint with the following code: 

```python
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import FileSystemInput

# Initialize the SageMaker session
sagemaker_session = sagemaker.Session()

# Define the EFS file system input
efs_data_input = FileSystemInput(
    file_system_id='fs-0b7a195df6775de4c', # MODIFY
    file_system_type='EFS',
    directory_path='/Lizard',
    file_system_access_mode='ro'
)

efs_model_input = FileSystemInput(
    file_system_id='fs-0b7a195df6775de4c',
    file_system_type='EFS',
    directory_path='/models',
    file_system_access_mode='ro'
)


# Configure the PyTorch estimator
estimator = PyTorch(
    source_dir='train',
    entry_point='train_lizard.py',
    role='arn:aws:iam::713881812217:role/EFS-SM-SageMakerRole-EyrRK8nNZo79',
    instance_count=1,
    instance_type='ml.p3.2xlarge',
    subnets=['subnet-008fa8aee9db06e83'], # MODIFY
    security_group_ids=['sg-09d4640079b19f275'], # MODIFY
    framework_version='2.3',
    py_version='py311',
    hyperparameters={
        'epochs': 10,
        'batch-size': 32,
        'learning-rate': 0.0001
    }
)

# Start the training job
estimator.fit({'training': efs_data_input, 'models': efs_model_input})
```

# Segmentation Training (Lizard)

The Lizard dataset is used for colonic nuclear instance segmentation. The training script is located at `train/train_lizard.py`. Here's how to start the training job:

```python
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import FileSystemInput

# Initialize the SageMaker session
sagemaker_session = sagemaker.Session()

# Define the EFS file system inputs
efs_data_input = FileSystemInput(
    file_system_id='fs-0b7a195df6775de4c', # MODIFY
    file_system_type='EFS',
    directory_path='/Lizard',
    file_system_access_mode='ro'
)

efs_model_input = FileSystemInput(
    file_system_id='fs-0b7a195df6775de4c',
    file_system_type='EFS',
    directory_path='/models',
    file_system_access_mode='ro'
)

# Configure the PyTorch estimator
estimator = PyTorch(
    source_dir='train',
    entry_point='train_lizard.py',
    role='arn:aws:iam::713881812217:role/EFS-SM-SageMakerRole-EyrRK8nNZo79',
    instance_count=1,
    instance_type='ml.g5.16xlarge',
    subnets=['subnet-008fa8aee9db06e83'], # MODIFY
    security_group_ids=['sg-09d4640079b19f275'], # MODIFY
    framework_version='2.3',
    py_version='py311',
    hyperparameters={
        'epochs': 200,
        'batch-size': 128,
        'learning-rate': 1e-5
    },
    metric_definitions=[
        {'Name': 'TrainingLoss', 'Regex': 'Training Loss: ([0-9\\.]+)'},
        {'Name': 'ValidationLoss', 'Regex': 'Validation Loss: ([0-9\\.]+)'},
        {'Name': 'ValidationIOU', 'Regex': 'Validation Mean_IOU: ([0-9\\.]+)'},
        {'Name': 'ValidationDice', 'Regex': 'Validation Mean_DICE: ([0-9\\.]+)'}
    ],
    base_job_name='Lizard-Segmentation'
)

# Start the training job
estimator.fit({'training': efs_data_input, 'models': efs_model_input}, wait=False)
```

# Feature Extraction

This task uses a custom Docker image for tiling and feature extraction from whole-slide images. Before running the job, make sure to build and push the Docker image:
```bash
cd preprocessing
bash ./build_and_push.sh cucim-tiler
```

Then, you can start the feature extraction job:

```python
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import FileSystemInput

# Initialize the SageMaker session
sagemaker_session = sagemaker.Session()

# Define the EFS file system inputs
efs_data_input = FileSystemInput(
    file_system_id='fs-0b7a195df6775de4c', # MODIFY
    file_system_type='EFS',
    directory_path='/TCGA-COAD',
    file_system_access_mode='ro'
)

efs_data_output = FileSystemInput(
    file_system_id='fs-0b7a195df6775de4c', # MODIFY
    file_system_type='EFS',
    directory_path='/TCGA-COAD-features3',
    file_system_access_mode='rw'
)

estimator = Estimator(
    role='arn:aws:iam::713881812217:role/EFS-SM-SageMakerRole-EyrRK8nNZo79',
    instance_count=1,
    image_uri="713881812217.dkr.ecr.us-west-2.amazonaws.com/cucim-tiler:latest",
    instance_type='ml.g5.2xlarge',
    subnets=['subnet-008fa8aee9db06e83'], # MODIFY
    security_group_ids=['sg-09d4640079b19f275'], # MODIFY
    base_job_name='Tile-Feature-Extraction',
    metric_definitions=[
        {'Name': 'Slide #', 'Regex': 'Processing slide #([0-9\\.]+)'},
    ],
)

# Start the training job
estimator.fit({'dataset': efs_data_input, 'output': efs_data_output}, wait=False)
```

# WSI Classification (Slide level)

This task uses the features extracted from the TCGA-COAD dataset to predict MSI status at the whole-slide level. The training script is located at train/train_msi_tcga.py. Here's how to start the training job:

```python
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import FileSystemInput

# Initialize the SageMaker session
sagemaker_session = sagemaker.Session()

# Define the EFS file system input
efs_data_input = FileSystemInput(
    file_system_id='fs-0b7a195df6775de4c', # MODIFY
    file_system_type='EFS',
    directory_path='/TCGA-COAD-features2',
    file_system_access_mode='ro'
)

efs_model_input = FileSystemInput(
    file_system_id='fs-0b7a195df6775de4c',
    file_system_type='EFS',
    directory_path='/models',
    file_system_access_mode='ro'
)

# Configure the PyTorch estimator
estimator = PyTorch(
    source_dir='train',
    entry_point='train_msi_tcga.py',
    role='arn:aws:iam::713881812217:role/EFS-SM-SageMakerRole-EyrRK8nNZo79',
    instance_count=1,
    instance_type='ml.g4dn.4xlarge',
    subnets=['subnet-008fa8aee9db06e83'], # MODIFY
    security_group_ids=['sg-09d4640079b19f275'], # MODIFY
    framework_version='2.2',
    py_version='py310',
    hyperparameters={
        'epochs': 100,
        'batch-size': 128,
        'learning-rate': 1e-5,
        'max-tiles': 6000
    },
    metric_definitions=[
        {'Name': 'TrainingLoss', 'Regex': 'Training Loss: ([0-9\\.]+)'},
        {'Name': 'ValidationLoss', 'Regex': 'Validation Loss: ([0-9\\.]+)'},
        {'Name': 'ValidationAccuracy', 'Regex': 'Validation Accuracy: ([0-9\\.]+)'}
    ],
    base_job_name='WSI-Classification',
)

# Start the training job
estimator.fit({'training': efs_data_input, 'models': efs_model_input}, wait=False)
```

