from pathlib import Path
import os
import argparse
import time 
from cucim import CuImage
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np
import boto3
from matplotlib import pyplot as plt
import random
import torch
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from hoptimus_model_backbone import HOPTIMUSZero
import h5py

TARGET_MPP = 0.5 
TILE_SIZE = 224
TILE_COUNT = 1000

def upload_to_s3(local_path: str, s3_bucket: str, s3_key: str) -> str:
    """
    Upload a file to S3 and return the S3 URI
    """
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(local_path, s3_bucket, s3_key)
        s3_uri = f"s3://{s3_bucket}/{s3_key}"
        print(f"Successfully uploaded file to {s3_uri}")
        return s3_uri
    except Exception as e:
        print(f"Error uploading to S3: {str(e)}")
        raise e

def get_slide_mpp(slide: CuImage) -> float | None:
    """Get the slide resolution in MPP."""
    if "aperio" in slide.metadata and "MPP" in slide.metadata["aperio"]:
        return float(slide.metadata["aperio"]["MPP"])
    return None

def canny_fcn(patch: np.ndarray) -> bool:
    """Check if a patch is a foreground patch using Canny edge detection."""
    patch_img = Image.fromarray(patch)
    tile_to_greyscale = patch_img.convert("L")
    # tile_to_greyscale is an PIL.Image.Image with image mode L
    # Note: If you have an L mode image, that means it is
    # a single channel image - normally interpreted as greyscale.
    # The L means that is just stores the Luminance.
    # It is very compact, but only stores a greyscale, not colour.
    tile2array = np.array(tile_to_greyscale)
    # Hardcoded thresholds.
    edge = cv2.Canny(tile2array, 40, 100)
    # Avoid dividing by zero.
    edge = edge / np.max(edge) if np.max(edge) != 0 else 0
    edge = (
        ((np.sum(np.sum(edge)) / (tile2array.shape[0] * tile2array.shape[1])) * 100)
        if (tile2array.shape[0] * tile2array.shape[1]) != 0
        else 0
    )
    is_foreground_image = edge >= 2.0
    return is_foreground_image

def get_tile_and_check_is_foreground(
    slide: CuImage, location: tuple[int, int], level: int, tile_size: int
) -> tuple[np.ndarray, tuple[int, int], bool]:
    tile = slide.read_region(
        location=location,
        level=level,
        size=(tile_size, tile_size),
    )
    tile = np.asarray(tile)
    is_foreground = canny_fcn(tile)
    return tile, location, is_foreground

def get_level_closest_to_mpp(slide: CuImage, target_mpp: float) -> int:
    """Get the slide level closest to the target MPP."""
    slide_mpp = get_slide_mpp(slide)  # Slide resolution in MPP.
    if slide_mpp is None:
        raise ValueError("Slide MPP is not available in metadata")
    # Get the resolutions of the pyramid.
    slide_level_mpps = [
        slide_mpp * float(d) for d in slide.resolutions["level_downsamples"]
    ]
    level_closest_to_mpp = int(
        np.argmin([np.abs(mpp - target_mpp) for mpp in slide_level_mpps])
    )
    return level_closest_to_mpp

class SlideTileDataset(Dataset):
    def __init__(self, patches, transform):
        self.tiles = patches
        self.transform = transform

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i: int) -> torch.Tensor:
        image = Image.fromarray(self.tiles[i])
        image = self.transform(image)
        return image

def extract_embeddings(slide_path, model):
    slide_name = Path(slide_path).stem
    print(f"Processing slide with name {slide_name}")

    # Read image using CuCim and CuPy
    slide_image = CuImage(slide_path)
    
    try:
        # For simplicity, we get the level corresponding to the closest MPP in the
        # image pyramid of the target MPP.
        # /!\ The actual MPP used to get the tiles can therefore be quite different
        # from the target MPP, for instance if the slide only has the MPP
        # [0.25, 1.0, 2.0].
        level = get_level_closest_to_mpp(slide_image, float(TARGET_MPP))
    except:
        print(f"Could not find MPP for {slide_path}, going to next slide...")
        return None
        
    slide_dims_at_level = slide_image.resolutions["level_dimensions"][level]
    num_tiles_at_level = (
        slide_dims_at_level[0] // TILE_SIZE,
        slide_dims_at_level[1] // TILE_SIZE,
    )
    
    print(
        f"Total number of tiles found: {num_tiles_at_level[0] * num_tiles_at_level[1]}."
    )
    locations = [
        (i * TILE_SIZE, j * TILE_SIZE)
        for i in range(num_tiles_at_level[0])
        for j in range(num_tiles_at_level[1])
    ]
    random.shuffle(locations)
    futures = []
    with ThreadPoolExecutor() as executor:
        for location in locations:
            futures.append(
                executor.submit(
                    get_tile_and_check_is_foreground,
                    slide=slide_image,
                    location=location,
                    level=level,
                    tile_size=TILE_SIZE,
                )
            )
        foreground_tiles = []
        for future in as_completed(futures):
            tile, location, is_foreground = future.result()
            if is_foreground:
                foreground_tiles.append(tile)

            if len(foreground_tiles) >= TILE_COUNT:
                executor.shutdown(wait=False, cancel_futures=True)
                break
        print(f"Number of foreground tiles kept: {len(foreground_tiles)}.")
        
        dataset = SlideTileDataset(foreground_tiles, transform=model.transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
        print("Starting feature extraction ...")
        start = time.time()
        features = []
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                for batch in tqdm(dataloader):
                    batch = batch.to("cuda")
                    features_b = model(batch)['x_norm_cls_token']
                    features.append(features_b.cpu().numpy())

        features = np.concatenate(features, axis=0)
        end = time.time()
        print(
            f"Feature extraction done in {end-start:.2f} seconds ! Features shape: {features.shape}"
        )  
        output_path = os.path.join('/tmp/', f"{slide_name}_embeddings.h5")

        with h5py.File(output_path, 'w') as f:
            f['feats']=features
        print(f"Saved embedddings for {slide_name}")
        return output_path
    
def model_fn(model_dir):
    """Load the model for inference"""
    try:
        import urllib.request
        print("Testing internet connectivity...")
        urllib.request.urlopen('http://google.com', timeout=10)
        print("Internet connection available")
    except Exception as e:
        print(f"Internet connectivity test failed: {str(e)}")
        
    print(f"Starting model instantiation")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HOPTIMUSZero()
    model = model.to(device)
    model.eval()
    print("Hoptimus model loaded")
    return model 


def input_fn(request_body, request_content_type):
    """Parse input data for prediction"""
    print(f"Received content type: {request_content_type}")
    if request_content_type == 'application/jsonlines':
        # Parse the JSON line to get the S3 path
        data = json.loads(request_body.decode())
        s3_path = data["source"]
        
        # Download the file locally
        local_path = f"/tmp/{os.path.basename(s3_path)}"
        s3 = boto3.client('s3')
        
        # Parse the S3 URI
        bucket = s3_path.split('/')[2]
        key = '/'.join(s3_path.split('/')[3:])
        
        print(f"Downloading {s3_path} to {local_path}")
        # Download the file
        s3.download_file(bucket, key, local_path)
        return local_path
    elif request_content_type == 'application/x-directory':
        # request_body will be the local path to the file
        return request_body
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(file_path, model):
    """Run prediction on the input data"""
    try:
        print(f"Processing file: {file_path}")
        output_path = extract_embeddings(file_path, model)
        s3_uri = upload_to_s3(output_path, "pathologybenchmark-s3bucket-u7pe00xtbplu", "embeddings")
        
        # Delete the input file after processing
        if os.path.exists(file_path):
            os.remove(file_path)
            os.remove(output_path)
            print(f"Deleted temporary file: {file_path}")
        
        # Return just the filename
        return os.path.basename(output_path)
    
    except Exception as e:
        print(f"Error in predict_fn: {str(e)}")
        # Ensure file is deleted even if an error occurs
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted temporary file due to error: {file_path}")
        raise e

    
def output_fn(prediction, accept):
    """Format the prediction output"""
    print(f"Output filename: {prediction}")
    return prediction


if __name__ == "__main__":
    # Initialize the model
    start_time = time.time()
    model = model_fn("test")
    predict_fn("slide.svs", model)
    print(f"total time: {time.time()-start_time}")

