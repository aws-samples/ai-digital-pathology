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
from matplotlib import pyplot as plt
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from hoptimus_model_backbone import HOPTIMUSZero
import h5py

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_mpp', type=float, default=0.5, help='Target microns per pixel')
    parser.add_argument('--num_process', type=int, default=4, help='Number of processes to use')
    parser.add_argument('--tile_size', type=int, default=224, help='Size of the tile')
    parser.add_argument('--tile_count', type=int, default=1000, help='Number of tiles')
    
    args = parser.parse_args()
    
    # Print all arguments
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    return args

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


def main(args):
    slide_paths = [path for path in Path(SLIDE_PATHS).glob('**/*.svs')]
    print(f'Found {len(slide_paths)} slide(s)')

    # Create Output Dictionary
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)

    # Load Model & preprocessing steps
    model = HOPTIMUSZero()
    model.to("cuda")
    model.eval()
    
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617),
                std=(0.211883, 0.230117, 0.177517),
            ),
        ]
    )
    print(f"FM Model loaded to GPU")

    for idx, slide_path in enumerate(slide_paths):
        slide_name = Path(slide_path).stem
        print(f"Processing slide #{idx+1}/{len(slide_paths)}, with name {slide_name}")

        if Path(f'{OUTDIR}/{slide_name}.h5').exists():
            print(f"Embeddings for {slide_name} already exist. Skipping slide")
            continue

        # Read image using CuCim and CuPy
        slide_image = CuImage(slide_path.as_posix())
        
        try:
            # For simplicity, we get the level corresponding to the closest MPP in the
            # image pyramid of the target MPP.
            # /!\ The actual MPP used to get the tiles can therefore be quite different
            # from the target MPP, for instance if the slide only has the MPP
            # [0.25, 1.0, 2.0].
            level = get_level_closest_to_mpp(slide_image, float(args.target_mpp))
        except:
            print(f"Could not find MPP for {slide_path}, going to next slide...")
            continue
            
        slide_dims_at_level = slide_image.resolutions["level_dimensions"][level]
        num_tiles_at_level = (
            slide_dims_at_level[0] // args.tile_size,
            slide_dims_at_level[1] // args.tile_size,
        )
        
        print(
            f"Total number of tiles found: {num_tiles_at_level[0] * num_tiles_at_level[1]}."
        )
        locations = [
            (i * args.tile_size, j * args.tile_size)
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
                        tile_size=args.tile_size,
                    )
                )
            foreground_tiles = []
            for future in as_completed(futures):
                tile, location, is_foreground = future.result()
                if is_foreground:
                    foreground_tiles.append(tile)

                if len(foreground_tiles) >= args.tile_count:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
        print(f"Number of foreground tiles kept: {len(foreground_tiles)}.")
        
        dataset = SlideTileDataset(foreground_tiles, transform=transform)
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
        with h5py.File(f'{OUTDIR}/{slide_name}.h5', 'w') as f:
            f['feats']=features
        print(f"Saved embedddings for {slide_name}")

def debug_cuda_versions():
    print("CUDA Version:")
    try:
        import pycuda.driver as cuda
        cuda_version = cuda.get_version()
        print(f"CUDA version: {cuda_version[0]}.{cuda_version[1]}")
    except ImportError:
        print("pycuda not found")
    
    # print("\nCuPy Version:")
    # try:
    #     import cupy as cp
    #     cupy_version = cp.__version__
    #     print(f"CuPy version: {cupy_version}")
    # except ImportError:
    #     print("CuPy not found")
    
    print("\nCuCIM Version:")
    try:
        import cucim
        cucim_version = cucim.__version__
        print(f"CuCIM version: {cucim_version}")
    except ImportError:
        print("CuCIM not found")
    
    print("\nTorch Version:")
    try:
        import torch
        torch_version = torch.__version__
        torch_cuda_version = torch.version.cuda
        print(f"PyTorch version: {torch_version}")
        print(f"PyTorch CUDA version: {torch_cuda_version}")
    except ImportError:
        print("PyTorch not found")



if __name__ == "__main__":
    debug_cuda_versions()
    SLIDE_PATHS = os.environ.get('SM_CHANNEL_DATASET', '/home/ec2-user/SageMaker/mnt/efs/TCGA-COAD')
    OUTDIR = os.environ.get('SM_CHANNEL_OUTPUT', '/home/ec2-user/SageMaker/mnt/efs/TCGA-COAD-acc/')
    args = parse_args()
    main(args)
