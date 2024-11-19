import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class MHISTDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.annotations = pd.read_csv(csv_file)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.img_dir = img_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Load Image and Resize
        img_name = f'{self.img_dir}/{self.annotations.iloc[idx, 0]}'
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)

        # Load Label
        label = 0.0 if self.annotations.iloc[idx, 1]=='HP' else 1.0
        
        return image, label


class MHIST():
    def __init__(self, dataset_path='/home/ec2-user/SageMaker/mnt/efs/MHIST', batch_size=16):
        csv_file = f'{dataset_path}/annotations.csv'
        img_dir = f'{dataset_path}/images'

        full_dataset = MHISTDataset(csv_file=csv_file, img_dir=img_dir)
        
        # Split the dataset into train and validation
        full_train_dataset = MHISTDataset(csv_file=csv_file, img_dir=img_dir)
        full_train_dataset.annotations = full_train_dataset.annotations[full_train_dataset.annotations['Partition'] == 'train']
        
        test_dataset = MHISTDataset(csv_file=csv_file, img_dir=img_dir)
        test_dataset.annotations = test_dataset.annotations[test_dataset.annotations['Partition'] == 'test']

        self.train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        self.classes = ('HP', 'SSA')

if __name__=="__main__":
    import matplotlib.pyplot as plt
    data = MHIST()
    images, labels = next(iter(data.train_loader, batch_size=4))
    
    # Plot the first 4 images
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    for i, ax in enumerate(axes.flat[:4]):
        ax.imshow(images[i].permute(1, 2, 0))
        ax.set_title(str(labels[i]))
        ax.axis('off')

    plt.tight_layout()
    plt.show()
