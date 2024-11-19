## Code Inspired by https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DINOv2/Train_a_linear_classifier_on_top_of_DINOv2_for_semantic_segmentation.ipynb

import argparse
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from hoptimus_model_backbone import HOPTIMUSZero
from data_utils.Lizard import Lizard
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import evaluate
import shutil
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.classification import Dice


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model_name', type=str, default='HOPTIMUSZero')
    return parser.parse_args()

class SegmentationModel(nn.Module):
    def __init__(self, num_classes=7):
        # Num_classes (including Background class !)
        super().__init__()
            
        self.encoder = HOPTIMUSZero()
        # Freeze the encoder weights
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
            
        # Define a Segmentation Convolutional Head
        self.segmentation_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(1536, 64, (3, 3), padding=(1, 1)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, num_classes, (3, 3), padding=(1, 1)),
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        with torch.inference_mode(False):
            features = self.encoder(x)['x_norm_patch_tokens']
            
        x = features.clone().detach().requires_grad_(True)
        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size, 1536, 16, 16)
        x = self.segmentation_conv(x)
        logits = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return logits

class Trainer(object):
    def __init__(self):
        args = parse_args()
        print(f"args: {args}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("using device: ", self.device)
        self.batch_size = args.batch_size
        self.lr = args.learning_rate
        self.epochs = args.epochs
        self.model_dir = args.model_dir
        self.output_dir = args.output_dir


        self.writer = SummaryWriter(log_dir='runs')
        self.best_iou = 0

        self.dataset = Lizard(dataset_path=os.environ.get('SM_CHANNEL_TRAINING', '/home/ec2-user/SageMaker/mnt/efs/Lizard/'), batch_size=self.batch_size)
        print("data access: SUCCESS")

        self.train_loader = self.dataset.train_loader
        self.val_loader = self.dataset.test_loader
        
        self.metric = evaluate.load('mean_iou', num_labels=len(self.dataset.id2class.keys()), ignore_index=0)
        self.dice = Dice(average='micro', ignore_index=0).to(self.device)
        
        self.model = SegmentationModel(num_classes=len(self.dataset.id2class.keys())).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        
    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            avg_loss = 0
            avg_dice = 0
            
            pbar = tqdm(self.train_loader, total=len(self.train_loader), desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for i, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item()
                
                # Zero the parameter gradient
                self.optimizer.zero_grad()

                # Compute mean_iou metric
                predicted = outputs.argmax(dim=1)
                self.metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())
                avg_dice += self.dice(outputs, labels).item()
                
        
                metrics = self.metric.compute(num_labels=len(self.dataset.id2class.keys()), ignore_index=0, reduce_labels=False)
                mean_iou = metrics['mean_iou']
                mean_acc = metrics['mean_accuracy']
                pbar.set_postfix({'loss': f'{avg_loss / (i+1):.4f}', 'mean_iou': f'{mean_iou:.4f}', 'mean_acc': f'{mean_acc:.4f}'})
            # Training Loss
            print(f"Epoch {epoch+1}/{self.epochs} - Training Loss: {avg_loss / len(self.train_loader):.4f}, Traning Dice: {avg_dice/len(self.train_loader):.4f}")
            
            # Validation 
            val_loss, val_mean_iou, val_mean_acc, val_dice = self.validation()
            print(f"Epoch {epoch+1}/{self.epochs} - Validation Loss: {val_loss:.4f}")
            print(f"Epoch {epoch+1}/{self.epochs} - Validation Mean_IOU: {val_mean_iou:.4f}")
            print(f"Epoch {epoch+1}/{self.epochs} - Validation Mean_Accuracy: {val_mean_acc:.4f}")
            print(f"Epoch {epoch+1}/{self.epochs} - Validation Mean_DICE: {val_mean_acc:.4f}")
            
                
    def validation(self):
        self.model.eval()
        total_val_loss = 0
        total_val_dice = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_val_loss += loss.item()

                # Compute mean_iou metric
                predicted = outputs.argmax(dim=1)
                self.metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())
                total_val_dice += self.dice(outputs, labels).item()

        val_loss = total_val_loss / len(self.val_loader)
        val_dice = total_val_dice / len(self.val_loader)
        
        val_metrics = self.metric.compute(num_labels=len(self.dataset.id2class.keys()), ignore_index=0)
        val_mean_iou = val_metrics['mean_iou']
        val_mean_acc = val_metrics['mean_accuracy']
        
        
        if val_mean_iou > self.best_iou:
            self.best_iou = val_mean_iou
            self.save_checkpoint(is_best=True)
        else:
            self.save_checkpoint(is_best=False)

        return val_loss, val_mean_iou, val_mean_acc, val_dice
    
    def save_checkpoint(self, is_best=False):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Save to file
        filename = os.path.join(self.model_dir, "segmentation_model.pth")
        print("saved model...")

        torch.save(self.model.state_dict(), filename)
        

        if is_best:
            best_filename = os.path.join(self.model_dir, 'segmentation_model_best.pth')
            shutil.copyfile(filename, best_filename)
            print("saved new model...")
        
            
def main():
    trainer = Trainer()
    trainer.train()


def visualize_map(dataset, image, segmentation_map):
    "Debug utility function to plot the segmentation map"
    # map every class to a random color
    id2color = {k: list(np.random.choice(range(256), size=3)) for k,v in dataset.id2class.items()}

    color_seg = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3
    for label, color in id2color.items():
        color_seg[segmentation_map == label, :] = color

    # Show image + mask
    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.show()



if __name__ == "__main__":
    main()
