import argparse
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from hoptimus_model_backbone import HOPTIMUSZero
from data_utils.MHIST import MHIST
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm 
import shutil
from sklearn import metrics
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model_name', type=str, default='HOPTIMUSZero')
    return parser.parse_args()

class LinearProbe(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.encoder = HOPTIMUSZero()
        # Freeze the encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(1536, output_size)

    def forward(self, x):
        with torch.inference_mode(False):
            features = self.encoder(x)['x_norm_cls_token']
        features = features.clone().detach().requires_grad_(True)
        return self.linear(features)

class Trainer(object):
    def __init__(self):
        args = parse_args()
        print(f"args: {args}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("using device: ", self.device)
        self.batch_size = args.batch_size
        self.lr = args.learning_rate
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.model_dir = args.model_dir
        self.output_dir = args.output_dir
        self.loss_func = nn.BCEWithLogitsLoss()

        self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, 'tensorboard_logs'))
        self.best_acc = 0

        self.dataset = MHIST(
            dataset_path=os.environ.get('SM_CHANNEL_TRAINING', '/home/ec2-user/SageMaker/mnt/efs/MHIST/'),
            batch_size=self.batch_size
        )

        print("data access: SUCCESS")

        self.train_loader = self.dataset.train_loader
        self.val_loader = self.dataset.val_loader

        self.model = LinearProbe(output_size=1).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()            
            avg_loss = 0
            
            pbar = tqdm(self.train_loader, total=len(self.train_loader), desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for i, (images, labels) in enumerate(pbar):
                self.optimizer.zero_grad()
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images).squeeze()
                loss = self.loss_func(logits, labels)
                loss.backward()
                self.optimizer.step()
                
                avg_loss += loss.item()

                pbar.set_postfix({'loss': f'{avg_loss / (i+1):.4f}'})
                
                if i % 100 == 0:
                    self.writer.add_scalar('training loss', avg_loss / (i+1), epoch * len(self.train_loader) + i)

            
            print(f"Epoch {epoch+1}/{self.epochs} - Training Loss: {avg_loss / len(self.train_loader):.4f}")
            
            val_loss, val_acc = self.validation(epoch)
            
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    def validation(self, epoch):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images).squeeze()
                preds = nn.functional.sigmoid(logits)

                loss = self.loss_func(logits, labels)
                total_loss += loss

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        predictions = np.array(all_preds)
        labels = np.array(all_labels)
        acc = metrics.accuracy_score(labels, predictions>0.5)
        auc = metrics.roc_auc_score(labels, predictions)

        avg_loss = total_loss / len(self.val_loader)
        # avg_acc = accuracy_score(all_labels, all_preds)

        print(f"Validation Loss: {avg_loss:.4f}")
        print(f"Validation AUC: {auc:.4f}")
        print(f"Validation Accuracy: {acc:.4f}")

        self.writer.add_scalar('validation/loss', avg_loss, epoch)
        self.writer.add_scalar('validation/accuracy', acc, epoch)
        self.writer.add_scalar('validation/auc', auc, epoch)

        if auc > self.best_acc:
            self.best_acc = acc
            self.save_checkpoint(is_best=True)
        else:
            self.save_checkpoint(is_best=False)

        return avg_loss, acc

    def save_checkpoint(self, is_best=False):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Save to file
        filename = os.path.join(self.model_dir, "classification_model.pth")
        print("saved model...")

        torch.save(self.model.state_dict(), filename)
        
        if is_best:
            best_filename = os.path.join(self.model_dir, 'classification_model_best.pth')
            shutil.copyfile(filename, best_filename)
            print("saved new model...")

if __name__ == "__main__":
    print("Started training container. Running training script.")
    trainer = Trainer()
    trainer.train()
