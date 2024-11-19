import os
import argparse 
import torch
import shutil
from tqdm import tqdm 
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score

from data_utils.MSI_TCGA_COAD import MSI_TCGA_COAD
from deepmil import DeepMIL


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--max-tiles', type=float, default=10000)
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    return parser.parse_args()

def save_checkpoint(model, is_best=False):
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    
    # Save to file
    filename = os.path.join(args.model_dir, "wsi_classification_model.pth")
    print("saved model...")

    torch.save(model.state_dict(), filename)
    
    if is_best:
        best_filename = os.path.join(args.model_dir, 'wsi_classification_model_best.pth')
        shutil.copyfile(filename, best_filename)
        print("saved new model...")


args = parse_args()

# Load Dataset
dataset = MSI_TCGA_COAD(dataset_path=args.data_dir, batch_size=args.batch_size, max_tiles=args.max_tiles)
dataloader_train = dataset.dataloader_train
dataloader_test = dataset.dataloader_test

# Define the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model
model = DeepMIL(
    in_features=1536,
    out_features=1,
).to(device)


# Define the loss function and optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# Training loop
num_epochs = args.epochs
best_acc = 0.0


for epoch in range(num_epochs):
    avg_loss = 0
    model.train()
    pbar = tqdm(dataloader_train, total=len(dataloader_train), desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch_idx, (embeddings_b, labels_b) in enumerate(pbar):
        embeddings_b, labels_b = embeddings_b.to(device), labels_b.to(device)
        mask_b = embeddings_b.sum(-1, keepdim=True) == 0.0
        mask_b = mask_b.to("cuda")
        
        optimizer.zero_grad()

        # Forward pass
        logits_b = model(embeddings_b, mask_b).squeeze()
        loss = criterion(logits_b, labels_b)
                

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()

        pbar.set_postfix({'loss': f'{avg_loss / (batch_idx+1):.4f}'})

        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader_train)}], Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss/len(dataloader_train):.4f}")

    # Validation
    model.eval()
    with torch.no_grad():
        validation_avg_loss = 0
        true_labels, predicted_labels = [], []

        for val_embeddings_b, val_labels_b in dataloader_test:
            mask_b = val_embeddings_b.sum(-1, keepdim=True) == 0.0
            val_embeddings_b, val_labels_b = val_embeddings_b.to("cuda"), val_labels_b.to("cuda")
            mask_b = mask_b.to("cuda")

            logits_b = model(val_embeddings_b, mask_b).squeeze()
            logits_b = logits_b.unsqueeze(0) if logits_b.dim() == 0 else logits_b
            preds_b = F.sigmoid(logits_b)
            
            #preds_b = torch.argmax(val_logits, dim=1)

            true_labels.extend(val_labels_b.cpu().numpy()) 
            predicted_labels.extend(preds_b.cpu().numpy())
            validation_avg_loss += criterion(preds_b, val_labels_b).item()
            
        accuracy = accuracy_score(np.array(true_labels), np.array(predicted_labels)>0.5)
        rocauc = roc_auc_score(np.array(true_labels), np.array(predicted_labels))

        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {validation_avg_loss/len(dataloader_test):.4f}")
        print(f"Validation Accuracy: {accuracy}")
        print(f"Validation ROC AUC: {rocauc}")

        if accuracy > best_acc:
            best_acc = accuracy
            save_checkpoint(model, is_best=True) 
        else:
            save_checkpoint(model, is_best=False)

    model.train()

