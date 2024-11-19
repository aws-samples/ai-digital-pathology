import h5py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, dataloader


class MSIDataset(Dataset):
    def __init__(self, excel_file, embedding_dir, max_tiles=1000, split='train'):
        self.max_tiles=int(max_tiles)
        _data = pd.read_excel(excel_file)
        self.data = _data[_data['TCGA Project Code']=='COAD'][['TCGA Participant Barcode', 'MSI Status']].dropna(inplace=False)
        _label_map = {'MSS': 0, 'MSI-L': 1, 'MSI-H': 2}
        self.data['MSI Status'] = self.data['MSI Status'].map(_label_map)
        
        # Only keep files for which we have a WSI
        self.valid_patient_codes = [p.stem[:12] for p in Path(embedding_dir).glob('*h5')]
        self.data = self.data[self.data['TCGA Participant Barcode'].isin(self.valid_patient_codes)]
        X_train, X_test, y_train, y_test  = train_test_split(self.data['TCGA Participant Barcode'], self.data['MSI Status'], test_size=0.2, random_state=42)
        
        if split=='train':
            self.data = pd.DataFrame({'TCGA Participant Barcode': X_train.reset_index(drop=True), 'MSI Status': y_train.reset_index(drop=True)})
        else:
            self.data = pd.DataFrame({'TCGA Participant Barcode': X_test.reset_index(drop=True), 'MSI Status': y_test.reset_index(drop=True)})
        
        self.embedding_dir=embedding_dir
        print(f"kept {len(self.data)} slides")

    def __getitem__(self, index):
        embedding_path = next(Path(self.embedding_dir).glob(f"{self.data['TCGA Participant Barcode'][index]}*.h5"))
        if Path(embedding_path).exists():
            with h5py.File(embedding_path, 'r') as f:
                embeddings_array = f['feats'][:]
            if len(embeddings_array) >= self.max_tiles:
                embeddings_array = torch.from_numpy(embeddings_array[:self.max_tiles]).float()
        label = float(self.data['MSI Status'][index])

        return embeddings_array, label
        
    def __len__(self):
        return len(self.data)
    

def padding_collate(batch):
    """A collate function to pad features matrix of different lengths."""
    samples_to_pad, other_samples = [], []
    for sample in batch:
        samples_to_pad.append(sample[0])
        other_samples.append(sample[1:])

    features_dim = samples_to_pad[0].size()[-1]
    max_len = max([s.size(0) for s in samples_to_pad])
    padded_dims = (len(samples_to_pad), max_len, features_dim)

    padded_samples = samples_to_pad[0].data.new(*padded_dims).fill_(0.0)

    for i, tensor in enumerate(samples_to_pad):
        length = tensor.size(0)
        padded_samples[i, :length, ...] = tensor[:max_len, ...]

    # Batching other members of the tuple using default_collate
    other_samples = dataloader.default_collate(other_samples)

    return (padded_samples, *other_samples)
class MSI_TCGA_COAD():
    def __init__(self, dataset_path='/home/ec2-user/SageMaker/mnt/efs/TCGA-COAD-features', batch_size=16, max_tiles=1000):
        
        csv_file = f'{dataset_path}/liu.xlsx'
        
        # Split the dataset into train and validation
        dataset_train = MSIDataset(csv_file, dataset_path, max_tiles=1000, split='train')
        dataset_test = MSIDataset(csv_file, dataset_path, max_tiles=1000, split='test')
        test_dataset = MSIDataset(csv_file, embedding_dir=dataset_path, max_tiles=max_tiles, split='test')
        
        self.dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, collate_fn=padding_collate)
        self.dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=True, collate_fn=padding_collate)



