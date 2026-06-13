import torch
from torch.utils.data import Dataset, Sampler
import numpy as np


class GeneExpressionDatasetPrediction(Dataset):
    '''
    Creates a Dataset class for gene expression dataset including both mRNA and miRNA data.
    The rows of the dataframe contain samples, and the columns contain gene expression values.
    '''

    def __init__(self, mrna_data, mirna_data, anno=None, scaling_type='sum'):
        '''
        Args:
            mrna_data: pandas dataframe containing mRNA input data
            mirna_data: pandas dataframe containing miRNA input data
            anno: pandas dataframe containing annotations
            label_position: column id of the class labels (assumed to be the same for both dataframes)
            scaling_type: type of scaling to apply ('mean' or 'max')
        '''
        self.scaling_type = scaling_type

        # Assuming labels are the same for both mrna and mirna data and are located in the same column
        if anno:
            if 'annotation' in anno:
                self.annotation = anno["annotation"].values
            if 'color' in anno:
                self.color = anno["color"].values
            if 'primary_site' in anno:
                self.primary_site = anno["primary_site"].values
            if 'tissue_type' in anno:
                self.tissue_type = anno["tissue_type"].values
            if 'batch' in anno:
                self.batch = anno["batch"].values

        
        # Make a new variable that convert labels to integers
        # Extract string labels and create integer mapping
        self.str_labels = self.annotation
        self.unique_labels = sorted(set(self.str_labels))
        self.label_to_index = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

        # Convert to integer labels
        self.label = np.array([self.label_to_index[l] for l in self.str_labels])

        # Convert data to tensors and remove label columns
        self.mrna_data = torch.tensor(mrna_data.values).float()
        self.mirna_data = torch.tensor(mirna_data.values).float()

        #self.mrna_data = self.mrna_data.to(device)
        #self.mirna_data = self.mirna_data.to(device)

    def __len__(self):
        # Assuming both mrna_data and mirna_data have the same number of samples
        return self.mrna_data.shape[0]

    def __getitem__(self, idx):
        if idx is None:
            idx = np.arange(self.__len__())
        mrna_expression = self.mrna_data[idx, :]
        mirna_expression = self.mirna_data[idx, :]
        label = self.label[idx]

        # Apply scaling if specified
        if self.scaling_type == 'mean':
            mrna_lib = torch.nanmean(mrna_expression, dim=-1)
            mirna_lib = torch.nanmean(mirna_expression, dim=-1)
        elif self.scaling_type == 'max':
            mrna_lib = torch.nanmax(mrna_expression, dim=-1).values
            mirna_lib = torch.nanmax(mirna_expression, dim=-1).values
        elif self.scaling_type == 'sum':
            mrna_lib = torch.nansum(mrna_expression, dim=-1)
            mirna_lib = torch.nansum(mirna_expression, dim=-1)

        return mrna_expression, mirna_expression, mrna_lib, mirna_lib, idx, label

    def __getlabel__(self, idx=None):
        if idx is None:
            idx = np.arange(self.__len__())
        return self.label[idx]
    
    def get_original_labels(self, indices):
        return [self.index_to_label[int(i)] for i in indices]

    

class StratifiedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, shuffle=True):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.unique_labels, self.label_counts = np.unique(self.labels, return_counts=True)
        self.label_to_idx = {label: np.where(self.labels == label)[0] for label in self.unique_labels}
        
    def __iter__(self):
        if self.shuffle:
            for label in self.unique_labels:
                np.random.shuffle(self.label_to_idx[label])
        
        num_batches = len(self.labels) // self.batch_size
        proportions = self.label_counts / len(self.labels)
        
        for _ in range(num_batches):
            batch = []
            for label, prop in zip(self.unique_labels, proportions):
                n_samples = int(prop * self.batch_size)
                if len(self.label_to_idx[label]) < n_samples:
                    # If we don't have enough samples, reshuffle
                    self.label_to_idx[label] = np.where(self.labels == label)[0]
                    if self.shuffle:
                        np.random.shuffle(self.label_to_idx[label])
                batch.extend(self.label_to_idx[label][:n_samples])
                self.label_to_idx[label] = self.label_to_idx[label][n_samples:]
            
            if len(batch) < self.batch_size:
                # Fill the remaining slots randomly
                remaining = self.batch_size - len(batch)
                batch.extend(np.random.choice(len(self.labels), remaining, replace=False))
            
            yield batch
    
    def __len__(self):
        return len(self.labels) // self.batch_size
