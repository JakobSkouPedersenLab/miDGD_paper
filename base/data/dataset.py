import torch
from torch.utils.data import Dataset
import numpy as np


class GeneExpressionDataset(Dataset):
    '''
    Creates a Dataset class for gene expression dataset
    gene_dim is the number of genes (features)
    The rows of the dataframe contain samples, and the
    columns contain gene expression values
    and the class label (tissue) at label_position.
    '''

    def __init__(self, data, label_position=-1, color_position=-2, sample_position=-3, tissue_position=-4, scaling_type='mean'):
        '''
        Args:
            gtex: pandas dataframe containing input and output data
            label_position: column id of the class labels
        '''
        self.scaling_type = scaling_type
        self.label_position = label_position
        self.color_position = color_position
        self.sample_position = sample_position
        self.tissue_position = tissue_position

        # convert labels to numbers
        self.label = data.iloc[:, label_position].values
        self.color = data.iloc[:, color_position].values
        self.sample_type = data.iloc[:, sample_position].values
        self.tissue_type = data.iloc[:, tissue_position].values
        self.data = torch.tensor(data.drop(
            data.columns[[sample_position, tissue_position, color_position, label_position]], axis=1).values).float()

    def __len__(self):
        return (self.data.shape[0])

    def __getitem__(self, idx=None):
        if idx is None:
            idx = np.arange(self.__len__())
        expression = self.data[idx, :]

        if self.scaling_type == 'mean':
            lib = torch.mean(expression, dim=-1)
        elif self.scaling_type == 'max':
            lib = torch.max(expression, dim=-1).values
        elif self.scaling_type == 'sum':
            lib = torch.sum(expression, dim=-1)

        return expression, lib, idx

    def __getlabel__(self, idx=None):
        if idx is None:
            idx = np.arange(self.__len__())
        return self.label[idx]