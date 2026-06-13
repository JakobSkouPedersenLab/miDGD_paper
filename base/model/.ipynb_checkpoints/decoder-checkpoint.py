import torch
import torch.nn as nn
from src.utils.helpers import get_activation

class Decoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: int, output_modules: list, activation="relu"):
        super(Decoder, self).__init__()
        # set up the shared decoder
        self.main = nn.ModuleList()
        for i in range(len(hidden_dims)):
            self.main.append(nn.Linear(input_dim, hidden_dims[i]))
            self.main.append(get_activation(activation))
            input_dim = hidden_dims[i]
        
        # set up the modality-specific output module(s)
        self.out_modules = nn.ModuleList()
        for i in range(len(output_modules)):
            self.out_modules.append(output_modules[i])
        self.n_out_groups = len(output_modules)
        self.n_out_features = sum([output_modules[i].n_features for i in range(self.n_out_groups)])
        
    def forward(self, z):
        for i in range(len(self.main)):
            z = self.main[i](z)
        out = [outmod(z) for outmod in self.out_modules]
        return out
    
    def log_prob(self, nn_output, target, scale=1, mod_id=None, feature_ids=None, reduction="sum"):
        '''
        calculating the log probability
        
        This function is providied with different options of how the output should be provided.
        All log-probs can be summed or averaged into one value (reduction == 'sum' or 'mean'), 
        or not reduced at all (thus giving a log-prob per feature per sample).

        Args:
        nn_output: list of tensors
            the output of the decoder
        target: list of tensors
            the target values
        scale: list of tensors
            the scale of the target values (if applicable, otherwise just 1)
        mod_id: int
            the id of the modality to calculate the log-prob for (if a subset is desired)
        feature_ids: list of int
            the ids of the features to calculate the log-prob for (if a subset is desired, only works if mod_id is not None)
        reduction: str
            the reduction method to use ('sum', 'mean', 'none')
        '''
        if reduction == 'sum':
            log_prob = 0.
            if mod_id is not None:
                log_prob += self.out_modules[mod_id].log_prob(nn_output,target,scale,feature_id=feature_ids).sum()
            else:
                for i in range(self.n_out_groups):
                    log_prob += self.out_modules[i].log_prob(nn_output[i],target[i],scale[i]).sum()
        elif reduction == 'mean':
            log_prob = 0.
            if mod_id is not None:
                log_prob += self.out_modules[mod_id].log_prob(nn_output,target,scale,feature_id=feature_ids).mean()
            else:
                for i in range(self.n_out_groups):
                    log_prob += self.out_modules[i].log_prob(nn_output[i],target[i],scale[i]).mean()
        else:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if mod_id is not None:
                log_prob = self.out_modules[mod_id].log_prob(nn_output,target,scale)
            else:
                n_features = sum([self.out_modules[i].n_features for i in range(self.n_out_groups)])
                log_prob = torch.zeros((nn_output[0].shape[0],n_features)).to(dev)
                start_features = 0
                for i in range(self.n_out_groups):
                    log_prob[:,start_features:(start_features+self.out_modules[i].n_features)] += self.out_modules[i].log_prob(nn_output[i],target[i],scale[i])
                    start_features += self.out_modules[i].n_features
        return log_prob
    
    def loss(self, nn_output, target, scale=None, mod_id=None, feature_ids=None, reduction="sum"):
        return -self.log_prob(nn_output, target, scale, mod_id, feature_ids, reduction)
