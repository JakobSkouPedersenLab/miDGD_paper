import torch
import torch.nn as nn
from base.utils.helpers import get_activation
from base.dgd.nn import NB_Module


class Decoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, 
                 output_module_mirna: NB_Module = None, output_module_mrna: NB_Module = None, 
                 output_module:NB_Module = None, activation="relu"):
        super(Decoder, self).__init__()
        # set up the shared decoder
        self.main = nn.ModuleList()
        for i in range(len(hidden_dims)):
            self.main.append(nn.Linear(input_dim, hidden_dims[i]))
            self.main.append(get_activation(activation))
            input_dim = hidden_dims[i]

        # set up the modality-specific output module(s)

        if output_module is None:
            # set up the modality-specific output module(s)
            self.out_module_mirna = output_module_mirna
            self.out_module_mrna = output_module_mrna
            
            self.midgd = True
            self.n_out_groups = 2
            self.n_out_features_mirna = output_module_mirna.n_features
            self.n_out_features_mrna = output_module_mrna.n_features
            self.n_out_features = self.n_out_features_mrna + self.n_out_features_mirna
        else:
            self.midgd = False
            self.out_module = output_module
            self.n_out_groups = 1
            self.n_out_features = output_module.n_features

    def forward(self, z):
        for i in range(len(self.main)):
            z = self.main[i](z)
        if self.midgd:
            out_mrna = self.out_module_mrna(z)
            out_mirna = self.out_module_mirna(z)
            out = [out_mirna, out_mrna]
        else:
            out = self.out_module(z)
        return out

    def log_prob(self, nn_output, target, scale=1, mod_id=None, feature_ids=None, reduction="sum"):
        '''
        Calculating the log probability

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
            log_prob_mirna = 0.
            log_prob_mrna = 0.
            log_prob = 0.

            if mod_id == "mirna":
                log_prob += self.out_module_mirna.log_prob(
                    nn_output, target, scale, feature_id=feature_ids).sum()
            elif mod_id == "mrna":
                log_prob += self.out_module_mrna.log_prob(
                    nn_output, target, scale, feature_id=feature_ids).sum()
            elif mod_id == "single":
                log_prob += self.out_module.log_prob(
                    nn_output, target, scale, feature_id=feature_ids).sum()
            else: # combined
                log_prob_mirna += self.out_module_mirna.log_prob(
                    nn_output[0], target[0], scale[0], feature_id=feature_ids).sum()
                log_prob_mrna += self.out_module_mrna.log_prob(
                    nn_output[1], target[1], scale[1], feature_id=feature_ids).sum()
                return log_prob_mirna, log_prob_mrna
            return log_prob
        elif reduction == 'mean':
            log_prob_mirna = 0.
            log_prob_mrna = 0.
            log_prob = 0.

            if mod_id == "mirna":
                log_prob += self.out_module_mirna.log_prob(
                    nn_output, target, scale, feature_id=feature_ids).mean()
            elif mod_id == "mrna":
                log_prob += self.out_module_mrna.log_prob(
                    nn_output, target, scale, feature_id=feature_ids).mean()
            elif mod_id == "single":
                log_prob += self.out_module.log_prob(
                    nn_output, target, scale, feature_id=feature_ids).mean()
            else:
                log_prob_mirna += self.out_module_mirna.log_prob(
                    nn_output[0], target[0], scale[0], feature_id=feature_ids).mean()
                log_prob_mrna += self.out_module_mrna.log_prob(
                    nn_output[1], target[1], scale[1], feature_id=feature_ids).mean()
                return log_prob_mirna, log_prob_mrna
            return log_prob
        elif reduction == 'sample':
            log_prob_mirna = []
            log_prob_mrna = []
            log_prob = []

            if mod_id == "mirna":
                log_prob += self.out_module_mirna.log_prob(
                    nn_output, target, scale, feature_id=feature_ids).mean()
            elif mod_id == "mrna":
                log_prob += self.out_module_mrna.log_prob(
                    nn_output, target, scale, feature_id=feature_ids).mean()
            elif mod_id == "single":
                log_prob += self.out_module.log_prob(
                    nn_output, target, scale, feature_id=feature_ids).mean()
            else:
                log_prob_mirna += self.out_module_mirna.log_prob(
                    nn_output[0], target[0], scale[0], feature_id=feature_ids).mean()
                log_prob_mrna += self.out_module_mrna.log_prob(
                    nn_output[1], target[1], scale[1], feature_id=feature_ids).mean()
                return log_prob_mirna, log_prob_mrna
            return log_prob
        else:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if mod_id is not None:
                log_prob = self.out_modules[mod_id].log_prob(
                    nn_output, target, scale)
            else:
                n_features = sum(
                    [self.out_modules[i].n_features for i in range(self.n_out_groups)])
                log_prob = torch.zeros(
                    (nn_output[0].shape[0], n_features)).to(dev)
                start_features = 0
                for i in range(self.n_out_groups):
                    log_prob[:, start_features:(start_features+self.out_modules[i].n_features)
                             ] += self.out_modules[i].log_prob(nn_output[i], target[i], scale[i])
                    start_features += self.out_modules[i].n_features
        return log_prob

    def loss(self, nn_output, target, scale=None, mod_id=None, feature_ids=None, reduction="sum", type="combined"):
        if type == "combined":
            log_prob_mirna, log_prob_mrna = self.log_prob(
                nn_output, target, scale, mod_id, feature_ids, reduction)
            return -log_prob_mirna, -log_prob_mrna
        else:
            return -self.log_prob(nn_output, target, scale, mod_id, feature_ids, reduction)