from base.dgd.latent import GaussianMixture
import torch.nn as nn

class DGD(nn.Module):
    def __init__(self, decoder, n_mix, rep_dim, gmm_spec={}):
        super(DGD, self).__init__()
        self.decoder = decoder
        self.rep_dim = rep_dim      # Dimension of representation

        self.gmm = GaussianMixture(n_mix, rep_dim, **gmm_spec, covariance_type="diagonal")
        self.train_rep = None
        self.val_rep = None
        self.test_rep = None

    def forward(self, z):
        return self.decoder(z)

    def loss(self, z, y, target, scale, gmm_loss=True, reduction="sum", type="combined"):
        if type == "combined":  # Both mRNA and miRNA data are passed
            self.dec_loss_mirna, self.dec_loss_mrna = self.decoder.loss(
                y, target, scale, reduction=reduction, type=type)
        elif type == "mrna" or type == "mirna":  
            self.dec_loss = self.decoder.loss(
                y, target, scale, mod_id="single", reduction=reduction, type=type)
        elif type == "midgd":
            self.dec_loss = self.decoder.loss(
                y[1], target[1], scale[1], mod_id="mrna", reduction=reduction, type=type)
        elif type == "switch":
            self.dec_loss = self.decoder.loss(
                y[0], target[0], scale[0], mod_id="mirna", reduction=reduction, type=type)
            
        if not gmm_loss:
            if type == "combined":
                return self.dec_loss_mirna, self.dec_loss_mrna, None
            else:
                return self.dec_loss, None
        else:
            self.gmm_loss = self.gmm(z)
            if reduction == "mean":
                self.gmm_loss = self.gmm_loss.mean()
            elif reduction == "sum":
                self.gmm_loss = self.gmm_loss.sum()

            if type == "combined":
                return self.dec_loss_mirna, self.dec_loss_mrna, self.gmm_loss
            else:
                return self.dec_loss, self.gmm_loss

    def forward_and_loss(self, z, target, scale, gmm_loss=True, reduction="sum", type="combined"):
        y = self.decoder(z)

        return self.loss(z, y, target, scale, gmm_loss, reduction, type)

    def get_representations(self, type="train"):
        if type == "train":
            return self.train_rep.z.detach().cpu().numpy()
        elif type == "val":
            return self.val_rep.z.detach().cpu().numpy()
        elif type == "test":
            return self.test_rep.z.detach().cpu().numpy()

    def get_gmm_means(self):
        return self.gmm.mean.detach().cpu().numpy()

    def get_latent_space_values(self, rep_type="train", n_samples=1000):
        # get representations
        if rep_type == "train":
            rep = self.train_rep.z.clone().detach().cpu().numpy()
        elif rep_type == "val":
            rep = self.val_rep.z.clone().detach().cpu().numpy()
        elif rep_type == "test":
            rep = self.test_rep.z.clone().detach().cpu().numpy()

        # get gmm means
        gmm_means = self.gmm.mean.clone().detach().cpu().numpy()

        # get some gmm samples
        gmm_samples = self.gmm.sample(n_samples).detach().cpu().numpy()

        return rep, gmm_means, gmm_samples

import torch
import torch.nn as nn
import numpy as np
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
        # Handling missing data
        # mods = ['single', 'mrna', 'mirna']

        # if mod_id in mods:
        #     mask = ~np.isnan(target)
        #     target = target[mask]
        #     nn_output = nn_output[mask]
        # else:
        #     mirna_mask = ~np.isnan(target[0])
        #     mrna_mask = ~np.isnan(target[1])

        #     target[0] = target[0][mirna_mask] 
        #     target[1] = target[1][mrna_mask]
        #     nn_output[0] = nn_output[0][mirna_mask]
        #     nn_output[1] = nn_output[1][mrna_mask]

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
        # Calculate loss
        if type == "combined":
            log_prob_mirna, log_prob_mrna = self.log_prob(
                nn_output, target, scale, mod_id, feature_ids, reduction)
            return -log_prob_mirna, -log_prob_mrna
        else:
            return -self.log_prob(nn_output, target, scale, mod_id, feature_ids, reduction)
        

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def logNBdensity(k, m, r):
    """
    Negative Binomial NB(k;m,r), where m is the mean and k is "number of failures"
    r can be real number (and so can k)
    k, and m are tensors of same shape
    r is tensor of shape (1, n_genes)
    Returns the log NB in same shape as k
    """
    # remember that gamma(n+1)=n!
    eps = 1e-16  # this is under-/over-flow protection
    x = torch.lgamma(k + r)
    x -= torch.lgamma(r)
    x -= torch.lgamma(k + 1)
    x += k * torch.log(m * (r + m + eps) ** (-1) + eps)
    x += r * torch.log(r * (r + m + eps) ** (-1))
    return x


def logZINBdensity(k, m, r, pi):
    """
    Zero-Inflated Negative Binomial ZINB(k;m,r,p), where m is the mean and k is "number of failures"
    r can be real number (and so can k)
    k and m are tensors of same shape
    r and pi is tensor of shape (1, n_genes)
    Returns the log ZINB in same shape as k
    """
    # remember that gamma(n+1)=n!
    eps = 1e-16  # this is under-/over-flow protection
    log_nb = logNBdensity(k, m, r)
    
    log_zinb = torch.where(
        k == 0,
        torch.log(pi + eps + (1 - pi) * torch.exp(log_nb)),
        torch.log(1 - pi + eps) + log_nb
    )
    
    return log_zinb


class OutputModule(nn.Module):
    """
    This is the basis output module class that stands between the decoder and the output data.

    Attributes
    ----------
    fc: torch.nn.modules.container.ModuleList
    n_in: int
        number of hidden units going into this layer
    n_out: int
        number of features that come out of this layer
    distribution: torch.nn.modules.module.Module
        specific class depends on modality argument

    Methods
    ----------
    forward(x)
        input goes through fc modulelist and distribution layer
    loss(model_output,target,scaling_factor,gene_id=None)
        returns loss of distribution for given output
    log_prob(model_output,target,scaling_factor,gene_id=None)
        returns log-prob of distribution for given output
    """

    def __init__(self, fc: torch.nn.modules.container.ModuleList, out_features: int):
        """Args:
        fc: feed-forward NN with at least 1 layer and no last activation function with output dimension
            equal to out_features and input dimension equal to the output dimension of the decoder used
        out_features: number of features from the data modelled by this module
        """
        super(OutputModule, self).__init__()

        self.fc = fc
        self.n_out = out_features

    def forward(self, x):
        for i in range(len(self.fc)):
            x = self.fc[i](x)
        return x

    """This included DEA but will be discussed later"""


class NB_Module(OutputModule):
    """
    This is the Negative Binomial version of the OutputModule distribution layer.

    Attributes
    ----------
    fc: torch.nn.modules.container.ModuleList
    log_r: torch.nn.parameter.Parameter
        log-dispersion parameter per feature

    Methods
    ----------
    forward(x)
        applies scaling-specific activation
    loss(model_output,target,scaling_factor,gene_id=None)
        returns loss of NB for given output
    log_prob(model_output,target,scaling_factor,gene_id=None)
        returns log-prob of NB for given output
    """

    def __init__(self, fc, out_features, r_init=2, scaling_type="sum"):
        """Args:
        fc: NN (see parent class)
        out_features: number of features from the data modelled by this module
        r_init: initial dispersion factor for all features
        scaling_type: describes type of transformation from model output to targets
            and determines what activation function is used on the output
        """
        super(NB_Module, self).__init__(fc, out_features)

        # substracting 1 now and adding it to the learned dispersion ensures a minimum value of 1
        self.log_r = torch.nn.Parameter(
            torch.full(fill_value=math.log(r_init - 1),
                       size=(1, out_features)),
            requires_grad=True,
        )
        self._scaling_type = scaling_type
        if self._scaling_type == "sum":  # could later re-implement more scalings, but sum is arguably the best so far
            self._activation = "softmax"
        elif self._scaling_type == "mean":
            self._activation = "softplus"
        elif self._scaling_type == "max":
            self._activation = "sigmoid"
        else:
            raise ValueError(
                "scaling_type must be one of 'sum', 'mean', or 'max', but is "
                + self._scaling_type
            )

    def forward(self, x):
        for i in range(len(self.fc)):
            x = self.fc[i](x)
        if self._activation == "softmax":
            return F.softmax(x, dim=-1)
        elif self._activation == "softplus":
            return F.softplus(x)
        elif self._activation == "sigmoid":
            return F.sigmoid(x)
        else:
            return x

    @staticmethod
    def rescale(scaling_factor, model_output):
        return scaling_factor * model_output

    def log_prob(self, model_output, target, scaling_factor, feature_id=None):
        # target is the true value
        # the model output represents the mean normalized count
        # the scaling factor is the used normalization
        if feature_id is not None:  # feature_id could be a single gene
            log_prob = logNBdensity(
                target,
                self.rescale(scaling_factor, model_output),
                (torch.exp(self.log_r) + 1)[0, feature_id],
            )
        else:
            log_prob = logNBdensity(
                target,
                self.rescale(scaling_factor, model_output),
                (torch.exp(self.log_r) + 1),
            )
        
        return log_prob

    def loss(self, model_output, target, scaling_factor, gene_id=None):
        return -self.log_prob(model_output, target, scaling_factor, gene_id)

    @property
    def dispersion(self):
        return torch.exp(self.log_r) + 1


class ZINB_Module(OutputModule):

    def __init__(self, fc, out_features, r_init=2, pi_init=0.5, scaling_type="sum"):
        """Args:
        fc: NN (see parent class)
        out_features: number of features from the data modelled by this module
        r_init: initial dispersion factor for all features
        scaling_type: describes type of transformation from model output to targets
            and determines what activation function is used on the output
        """
        super(ZINB_Module, self).__init__(fc, out_features)

        # substracting 1 now and adding it to the learned dispersion ensures a minimum value of 1
        self.log_r = torch.nn.Parameter(
            torch.full(fill_value=math.log(r_init - 1),
                       size=(1, out_features)),
            requires_grad=True,
        )
        self.logit_pi = torch.nn.Parameter(
            torch.full(fill_value=math.log(pi_init / (1 - pi_init)),
                       size=(1, out_features)),
            requires_grad=True,
        )
        self._scaling_type = scaling_type
        if self._scaling_type == "sum":  # could later re-implement more scalings, but sum is arguably the best so far
            self._activation = "softmax"
        elif self._scaling_type == "mean":
            self._activation = "softplus"
        elif self._scaling_type == "max":
            self._activation = "sigmoid"
        else:
            raise ValueError(
                "scaling_type must be one of 'sum', 'mean', or 'max', but is "
                + self._scaling_type
            )

    def forward(self, x):
        for i in range(len(self.fc)):
            x = self.fc[i](x)
        if self._activation == "softmax":
            return F.softmax(x, dim=-1)
        elif self._activation == "softplus":
            return F.softplus(x)
        elif self._activation == "sigmoid":
            return F.sigmoid(x)
        else:
            return x

    @staticmethod
    def rescale(scaling_factor, model_output):
        return scaling_factor * model_output

    def log_prob(self, model_output, target, scaling_factor, feature_id=None):
        # target is the true value
        # the model output represents the mean normalized count
        # the scaling factor is the used normalization
        if feature_id is not None:
            mu = self.rescale(scaling_factor, model_output)[0, feature_id]
            r = (torch.exp(self.log_r) + 1)[0, feature_id]
            pi = torch.sigmoid(self.logit_pi)[0, feature_id]
        else:
            mu = self.rescale(scaling_factor, model_output)
            r = (torch.exp(self.log_r) + 1)
            pi = torch.sigmoid(self.logit_pi)

        log_prob = logZINBdensity(target, mu, r, pi)

        return log_prob

    def loss(self, model_output, target, scaling_factor, gene_id=None):
        return -self.log_prob(model_output, target, scaling_factor, gene_id)

    @property
    def dispersion(self):
        return torch.exp(self.log_r) + 1
    
    @property
    def pi(self):
        return torch.sigmoid(self.logit_pi)


import math
import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np


class RepresentationLayer(torch.nn.Module):
    """
    Implements a representation layer, that accumulates pytorch gradients.

    Representations are vectors in n_rep-dimensional real space. By default
    they will be initialized as a tensor of dimension n_sample x n_rep at origin (zero).

    One can also supply a tensor to initialize the representations (values=tensor).
    The representations will then have the same dimension and will assumes that
    the first dimension is n_sample (and the last is n_rep).

    The representations can be updated once per epoch by standard pytorch optimizers.

    Attributes
    ----------
    n_rep: int
        dimensionality of the representation space
    n_sample: int
        number of samples to be modelled (has to match corresponding dataset)
    z: torch.nn.parameter.Parameter
        tensor of learnable representations of shape (n_sample,n_rep)

    Methods
    ----------
    forward(idx=None)
        takes sample index and returns corresponding representation
    """

    def __init__(self, n_rep: int, n_sample: int, value_init="zero"):
        """Args:
        n_rep: dimensionality of the representation
        n_sample: number of samples to be modelled by the representation
        value_init: per default set to `zero`, leading to an initialization
            of all representations at origin.
            Can also be a tensor of shape (n_sample, n_rep) with custom initialization values
        """
        super(RepresentationLayer, self).__init__()

        self.n_rep = n_rep
        self.n_sample = n_sample

        if value_init == "zero":
            self._value_init = "zero"
            self.z = torch.nn.Parameter(
                torch.zeros(size=(self.n_sample, self.n_rep)), requires_grad=True
            )
        else:
            self._value_init = "custom"
            # Initialize representations from a tensor with values
            assert value_init.shape == (self.n_sample, self.n_rep)
            if isinstance(value_init, torch.Tensor):
                self.z = torch.nn.Parameter(value_init, requires_grad=True)
            else:
                try:
                    self.z = torch.nn.Parameter(
                        torch.Tensor(value_init), requires_grad=True
                    )
                except:
                    raise ValueError(
                        "not able to transform representation init values to torch.Tensor"
                    )

    def forward(self, idx=None):
        """
        Forward pass returns indexed representations
        """
        if idx is None:
            return self.z
        else:
            return self.z[idx]

    def __str__(self):
        return f"""
        RepresentationLayer:
            Dimensionality: {self.n_rep}
            Number of samples: {self.n_sample}
            Value initialization: {self._value_init}
        """


class gaussian:
    """
    This is a simple Gaussian prior used for initializing mixture model means

    Attributes
    ----------
    dim: int
        dimensionality of the space in which samples live
    mean: float
        value of the intended mean of the Normal distribution
    stddev: float
        value of the intended standard deviation

    Methods
    ----------
    sample(n)
        generates samples from the prior
    log_prob(z)
        returns log probability of a vector
    """

    def __init__(self, dim: int, mean: float, stddev: float):
        """Args:
        dim: dimensionality of the latent space
        mean: value for the mean of the prior
        stddev: value for the standard deviation of the prior
        """

        self.dim = dim
        self.mean = mean
        self.stddev = stddev
        self._distrib = torch.distributions.normal.Normal(mean, stddev)

    def sample(self, n):
        """sampling from torch Normal distribution"""
        return self._distrib.sample((n, self.dim))

    def log_prob(self, x):
        """compute log probability of the gaussian prior"""
        return self._distrib.log_prob(x)


class softball:
    """
    Approximate mollified uniform prior.
    It can be imagined as an m-dimensional ball.

    The logistic function creates a soft (differentiable) boundary.
    The prior takes a tensor with a batch of z
    vectors (last dim) and returns a tensor of prior log-probabilities.
    The sample function returns n samples from the prior (approximate
    samples uniform from the m-ball). NOTE: APPROXIMATE SAMPLING.

    Attributes
    ----------
    dim: int
        dimensionality of the space in which samples live
    radius: int
        radius of the m-ball
    sharpness: int
        sharpness of the differentiable boundary

    Methods
    ----------
    sample(n)
        generates samples from the prior with APPROXIMATE sampling
    log_prob(z)
        returns log probability of a vector
    """

    def __init__(self, dim: int, radius: int, sharpness=1):
        """Args:
        dim: dimensionality of the latent space
        radius: radius of the imagined m-ball
        sharpness: sharpness of the differentiable boundary
        """

        self.dim = dim
        self.radius = radius
        self.sharpness = sharpness
        self._norm = math.lgamma(1 + dim * 0.5) - dim * (
            math.log(radius) + 0.5 * math.log(math.pi)
        )

    def sample(self, n):
        """APPROXIMATE sampling of the softball prior"""
        # Return n random samples
        # Approximate: We sample uniformly from n-ball
        with torch.no_grad():
            # Gaussian sample
            sample = torch.randn((n, self.dim))
            # n random directions
            sample.div_(sample.norm(dim=-1, keepdim=True))
            # n random lengths
            local_len = self.radius * \
                torch.pow(torch.rand((n, 1)), 1.0 / self.dim)
            sample.mul_(local_len.expand(-1, self.dim))
        return sample

    def log_prob(self, z):
        """compute log probability of the softball prior"""
        # Return log probabilities of elements of tensor (last dim assumed to be z vectors)
        return self._norm - torch.log(
            1 + torch.exp(self.sharpness * (z.norm(dim=-1) / self.radius - 1))
        )


class GaussianMixture(nn.Module):
    """
    A mixture of multi-variate Gaussians.

    m_mix_comp is the number of components in the mixture
    dim is the dimension of the space
    covariance_type can be "fixed", "isotropic" or "diagonal"
    the mean_prior is initialized as a softball (mollified uniform) with
        mean_init(<radius>, <hardness>)
    log_var_prior is a prior class for the negative log variance of the mixture components
        - log_var = log(sigma^2)
        - If it is not specified, we make this prior a Gaussian from sd_init parameters
        - For the sake of interpretability, the sd_init parameters represent the desired mean and (approximately) sd of the standard deviation
        - the difference btw giving a prior beforehand and giving only init values is that with a given prior, the log_var will be sampled from it, otherwise they will be initialized the same
    alpha determines the Dirichlet prior on mixture coefficients
    Mixture coefficients are initialized uniformly
    Other parameters are sampled from prior

    Attributes
    ----------
    dim: int
        dimensionality of the space
    n_mix_comp: int
        number of mixture components
    mean: torch.nn.parameter.Parameter
        learnable parameter for the GMM means with shape (n_mix_comp,dim)
    log_var: torch.nn.parameter.Parameter
        learnable parameter for the log-variance of the components
        shape depends on what covariances we take into account
            diagonal: (n_mix_comp, dim)
            isotropic: (n_mix_comp)
            fixed: 0
    weight: torch.nn.parameter.Parameter
        learnable parameter for the component weights with shape (n_mix_comp)

    Methods
    ----------
    forward(x)
        returning negative log probability density of a set of representations
    log_prob(x)
        computes summed log probability density
    sample(n_sample)
        creates n_sample random samples from the GMM (taking into account mixture weights)
    component_sample(n_sample)
        creates n_sample new samples PER mixture component
    sample_probs(x)
        computes log-probs per sample (not summed and not including priors)
    sample_new_points(n_points, option='random', n_new=1)
        creating new samples either like in component_sample or from component means
    reshape_targets(y, y_type='true')
        reshaping targets (true counts) to be comparable to model outputs
        from multiple representations per sample
    choose_best_representations(x, losses)
        reducing new representations to 1 per sample (based on lowest loss)
    choose_old_or_new(z_new, loss_new, z_old, loss_old)
        selecting best representation per sample between tow representation tensors (pairwise)
    """

    def __init__(
        self,
        n_mix_comp: int,
        dim: int,
        covariance_type="diagonal",
        mean_init=(2.0, 5.0),
        sd_init=(0.5, 1.0),
        weight_alpha=1,
    ):
        """Args:
        n_mix_comp: number of mixture components
        dim: dimensionality of the latent space (or at least of the corresponding representation)
        covariance_type: string variable determining the portion of the full covariance matrix used
            can be
                `fixed`: all components have the same (not learnable) variance in every dimension
                `isotropic`: every component has 1 learnable variance
                `diagonal`: gives covariance matrix of shape (n_mix_comp,dim)
        mean_init: tuple of mean and std used for the prior over means
            (from which the component means are sampled at initialization)
        sd_init: first value presents the intended mean of the prior over standard deviation
            this prior and corresponding learnable parameter (log_var) are learned as the log-variance
            and the first sd_init value is transformed accordingly.
            The second value presents the standard deviation of the log_var prior.
            This normal distribution over log-variance practically approximates well
            an inverse gamma distribution over variance (found this to be used in Bayesian statistics
             as the marginal posterior for the variance of a Gaussian).
        weight_alpha: concentration parameter of the dirichlet prior
        """
        super().__init__()

        # dimensionality of space and number of components
        self.dim = dim
        self.n_mix_comp = n_mix_comp

        # initialize public parameters
        self._init_means(mean_init)
        self._init_log_var(sd_init, covariance_type)
        self._init_weights(weight_alpha)

        # a dimensionality-dependent term needed in PDF
        self._pi_term = -0.5 * self.dim * math.log(2 * math.pi)

    def _init_means(self, mean_init):
        self._mean_prior = softball(self.dim, mean_init[0], mean_init[1])
        self._mean = nn.Parameter(
            self._mean_prior.sample(self.n_mix_comp), requires_grad=True
        )

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(
        self, value
    ):  # forbid user from changing this parameter outside .load_state_dict
        raise ValueError("GMM mean may not be changed")

    def _init_log_var(self, sd_init, covariance_type):
        # init parameter to learn covariance matrix (as negative log variance to ensure it to be positive definite)
        self._sd_init = sd_init
        self._log_var_factor = self.dim * 0.5  # dimensionality factor in PDF
        self._log_var_dim = 1  # If 'diagonal' the dimension of is dim
        if covariance_type == "fixed":
            # here there are no gradients needed for training
            # this would mainly be used to assume a standard Gaussian
            self._log_var = nn.Parameter(
                torch.empty(self.n_mix_comp, self._log_var_dim), requires_grad=False
            )
        else:
            if covariance_type == "diagonal":
                self._log_var_factor = 0.5
                self._log_var_dim = self.dim
            elif covariance_type != "isotropic":
                raise ValueError(
                    "type must be 'isotropic' (default), 'diagonal', or 'fixed'"
                )

            self._log_var = nn.Parameter(
                torch.empty(self.n_mix_comp, self._log_var_dim), requires_grad=True
            )
        with torch.no_grad():
            self._log_var.fill_(2 * math.log(sd_init[0]))
        self._log_var_prior = gaussian(
            self._log_var_dim, -2 * math.log(sd_init[0]), sd_init[1]
        )
        # this needs to have the negative log variance as a mean to ensure the approximation of
        # an inverse gamma over the variance

    @property
    def log_var(self):
        return self._log_var

    @log_var.setter
    def log_var(
        self, value
    ):  # forbid user from changing this parameter outside .load_state_dict
        raise ValueError("GMM log-variance may not be changed")

    def _init_weights(self, alpha):
        """i.e. Dirichlet prior on mixture"""
        # dirichlet alpha determining the uniformity of the weights
        self._weight_alpha = alpha
        self._dirichlet_constant = math.lgamma(
            self.n_mix_comp * self._weight_alpha
        ) - self.n_mix_comp * math.lgamma(self._weight_alpha)
        # weights are initialized uniformly so that components start out equi-probable
        self._weight = nn.Parameter(torch.ones(
            self.n_mix_comp), requires_grad=True)

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(
        self, value
    ):  # forbid user from changing this parameter outside .load_state_dict
        raise ValueError("GMM weigths may not be changed")

    def forward(self, x):
        """
        Forward pass computes the negative log density
        of the probability of z being drawn from the mixture model
        """

        # y = logp = - 0.5k*log(2pi) -(0.5*(x-mean[i])^2)/variance - 0.5k*log(variance)
        # sum terms for each component (sum is over last dimension)
        # x is unsqueezed to (n_sample,1,dim), so broadcasting of mean (n_mix_comp,dim) works
        y = -(x.unsqueeze(-2) - self.mean.to(x.device)
              ).square().div(2 * self.covariance.to(x.device)).sum(-1)
        y = y - self._log_var_factor * self.log_var.to(x.device).sum(-1)
        y = y + self._pi_term

        # For each component multiply by mixture probs
        y = y + torch.log_softmax(self.weight.to(x.device), dim=0)
        y = torch.logsumexp(y, dim=-1)
        y = y + self._prior_log_prob()  # += gives cuda error

        return -y  # returning negative log probability density

    def _prior_log_prob(self):
        """Calculate log prob of prior on mean, log_var, and mixture coefficients"""
        # Mixture weights
        p = self._dirichlet_constant
        if self._weight_alpha != 1:
            p = p + (self._weight_alpha - 1.0) * \
                (self.mixture_probs().log().sum())
        # Means
        p = p + self._mean_prior.log_prob(self.mean).sum()
        # log_var
        if self._log_var_prior is not None:
            p = (
                p + self._log_var_prior.log_prob(-self.log_var).sum()
            )  # ensuring correct approximation
        return p

    def log_prob(self, x):
        """return the log density of the probability of z being drawn from the mixture model"""
        return -self.forward(x)

    def mixture_probs(self):
        """transform weights to mixture probabilites"""
        return torch.softmax(self.weight, dim=-1)

    @property
    def covariance(self):
        """transform negative log variance into covariances"""
        return torch.exp(self.log_var)

    @property
    def stddev(self):
        return torch.sqrt(self.covariance)

    def _Distribution(self):
        """create a distribution from mixture model (for sampling)"""
        with torch.no_grad():
            mix = D.Categorical(probs=torch.softmax(self.weight, dim=-1))
            comp = D.Independent(D.Normal(self.mean, self.stddev), 1)
            return D.MixtureSameFamily(mix, comp)

    def sample(self, n_sample):
        """create samples from the GMM distribution"""
        with torch.no_grad():
            gmm = self._Distribution()
            return gmm.sample(torch.tensor([n_sample]))

    def component_sample(self, n_sample):
        """Returns a sample from each component. Tensor shape (n_sample,n_mix_comp,dim)"""
        with torch.no_grad():
            comp = D.Independent(D.Normal(self.mean, self.stddev), 1)
            return comp.sample(torch.tensor([n_sample]))

    def sample_probs(self, x):
        """compute probability densities per sample without prior. returns tensor of shape (n_sample, n_mix_comp)"""
        y = -(x.unsqueeze(-2) - self.mean).square().div(2 * self.covariance).sum(-1)
        y = y - self._log_var_factor * self.log_var.sum(-1)
        y = y + self._pi_term
        y = y + torch.log_softmax(self.weight, dim=0)
        return torch.exp(y)

    def __str__(self):
        return f"""
        Gaussian_mix_compture:
            Dimensionality: {self.dim}
            Number of components: {self.n_mix_comp}
        """

    def sample_new_points(self, resample_type="mean", n_new_samples=1):
        """
        creates a Tensor with potential new representations.
        These can be drawn from component samples if resample_type is 'sample' or
        from the mean if 'mean'. For drawn samples, n_new_samples defines the number
        of random samples drawn from each component.
        """

        if resample_type == "mean":
            samples = self.mean.clone().cpu().detach()
        else:
            samples = (
                self.component_sample(
                    n_new_samples).view(-1, self.dim).cpu().detach()
            )
        return samples

    def clustering(self, x):
        """compute the cluster assignment (as int) for each sample"""
        return torch.argmax(self.sample_probs(x), dim=-1).to(torch.int16)


class GaussianMixtureSupervised(GaussianMixture):
    """
    Supervised GaussianMixutre class.

    Attributes
    ----------
    Nclass: int
        number of classes to be modeled
    Ncpc: int
        number of components that should model each class
    """

    def __init__(
            self,
            Nclass: int,
            Ncompperclass: int,
            dim: int,
            covariance_type="diagonal",
            mean_init=(2.0, 5.0),
            sd_init=(0.5, 1.0),
            weight_alpha=1
    ):
        super(GaussianMixtureSupervised, self).__init__(
            Nclass*Ncompperclass, dim, covariance_type, mean_init, sd_init, weight_alpha)

        self.Nclass = Nclass  # number of classes in the data
        self.Ncpc = Ncompperclass  # number of components per class

    def forward(self, x, label=None):

        # return unsupervized loss if there are no labels provided
        if label is None:
            y = super().forward(x)
            return y

        y = - (x.unsqueeze(-2).unsqueeze(-2) - self.mean.view(self.Nclass, self.Ncpc, -1)
               ).square().div(2 * self.covariance.view(self.Nclass, self.Ncpc, -1)).sum(-1)
        y = y + self._log_var_factor * \
            self.log_var.view(self.Nclass, self.Ncpc, -1).sum(-1)
        y = y + self._pi_term
        y += torch.log_softmax(self.weight.view(self.Nclass,
                               self.Ncpc), dim=-1)
        y = y.sum(-1)
        # this is replacement for logsumexp of supervised samples
        y = y[(np.arange(y.shape[0]), label)] * self.Nclass

        y = y + self._prior_log_prob()
        return - y

    def label_mixture_probs(self, label):
        return torch.softmax(self.weight[label], dim=-1)

    def supervised_sampling(self, label, sample_type='random'):
        # get samples for each component
        if sample_type == 'origin':
            # choose the component means
            samples = self.mean.clone().detach().unsqueeze(0).repeat(len(label), 1, 1)
        else:
            samples = self.component_sample(len(label))
        # then select the correct component
        return samples[range(len(label)), label]


import torch
from tqdm import tqdm
from base.dgd.latent import RepresentationLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_potential_reps(sample_list):
    """
    takes a list of samples drawn from the DGD's distributions.
    The length gives the number of distributions which defines
    the dimensionality of the output tensor.
    If the list of samples is longer than 1, we will create representations
    from the combination of each GMM's samples.
    """
    return sample_list[0]
    
def learn_new_representation(dgd, 
                             data_loader,
                             test_epochs=50,
                             learning_rates=1e-2, 
                             weight_decay=0.,
                             betas=(0.5, 0.7),
                             reduction_type="sum",
                             resampling_type="mean"):
    """
    This function learns a new representation layer for the DGD.
    The new representation layer is learned by sampling new points
    from the GMMs and finding the best fitting GMM for each sample.
    The new representation layer is then optimized to minimize the
    reconstruction loss of the DGD.
    """
    
    gmm_loss = True
    n_samples_new = len(data_loader.dataset)
    potential_reps = prepare_potential_reps([dgd.gmm.sample_new_points(resampling_type)])

    dgd.eval()
    X_mirna, X_mrna = dgd.decoder(potential_reps.to(device))

    rep_init_values = torch.zeros((n_samples_new, potential_reps.shape[-1]))

    for (mrna_data, mirna_data, lib_mrna, lib_mirna, i) in tqdm(data_loader.dataset):
        loss = torch.empty(0).to(device)
        for X in X_mrna:
            mrna_recon_loss = dgd.decoder.loss(
                nn_output=X.to(device), 
                target=mrna_data.to(device), 
                scale=lib_mrna, 
                mod_id="mrna", 
                feature_ids=None, 
                reduction="sum", 
                type="midgd"
            )
            loss = torch.cat((loss, mrna_recon_loss.unsqueeze(0)))
        best_fit_ids = torch.argmin(loss, dim=-1).detach().cpu()
        rep_init_values[i, :] = potential_reps.clone()[best_fit_ids, :]

    Ntest=len(data_loader.dataset)
    new_rep = RepresentationLayer(n_rep=dgd.rep_dim, 
                                  n_sample=Ntest,
                                  value_init=rep_init_values).to(device)
    test_rep_optimizer = torch.optim.AdamW(new_rep.parameters(), lr=learning_rates, weight_decay=weight_decay, betas=betas)

    for epoch in tqdm(range(test_epochs)):
        test_rep_optimizer.zero_grad()
        for (mrna_data, mirna_data, lib_mrna, lib_mirna, index) in data_loader:
            mirna_recon_loss, mrna_recon_loss, gmm_loss = dgd.forward_and_loss(
                z=new_rep(index),
                target=[mirna_data.to(device), mrna_data.to(device)],
                scale=[lib_mirna.unsqueeze(1).to(device), lib_mrna.unsqueeze(1).to(device)], 
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="combined"
            )
            loss = mrna_recon_loss + gmm_loss
            loss.backward()
        test_rep_optimizer.step()
    
    return new_rep 


def learn_new_representation_mrna(dgd, 
                             data_loader,
                             test_epochs=50,
                             learning_rates=1e-2, 
                             weight_decay=0.,
                             betas=(0.5, 0.7),
                             reduction_type="sum",
                             resampling_type="mean"):
    """
    This function learns a new representation layer for the DGD.
    The new representation layer is learned by sampling new points
    from the GMMs and finding the best fitting GMM for each sample.
    The new representation layer is then optimized to minimize the
    reconstruction loss of the DGD.
    """
    
    gmm_loss = True
    n_samples_new = len(data_loader.dataset)
    potential_reps = prepare_potential_reps([dgd.gmm.sample_new_points(resampling_type)])

    dgd.eval()
    X_mirna, X_mrna = dgd.decoder(potential_reps.to(device))

    rep_init_values = torch.zeros((n_samples_new, potential_reps.shape[-1]))

    for (mrna_data, mirna_data, lib_mrna, lib_mirna, i) in tqdm(data_loader.dataset):
        loss = torch.empty(0).to(device)
        for X in X_mirna:
            mirna_recon_loss = dgd.decoder.loss(
                nn_output=X.to(device), 
                target=mirna_data.to(device), 
                scale=lib_mirna, 
                mod_id="mirna", 
                feature_ids=None, 
                reduction="sum", 
                type="midgd"
            )
            loss = torch.cat((loss, mirna_recon_loss.unsqueeze(0)))
        best_fit_ids = torch.argmin(loss, dim=-1).detach().cpu()
        rep_init_values[i, :] = potential_reps.clone()[best_fit_ids, :]

    Ntest=len(data_loader.dataset)
    new_rep = RepresentationLayer(n_rep=dgd.rep_dim, 
                                  n_sample=Ntest,
                                  value_init=rep_init_values).to(device)
    test_rep_optimizer = torch.optim.AdamW(new_rep.parameters(), lr=learning_rates, weight_decay=weight_decay, betas=betas)

    for epoch in tqdm(range(test_epochs)):
        test_rep_optimizer.zero_grad()
        for (mrna_data, mirna_data, lib_mrna, lib_mirna, index) in data_loader:
            mirna_recon_loss, mrna_recon_loss, gmm_loss = dgd.forward_and_loss(
                z=new_rep(index),
                target=[mirna_data.to(device), mrna_data.to(device)],
                scale=[lib_mirna.unsqueeze(1).to(device), lib_mrna.unsqueeze(1).to(device)], 
                gmm_loss=gmm_loss,
                reduction=reduction_type,
                type="combined"
            )
            loss = mirna_recon_loss + gmm_loss
            loss.backward()
        test_rep_optimizer.step()
    
    return new_rep 


import torch
from torch.utils.data import Dataset, Sampler
import numpy as np


class GeneExpressionDatasetCombined(Dataset):
    '''
    Creates a Dataset class for gene expression dataset including both mRNA and miRNA data.
    The rows of the dataframe contain samples, and the columns contain gene expression values.
    '''

    def __init__(self, mrna_data, mirna_data, label_position=-1, color_position = -2, 
                 sample_position = -3, tissue_position = -4, scaling_type='mean'):
        '''
        Args:
            mrna_data: pandas dataframe containing mRNA input data
            mirna_data: pandas dataframe containing miRNA input data
            label_position: column id of the class labels (assumed to be the same for both dataframes)
            scaling_type: type of scaling to apply ('mean' or 'max')
        '''
        self.scaling_type = scaling_type
        self.label_position = label_position
        self.color_position = color_position
        self.sample_position = sample_position
        self.tissue_position = tissue_position

        # Assuming labels are the same for both mrna and mirna data and are located in the same column
        self.label = mrna_data.iloc[:, label_position].values
        self.color = mrna_data.iloc[:, color_position].values
        self.sample_type = mrna_data.iloc[:, sample_position].values
        self.tissue_type = mrna_data.iloc[:, tissue_position].values

        # Convert data to tensors and remove label columns
        self.mrna_data = torch.tensor(mrna_data.drop(
            mrna_data.columns[[tissue_position, sample_position, color_position, label_position]], axis=1).values).float()
        self.mirna_data = torch.tensor(mirna_data.drop(
            mirna_data.columns[[tissue_position, sample_position, color_position, label_position]], axis=1).values).float()

    def __len__(self):
        # Assuming both mrna_data and mirna_data have the same number of samples
        return self.mrna_data.shape[0]

    def __getitem__(self, idx):
        if idx is None:
            idx = np.arange(self.__len__())
        mrna_expression = self.mrna_data[idx, :]
        mirna_expression = self.mirna_data[idx, :]

        # Apply scaling if specified
        if self.scaling_type == 'mean':
            mrna_lib = torch.mean(mrna_expression, dim=-1)
            mirna_lib = torch.mean(mirna_expression, dim=-1)
        elif self.scaling_type == 'max':
            mrna_lib = torch.max(mrna_expression, dim=-1).values
            mirna_lib = torch.max(mirna_expression, dim=-1).values
        elif self.scaling_type == 'sum':
            mrna_lib = torch.sum(mrna_expression, dim=-1)
            mirna_lib = torch.sum(mirna_expression, dim=-1)

        return mrna_expression, mirna_expression, mrna_lib, mirna_lib, idx

    def __getlabel__(self, idx=None):
        if idx is None:
            idx = np.arange(self.__len__())
        return self.label[idx]
    

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
