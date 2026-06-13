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
            local_len = self.radius * torch.pow(torch.rand((n, 1)), 1.0 / self.dim)
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
        self._weight = nn.Parameter(torch.ones(self.n_mix_comp), requires_grad=True)

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
        # y = -(x.unsqueeze(-2) - self.mean).square().div(2 * self.covariance).sum(-1)
        y = -(x.unsqueeze(-2) - self.mean.to(x.device)).square().div(2 * self.covariance.to(x.device)).sum(-1)
        y = y - self._log_var_factor * self.log_var.sum(-1)
        y = y + self._pi_term
        # For each component multiply by mixture probs
        y = y + torch.log_softmax(self.weight, dim=0)
        y = torch.logsumexp(y, dim=-1)
        y = y + self._prior_log_prob()  # += gives cuda error

        return -y  # returning negative log probability density

    def _prior_log_prob(self):
        """Calculate log prob of prior on mean, log_var, and mixture coefficients"""
        # Mixture weights
        p = self._dirichlet_constant
        if self._weight_alpha != 1:
            p = p + (self._weight_alpha - 1.0) * (self.mixture_probs().log().sum())
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
                self.component_sample(n_new_samples).view(-1, self.dim).cpu().detach()
            )
        return samples
    
    def clustering(self, x):
        """compute the cluster assignment (as int) for each sample"""
        return torch.argmax(self.sample_probs(x),dim=-1).to(torch.int16)

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
            mean_init=(2.0,5.0),
            sd_init=(0.5,1.0),
            weight_alpha=1
            ):
        super(GaussianMixtureSupervised, self).__init__(Nclass*Ncompperclass, dim, covariance_type, mean_init, sd_init, weight_alpha)
        
        self.Nclass = Nclass # number of classes in the data
        self.Ncpc = Ncompperclass # number of components per class

    def forward(self,x,label=None):

        # return unsupervized loss if there are no labels provided
        if label is None:
            y = super().forward(x)
            return y
        
        y = - (x.unsqueeze(-2).unsqueeze(-2) - self.mean.view(self.Nclass,self.Ncpc,-1)).square().div(2 * self.covariance.view(self.Nclass,self.Ncpc,-1)).sum(-1)
        y = y + self._log_var_factor * self.log_var.view(self.Nclass,self.Ncpc,-1).sum(-1)
        y = y + self._pi_term
        y += torch.log_softmax(self.weight.view(self.Nclass,self.Ncpc),dim=-1)
        y = y.sum(-1)
        y = y[(np.arange(y.shape[0]),label)] * self.Nclass # this is replacement for logsumexp of supervised samples
        
        y = y + self._prior_log_prob()
        return - y

    def label_mixture_probs(self,label):
        return torch.softmax(self.weight[label],dim=-1)
    
    def supervised_sampling(self, label, sample_type='random'):
        # get samples for each component
        if sample_type == 'origin':
            # choose the component means
            samples = self.mean.clone().detach().unsqueeze(0).repeat(len(label),1,1)
        else:
            samples = self.component_sample(len(label))
        # then select the correct component
        return samples[range(len(label)),label]