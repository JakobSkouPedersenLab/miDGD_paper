from base.dgd.latent import GaussianMixture, GaussianMixtureSupervised
import torch.nn as nn
import torch

class DGD(nn.Module):
    def __init__(self, decoder, n_mix, rep_dim, gmm_spec={}, is_supervised=False, comp_per_class = 2):
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
    
class SupervisedDGD(DGD):
    def __init__(self, 
                 decoder, 
                 n_mix, 
                 rep_dim, 
                 gmm_spec={}, 
                 comp_per_class = 2):
        super(SupervisedDGD, self).__init__(decoder, n_mix, rep_dim, gmm_spec=gmm_spec, comp_per_class=comp_per_class)
        self.gmm = GaussianMixtureSupervised(n_mix, comp_per_class, rep_dim, **gmm_spec, covariance_type="diagonal")
    
    def loss(self, z, y, target, scale, gmm_loss=True, reduction="sum", type="combined", label=None):
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
            self.gmm_loss = self.gmm(z, label)
            if reduction == "mean":
                self.gmm_loss = self.gmm_loss.mean()
            elif reduction == "sum":
                self.gmm_loss = self.gmm_loss.sum()

            if type == "combined":
                return self.dec_loss_mirna, self.dec_loss_mrna, self.gmm_loss
            else:
                return self.dec_loss, self.gmm_loss
    
    def forward_and_loss(self, z, target, scale, gmm_loss=True, reduction="sum", type="combined", label=None):
        y = self.decoder(z)

        return self.loss(z, y, target, scale, gmm_loss, reduction, type, label)


class CovarDGD(DGD):
    def __init__(self, 
                 decoder, 
                 n_mix, 
                 n_covar,
                 comp_per_covar,
                 rep_dim,
                 rep_covar_dim, 
                 gmm_spec={},
                 gmm_covar_spec={}):
        super(CovarDGD, self).__init__(
            decoder=decoder,
            n_mix=n_mix,
            rep_dim=rep_dim,       # Position matches DGD's 3rd param
            gmm_spec=gmm_spec      # Explicit keyword
        )
        self.rep_covar_dim = rep_covar_dim 
        self.gmm_covar = GaussianMixtureSupervised(n_covar, comp_per_covar, rep_covar_dim, **gmm_covar_spec, covariance_type="diagonal")
    
    def loss(self, z_rep, z_covar, y, target, scale, gmm_loss=True, reduction="sum", type="combined", label=None):
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
            self.gmm_loss = self.gmm(z_rep)
            self.gmm_covar_loss = self.gmm_covar(z_covar, label)

            if reduction == "mean":
                self.gmm_loss = self.gmm_loss.mean()
                self.gmm_covar_loss = self.gmm_covar_loss.mean()

            elif reduction == "sum":
                self.gmm_loss = self.gmm_loss.sum()
                self.gmm_covar_loss = self.gmm_covar_loss.sum()

            if type == "combined":
                return self.dec_loss_mirna, self.dec_loss_mrna, self.gmm_loss, self.gmm_covar_loss
            else:
                return self.dec_loss, self.gmm_loss, self.gmm_covar_loss
    
    def forward(self, z_rep, z_covar):
        z_concat = torch.cat((z_rep, z_covar), dim=1)

        return self.decoder(z_concat)

    def forward_and_loss(self, z_rep, z_covar, target, scale, gmm_loss=True, reduction="sum", type="combined", label=None):
        # Forward pass through the decoder
        y = self.forward(z_rep, z_covar)

        return self.loss(z_rep, z_covar, y, target, scale, gmm_loss, reduction, type, label)
    
    def get_representations_covar(self, type="train"):
        if type == "train":
            return self.train_rep_covar.z.detach().cpu().numpy()
        elif type == "val":
            return self.val_rep_covar.z.detach().cpu().numpy()
        elif type == "test":
            return self.test_rep_covar.z.detach().cpu().numpy()

    def get_gmm_covar_means(self):
        return self.gmm_covar.mean.detach().cpu().numpy()

    def get_covar_latent_space_values(self, rep_type="train", n_samples=1000):
        # get representations
        if rep_type == "train":
            rep = self.train_rep_covar.z.clone().detach().cpu().numpy()
        elif rep_type == "val":
            rep = self.val_rep_covar.z.clone().detach().cpu().numpy()
        elif rep_type == "test":
            rep = self.test_rep_covar.z.clone().detach().cpu().numpy()

        # get gmm means
        gmm_means = self.gmm_covar.mean.clone().detach().cpu().numpy()

        # get some gmm samples
        gmm_samples = self.gmm_covar.sample(n_samples).detach().cpu().numpy()

        return rep, gmm_means, gmm_samples