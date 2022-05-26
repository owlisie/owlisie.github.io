import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,
                 device = "cuda",
                 in_dcg=60000,
                 in_ecg=12000,
                 hidden = [1024, 512, 256, 128, 64],
                 latent_size=1024,
                 num_layers=20,
                 bidirectional=True):
        super().__init__()

        self.device = device
        self.in_dcg = in_dcg
        self.in_ecg = in_ecg
        self.hidden = hidden
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.eps = 1e-7

        self.actv = nn.LeakyReLU()

        self.enc1 = nn.Linear(self.in_dcg, self.hidden[0])
        self.enc2 = nn.Linear(self.hidden[0], self.hidden[2])
        self.enc3 = nn.Linear(self.hidden[2], self.hidden[4])

        self.bn_enc1 = nn.BatchNorm1d(self.hidden[0])
        self.bn_enc2 = nn.BatchNorm1d(self.hidden[2])
        self.bn_enc3 = nn.BatchNorm1d(self.hidden[4])

        self.fc_mu = nn.Linear(self.hidden[4], self.latent_size)
        self.fc_logvar = nn.Linear(self.hidden[4], self.latent_size)

    """
        self.enc1 = nn.Linear(self.in_dcg, self.hidden[0])
        self.enc2 = nn.Linear(self.hidden[0], self.hidden[1])
        self.enc3 = nn.Linear(self.hidden[1], self.hidden[2])
        self.enc4 = nn.Linear(self.hidden[2], self.hidden[3])
        self.enc5 = nn.Linear(self.hidden[3], self.hidden[4])
        self.enc6 = nn.Linear(self.hidden[4], self.hidden[5])

        self.bn_enc1 = nn.BatchNorm1d(self.hidden[0])
        self.bn_enc2 = nn.BatchNorm1d(self.hidden[1])
        self.bn_enc3 = nn.BatchNorm1d(self.hidden[2])
        self.bn_enc4 = nn.BatchNorm1d(self.hidden[3])
        self.bn_enc5 = nn.BatchNorm1d(self.hidden[4])
        self.bn_enc6 = nn.BatchNorm1d(self.hidden[5])

        self.fc_mu = nn.Linear(self.hidden[5], self.latent_size)
        self.fc_logvar = nn.Linear(self.hidden[5], self.latent_size)
        
    def encode(self, x):
        x = torch.nan_to_num(self.enc1(x), nan=self.eps)
        x = self.actv(x)
        x = self.bn_enc1(x)
        x = torch.nan_to_num(self.enc2(x), nan=self.eps)
        x = self.actv(x)
        x = self.bn_enc2(x)
        x = torch.nan_to_num(self.enc3(x), nan=self.eps)
        x = self.actv(x)
        x = self.bn_enc3(x)
        x = torch.nan_to_num(self.enc4(x), nan=self.eps)
        x = self.actv(x)
        x = self.bn_enc4(x)
        x = torch.nan_to_num(self.enc5(x), nan=self.eps)
        x = self.actv(x)
        x = self.bn_enc5(x)
        x = torch.nan_to_num(self.enc6(x), nan=self.eps)
        x = self.actv(x)
        x = self.bn_enc6(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar
    """

    def encode(self, x):
        x = torch.nan_to_num(self.enc1(x), nan=self.eps)
        x = self.actv(x)
        x = self.bn_enc1(x)
        x = torch.nan_to_num(self.enc2(x), nan=self.eps)
        x = self.actv(x)
        x = self.bn_enc2(x)
        x = torch.nan_to_num(self.enc3(x), nan=self.eps)
        x = self.actv(x)
        x = self.bn_enc3(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.in_dcg))
        z = self.sampling(mu, logvar)

        return mu, logvar, z

def generate_model(**kwargs):
    model = MLP(**kwargs)

    return model
