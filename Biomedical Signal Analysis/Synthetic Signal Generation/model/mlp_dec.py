import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,
                 device = "cuda",
                 in_dcg=60000,
                 in_ecg=12000,
                 hidden = [1024, 512, 256, 128, 64, 32, 16],
                 latent_size=100,
                 num_layers=7,
                 bidirectional=True):
        super().__init__()

        self.device = device
        self.in_dcg = in_dcg
        self.in_ecg = in_ecg
        self.hidden = hidden
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.eps = 1e-5

        self.actv = nn.LeakyReLU()

        """
        self.dec_dcg1 = nn.Linear(self.latent_size, self.hidden[5])
        self.dec_dcg2 = nn.Linear(self.hidden[5], self.hidden[4])
        self.dec_dcg3 = nn.Linear(self.hidden[4], self.hidden[3])
        self.dec_dcg4 = nn.Linear(self.hidden[3], self.hidden[2])
        self.dec_dcg5 = nn.Linear(self.hidden[2], self.hidden[1])
        self.dec_dcg6 = nn.Linear(self.hidden[1], self.hidden[0])
        self.dec_dcg7 = nn.Linear(self.hidden[0], self.in_dcg)

        self.bn_dec_dcg1 = nn.BatchNorm1d(self.hidden[5])
        self.bn_dec_dcg2 = nn.BatchNorm1d(self.hidden[4])
        self.bn_dec_dcg3 = nn.BatchNorm1d(self.hidden[3])
        self.bn_dec_dcg4 = nn.BatchNorm1d(self.hidden[2])
        self.bn_dec_dcg5 = nn.BatchNorm1d(self.hidden[1])
        self.bn_dec_dcg6 = nn.BatchNorm1d(self.hidden[0])

        self.dec_ecg1 = nn.Linear(self.latent_size, self.hidden[5])
        self.dec_ecg2 = nn.Linear(self.hidden[5], self.hidden[4])
        self.dec_ecg3 = nn.Linear(self.hidden[4], self.hidden[3])
        self.dec_ecg4 = nn.Linear(self.hidden[3], self.hidden[2])
        self.dec_ecg5 = nn.Linear(self.hidden[2], self.hidden[1])
        self.dec_ecg6 = nn.Linear(self.hidden[1], self.hidden[0])
        self.dec_ecg7 = nn.Linear(self.hidden[0], self.in_ecg)

        self.bn_dec_ecg1 = nn.BatchNorm1d(self.hidden[5])
        self.bn_dec_ecg2 = nn.BatchNorm1d(self.hidden[4])
        self.bn_dec_ecg3 = nn.BatchNorm1d(self.hidden[3])
        self.bn_dec_ecg4 = nn.BatchNorm1d(self.hidden[2])
        self.bn_dec_ecg5 = nn.BatchNorm1d(self.hidden[1])
        self.bn_dec_ecg6 = nn.BatchNorm1d(self.hidden[0])

    def decode(self, z):
        x_dcg = torch.nan_to_num(self.dec_dcg1(z), self.eps)
        x_ecg = torch.nan_to_num(self.dec_ecg1(z), self.eps)
        x_dcg = self.actv(x_dcg)
        x_ecg = self.actv(x_ecg)
        x_dcg = self.bn_dec_dcg1(x_dcg)
        x_ecg = self.bn_dec_ecg1(x_ecg)
        x_dcg = torch.nan_to_num(self.dec_dcg2(x_dcg), self.eps)
        x_ecg = torch.nan_to_num(self.dec_ecg2(x_ecg), self.eps)
        x_dcg = self.actv(x_dcg)
        x_ecg = self.actv(x_ecg)
        x_dcg = self.bn_dec_dcg2(x_dcg)
        x_ecg = self.bn_dec_ecg2(x_ecg)
        x_dcg = torch.nan_to_num(self.dec_dcg3(x_dcg), self.eps)
        x_ecg = torch.nan_to_num(self.dec_ecg3(x_ecg), self.eps)
        x_dcg = self.actv(x_dcg)
        x_ecg = self.actv(x_ecg)
        x_dcg = self.bn_dec_dcg3(x_dcg)
        x_ecg = self.bn_dec_ecg3(x_ecg)
        x_dcg = torch.nan_to_num(self.dec_dcg4(x_dcg), self.eps)
        x_ecg = torch.nan_to_num(self.dec_ecg4(x_ecg), self.eps)
        x_dcg = self.actv(x_dcg)
        x_ecg = self.actv(x_ecg)
        x_dcg = self.bn_dec_dcg4(x_dcg)
        x_ecg = self.bn_dec_ecg4(x_ecg)
        x_dcg = torch.nan_to_num(self.dec_dcg5(x_dcg), self.eps)
        x_ecg = torch.nan_to_num(self.dec_ecg5(x_ecg), self.eps)
        x_dcg = self.actv(x_dcg)
        x_ecg = self.actv(x_ecg)
        x_dcg = self.bn_dec_dcg5(x_dcg)
        x_ecg = self.bn_dec_ecg5(x_ecg)
        x_dcg = torch.nan_to_num(self.dec_dcg6(x_dcg), self.eps)
        x_ecg = torch.nan_to_num(self.dec_ecg6(x_ecg), self.eps)
        x_dcg = self.actv(x_dcg)
        x_ecg = self.actv(x_ecg)
        x_dcg = self.bn_dec_dcg6(x_dcg)
        x_ecg = self.bn_dec_ecg6(x_ecg)
        x_dcg = torch.nan_to_num(self.dec_dcg7(x_dcg), self.eps)
        x_ecg = torch.nan_to_num(self.dec_ecg7(x_ecg), self.eps)

        re_dcg = torch.sigmoid(x_dcg)
        re_ecg = torch.sigmoid(x_ecg)

        return re_dcg, re_ecg
        
        """

        self.dec_dcg1 = nn.Linear(self.latent_size, self.hidden[5])
        self.dec_dcg2 = nn.Linear(self.hidden[5], self.hidden[3])
        self.dec_dcg3 = nn.Linear(self.hidden[3], self.hidden[2])
        self.dec_dcg4 = nn.Linear(self.hidden[2], self.hidden[1])
        self.dec_dcg5 = nn.Linear(self.hidden[1], self.in_dcg)

        self.bn_dec_dcg1 = nn.BatchNorm1d(self.hidden[5])
        self.bn_dec_dcg2 = nn.BatchNorm1d(self.hidden[3])
        self.bn_dec_dcg3 = nn.BatchNorm1d(self.hidden[2])
        self.bn_dec_dcg4 = nn.BatchNorm1d(self.hidden[1])

        self.dec_ecg1 = nn.Linear(self.latent_size, self.hidden[5])
        self.dec_ecg2 = nn.Linear(self.hidden[5], self.hidden[3])
        self.dec_ecg3 = nn.Linear(self.hidden[3], self.hidden[2])
        self.dec_ecg4 = nn.Linear(self.hidden[2], self.hidden[1])
        self.dec_ecg5 = nn.Linear(self.hidden[1], self.in_ecg)

        self.bn_dec_ecg1 = nn.BatchNorm1d(self.hidden[5])
        self.bn_dec_ecg2 = nn.BatchNorm1d(self.hidden[3])
        self.bn_dec_ecg3 = nn.BatchNorm1d(self.hidden[2])
        self.bn_dec_ecg4 = nn.BatchNorm1d(self.hidden[1])

    def decode(self, z):
        x_dcg = torch.nan_to_num(self.dec_dcg1(z), self.eps)
        x_ecg = torch.nan_to_num(self.dec_ecg1(z), self.eps)
        x_dcg = self.actv(x_dcg)
        x_ecg = self.actv(x_ecg)
        x_dcg = self.bn_dec_dcg1(x_dcg)
        x_ecg = self.bn_dec_ecg1(x_ecg)
        x_dcg = torch.nan_to_num(self.dec_dcg2(x_dcg), self.eps)
        x_ecg = torch.nan_to_num(self.dec_ecg2(x_ecg), self.eps)
        x_dcg = self.actv(x_dcg)
        x_ecg = self.actv(x_ecg)
        x_dcg = self.bn_dec_dcg2(x_dcg)
        x_ecg = self.bn_dec_ecg2(x_ecg)
        x_dcg = torch.nan_to_num(self.dec_dcg3(x_dcg), self.eps)
        x_ecg = torch.nan_to_num(self.dec_ecg3(x_ecg), self.eps)
        x_dcg = self.actv(x_dcg)
        x_ecg = self.actv(x_ecg)
        x_dcg = self.bn_dec_dcg3(x_dcg)
        x_ecg = self.bn_dec_ecg3(x_ecg)
        x_dcg = torch.nan_to_num(self.dec_dcg4(x_dcg), self.eps)
        x_ecg = torch.nan_to_num(self.dec_ecg4(x_ecg), self.eps)
        x_dcg = self.actv(x_dcg)
        x_ecg = self.actv(x_ecg)
        x_dcg = self.bn_dec_dcg4(x_dcg)
        x_ecg = self.bn_dec_ecg4(x_ecg)
        x_dcg = torch.nan_to_num(self.dec_dcg5(x_dcg), self.eps)
        x_ecg = torch.nan_to_num(self.dec_ecg5(x_ecg), self.eps)

        re_dcg = torch.sigmoid(x_dcg)
        re_ecg = torch.sigmoid(x_ecg)

        return re_dcg, re_ecg

    def forward(self, z):
        re_dcg, re_ecg = self.decode(z)

        return re_dcg, re_ecg

def generate_model(**kwargs):
    model = MLP(**kwargs)

    return model
