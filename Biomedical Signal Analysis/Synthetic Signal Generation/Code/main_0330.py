from __future__ import print_function

import os
import csv
import math
import datetime
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks

import torch
import torch.utils.data
from torch.nn import functional as F
from torch.utils.data import DataLoader

from model import mlp_enc, cnn_enc, lstm_enc, mlp_dec, cnn_dec, lstm_dec

model_list = ['MLP', 'CNN', 'LSTM']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"

s_type = ['dcg', 'ecg']
time = 5
s_rate = 5
s_hz = str(int(1000 / s_rate))
process_list = ['norm', 'sim', 'det'] # both, ecg, dcg
cr = 100

dcg_exist = 1
ecg_exist = 1

# dataPath = '../data/dataset_val_start/'
# dataPath = '../data/data_220325/'
dataPath = '../data/data_com/'
data_date = '1012'

if os.path.exists(dataPath):
    pass
else:
    # dataPath = '../../data/dataset_val_start/'
    # dataPath = '../../data/data_220325/'
    dataPath = '../data/data_com/'


class Options:
    def __init__(self):
        self.batch_size = 64
        self.workers = 1
        self.input_size_dcg = int(time * 1000 / s_rate)  # if s_type == 'dcg' else int(200*time)  # resnet default size
        self.input_size_ecg = int(200 * time)
        self.hidden = [int(self.input_size_dcg // 2), int(self.input_size_dcg // 4), int(self.input_size_dcg // 8),
                       int(self.input_size_dcg // 16), int(self.input_size_dcg // 32), int(self.input_size_dcg // 64)]
        self.num_layers = 3
        self.kernel_size = 3
        self.out_size = int(self.input_size_dcg / cr)

        self.win_size = 101
        self.poly_order = 5

        self.lr = 1e-3
        self.epoch = 100  # train loss vs validation loss compare
        self.model = model_list[0]
        self.dcg_norm = process_list[2]
        self.ecg_norm = process_list[1]

class signalDataset:

    def __init__(self, dataset_name, dcg_norm=False, ecg_norm=False):
        self.dataset_name = dataset_name
        t_data = pd.read_csv(dataset_name, header=None)  # data 도중 length -1 문제 발생 : header = 0 확인!
        data = torch.from_numpy(np.expand_dims(np.array([t_data.iloc[i] for i in range(0, len(t_data))]), -1)).float()

        if dcg_norm == 'norm':
            self.data = self.normalize(data)
        elif dcg_norm == 'det' :
            self.data = self.dcg_peak(data)
        elif ecg_norm == 'norm':
            self.data = self.normalize(data)
        elif ecg_norm == 'sim':
            self.data = self.ecg_peak_simplify(data)
        elif ecg_norm == 'det':
            self.data = self.ecg_peak_detrend(data)
        else:
            self.data = data

        original_deltas = data[:, -1] - data[:, 0]
        self.original_deltas = original_deltas
        self.or_delta_max, self.or_delta_min = original_deltas.max(), original_deltas.min()
        deltas = self.data[:, -1] - self.data[:, 0]
        self.deltas = deltas
        self.delta_mean, self.delta_std = deltas.mean(), deltas.std()
        self.delta_max, self.delta_min = deltas.max(), deltas.min()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx]

    def normalize(self, x_list):
        x_norm = torch.empty(0, dtype=torch.float)
        for x in x_list:
            self.max = x.max()
            self.min = x.min()

            x = (x - self.min) / (self.max - self.min)  # 0~1y/
            x = torch.unsqueeze(x, 0)
            x_norm = torch.cat([x_norm, x], dim=0)

        return x_norm

    def ecg_peak_simplify(self, x_list):
        x_norm = torch.empty(0, dtype=torch.float)
        for x in x_list:
            self.max = x.max()
            self.min = x.min()

            x = (x - self.min) / (self.max - self.min)

            x = torch.squeeze(x)
            x = x.tolist()

            # find peak & transform non-peak data
            idx_pks, props = find_peaks(x, height=0.6, distance=100)  # peak distance

            i_p = 1
            ppi = np.zeros(len(idx_pks) - 1, dtype=float)

            for i in range(len(ppi)):
                ppi[i] = idx_pks[i + 1] - idx_pks[i]

            avg_ppi = np.mean(ppi)
            max_ppi = np.max(ppi)

            if avg_ppi < idx_pks[0]:
                avg_ppi = idx_pks[0] + 100

            if max_ppi < (len(x) - idx_pks[-1]):
                max_ppi = len(x) - idx_pks[-1] + 100

            if max_ppi < idx_pks[0]:
                max_ppi = idx_pks[0] + 100

            for i, item in enumerate(x):
                if i < idx_pks[0]:
                    x[i] = (abs(max_ppi + i - idx_pks[0]) / max_ppi) * x[idx_pks[0]]

                elif i > idx_pks[-1]:
                    x[i] = (abs(i - idx_pks[-1]) / max_ppi)  # last peak 이후 signal 처리 (len(x) - idx_pks[-1])

                elif i == idx_pks[i_p]:
                    i_p += 1

                elif i < idx_pks[i_p]:
                    idx_ = idx_pks[i_p]
                    x[i] = abs(i - idx_pks[i_p - 1]) * x[idx_] / (
                            idx_pks[i_p] - idx_pks[i_p - 1])  # peak to peak 사이 값 처리


            x = torch.Tensor(x)
            x = torch.unsqueeze(x, 0)

            x_norm = torch.cat([x_norm, x], dim=0)

        return x_norm

    def ecg_peak_detrend(self, x_list):
        x_norm = torch.empty(0, dtype=torch.float)
        for x in x_list:
            self.max = x.max()
            self.min = x.min()

            x = (x - self.min) / (self.max - self.min)

            x = torch.squeeze(x)
            x = x.tolist()

            # find peak & transform non-peak data
            idx_pks, props = find_peaks(x, height=0.8, distance=100)  # peak distance

            i_p = 1
            ppi = np.zeros(len(idx_pks) - 1, dtype=float)

            for i in range(len(ppi)):
                ppi[i] = idx_pks[i + 1] - idx_pks[i]

            avg_ppi = np.mean(ppi)
            if avg_ppi < idx_pks[0]:
                avg_ppi = idx_pks[0] + 100

            for i, item in enumerate(x):
                if i < idx_pks[0]:
                    x[i] = (abs(avg_ppi + i - idx_pks[0]) / avg_ppi)

                elif i > idx_pks[-1]:
                    x[i] = (abs(i - idx_pks[-1]) / avg_ppi)  # last peak 이후 signal 처리 (len(x) - idx_pks[-1])

                elif i == idx_pks[i_p]:
                    i_p += 1
                    x[i] = 1

                elif i < idx_pks[i_p]:
                    x[i] = abs(i - idx_pks[i_p - 1]) / (idx_pks[i_p] - idx_pks[i_p - 1])  # peak to peak 사이 값 처리

            x = torch.Tensor(x)
            x = torch.unsqueeze(x, 0)

            x_norm = torch.cat([x_norm, x], dim=0)

        return x_norm

    def dcg_peak(self, x_list):
        x_norm = torch.empty(0, dtype=torch.float)
        for x in x_list:
            x = torch.squeeze(x)
            x = x.tolist()

            x_h = signal.hilbert(x)  # hilbert transform
            env = np.abs(x_h)
            x = x / env

            self.max = x.max()
            self.min = x.min()

            x = (x - self.min) / (self.max - self.min)

            x = torch.Tensor(x)
            x = torch.unsqueeze(x, 0)

            x_norm = torch.cat([x_norm, x], dim=0)

        return x_norm

    def denormalize(self, x):
        if not hasattr(self, 'max') or not hasattr(self, 'min'):
            raise Exception("You are calling denormalize, but the input was not normalized")
        return 0.5 * (x * self.max - x * self.min + self.max + self.min)

    def sample_deltas(self, number):
        return (torch.randn(number, 1) + self.delta_mean) * self.delta_std

    def normalize_deltas(self, x):
        return ((self.delta_max - self.delta_min) * (x - self.or_delta_min) / (self.or_delta_max - self.or_delta_min) + self.delta_min)


def generate_model(opt):
    if opt.model == 'MLP':
        enc = mlp_enc.generate_model(device=device,
                                     in_dcg=opt.input_size_dcg,
                                     in_ecg=opt.input_size_ecg,
                                     hidden=opt.hidden,
                                     latent_size=opt.out_size,
                                     num_layers=opt.num_layers)
        dec = mlp_dec.generate_model(device=device,
                                     in_dcg=opt.input_size_dcg,
                                     in_ecg=opt.input_size_ecg,
                                     hidden=opt.hidden,
                                     latent_size=opt.out_size,
                                     num_layers=opt.num_layers)

    elif opt.model == 'CNN':
        enc = cnn_enc.generate_model(device=device,
                                     in_dcg=opt.input_size_dcg,
                                     in_ecg=opt.input_size_ecg,
                                     hidden=opt.hidden,
                                     latent_size=opt.out_size,
                                     kernel_size=opt.kernel_size)
        dec = cnn_dec.generate_model(device=device,
                                     in_dcg=opt.input_size_dcg,
                                     in_ecg=opt.input_size_ecg,
                                     hidden=opt.hidden,
                                     latent_size=opt.out_size,
                                     kernel_size=opt.kernel_size)

    elif opt.model == 'LSTM':
        enc = lstm_enc.generate_model(device=device,
                                      in_dcg=opt.input_size_dcg,
                                      in_ecg=opt.input_size_ecg,
                                      hidden=opt.hidden,
                                      latent_size=opt.out_size,
                                      num_layers=opt.num_layers)
        dec = lstm_dec.generate_model(device=device,
                                      in_dcg=opt.input_size_dcg,
                                      in_ecg=opt.input_size_ecg,
                                      hidden=opt.hidden,
                                      latent_size=opt.out_size,
                                      num_layers=opt.num_layers)

    return enc, dec

def save_checkpoint(model, pth):
    torch.save(model.state_dict(), pth)


def loss_enc(mu, log_var, epoch, i):
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return KLD


def loss_dec(re_ecg, ecg, input_size_ecg):
    BCE_ecg = F.binary_cross_entropy(re_ecg, ecg.view(-1, input_size_ecg), reduction='sum')

    return BCE_ecg


def loss_reg(ecg, re_ecg):
    MSE = F.mse_loss(ecg, re_ecg, reduction='sum')

    return MSE

def Train():
    opt = Options()

    now_time = datetime.datetime.now().strftime('%m%d_%H%M_')

    trainDCG = dataPath + str(time) + '/' + data_date + '_' + s_type[0] + '_' + str(s_rate) + '_' + opt.dcg_norm + '.csv'
    valDCG = dataPath + str(time) + '/' + data_date + '_' + s_type[0] + '_' + str(s_rate) + '_' + opt.dcg_norm + '_val.csv'
    trainECG = dataPath + str(time) + '/' + data_date + '_' + s_type[1] + '_' + str(s_rate) + '_' + opt.ecg_norm + '.csv'
    valECG = dataPath + str(time) + '/' + data_date + '_' + s_type[1] + '_' + str(s_rate) + '_' + opt.ecg_norm + '_val.csv'

    enc, dec = generate_model(opt)
    enc.to(device)
    dec.to(device)
    enc.train()
    dec.train()

    optim_enc = torch.optim.Adam(enc.parameters(), lr=opt.lr)
    optim_dec = torch.optim.Adam(dec.parameters(), lr=opt.lr)

    if os.path.isfile(trainDCG):
        dcg_exist = 1
        t_data = pd.read_csv(trainDCG, header=None)  # data 도중 length -1 문제 발생 : header = 0 확인!
        train_data_dcg = torch.from_numpy(np.expand_dims(np.array([t_data.iloc[i] for i in range(0, len(t_data))]), -1)).float()
        t_data = pd.read_csv(valDCG, header=None)  # data 도중 length -1 문제 발생 : header = 0 확인!
        val_data_dcg = torch.from_numpy(np.expand_dims(np.array([t_data.iloc[i] for i in range(0, len(t_data))]), -1)).float()
    else:
        dcg_exist = 0
        trainDCG = dataPath + str(time) + '/' + data_date + '_' + s_type[0] + '_' + str(s_rate) + '.csv'
        valDCG = dataPath + str(time) + '/' + data_date + '_' + s_type[0] + '_' + str(s_rate) + '_val.csv'
        train_data_dcg = signalDataset(trainDCG, dcg_norm=opt.dcg_norm)
        val_data_dcg = signalDataset(valDCG, dcg_norm=opt.dcg_norm)

        train_dcg_org = dataPath + str(time) + '/' + data_date + '_' + s_type[0] + '_' + str(s_rate) + '_' + opt.dcg_norm + '.csv'
        val_dcg_org = dataPath + str(time) + '/' + data_date + '_' + s_type[0] + '_' + str(s_rate) + '_' + opt.dcg_norm + '_val.csv'

        wTrainDCG_org = open(train_dcg_org, 'w', newline='')
        wTrainDCGOrg = csv.writer(wTrainDCG_org)
        wValDCG_org = open(val_dcg_org, 'w', newline='')
        wValDCGOrg = csv.writer(wValDCG_org)

    if os.path.isfile(trainECG):
        ecg_exist = 1
        t_data = pd.read_csv(trainECG, header=None)  # data 도중 length -1 문제 발생 : header = 0 확인!
        train_data_ecg = torch.from_numpy(np.expand_dims(np.array([t_data.iloc[i] for i in range(0, len(t_data))]), -1)).float()
        t_data = pd.read_csv(valECG, header=None)  # data 도중 length -1 문제 발생 : header = 0 확인!
        val_data_ecg = torch.from_numpy(np.expand_dims(np.array([t_data.iloc[i] for i in range(0, len(t_data))]), -1)).float()
    else:
        ecg_exist = 0
        trainECG = dataPath + str(time) + '/' + data_date + '_' + s_type[1] + '_' + str(s_rate) + '.csv'
        valECG = dataPath + str(time) + '/' + data_date + '_' + s_type[1] + '_' + str(s_rate) + '_val.csv'
        train_data_ecg = signalDataset(trainECG, ecg_norm=opt.ecg_norm)
        val_data_ecg = signalDataset(valECG, ecg_norm=opt.ecg_norm)

        train_ecg_org = dataPath + str(time) + '/' + data_date + '_' + s_type[1] + '_' + str(s_rate) + '_' + opt.ecg_norm + '.csv'
        val_ecg_org = dataPath + str(time) + '/' + data_date + '_' + s_type[1] + '_' + str(s_rate) + '_' + opt.ecg_norm + '_val.csv'

        wTrainECG_org = open(train_ecg_org, 'w', newline='')
        wTrainECGOrg = csv.writer(wTrainECG_org)
        wValECG_org = open(val_ecg_org, 'w', newline='')
        wValECGOrg = csv.writer(wValECG_org)

    print("Start Training : {}, Model : {}, Sample rate : {}Hz, Epoch : {}, Latent size : {}".format(
        datetime.datetime.now().strftime('%m%d_%H:%M'), opt.model, s_hz, opt.epoch, opt.out_size))

    if len(train_data_dcg) % opt.batch_size != 1:
        train_set_dcg = DataLoader(dataset=train_data_dcg, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)
    else:
        train_set_dcg = DataLoader(dataset=train_data_dcg, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, drop_last=True)
    if len(val_data_dcg) % opt.batch_size != 1:
        val_set_dcg = DataLoader(dataset=val_data_dcg, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)
    else:
        val_set_dcg = DataLoader(dataset=val_data_dcg, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, drop_last=True)
    if len(train_data_ecg) % opt.batch_size != 1:
        train_set_ecg = DataLoader(dataset=train_data_ecg, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)
    else:
        train_set_ecg = DataLoader(dataset=train_data_ecg, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, drop_last=True)
    if len(val_data_ecg) % opt.batch_size != 1:
        val_set_ecg = DataLoader(dataset=val_data_ecg, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)
    else:
        val_set_ecg = DataLoader(dataset=val_data_ecg, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, drop_last=True)

    for epoch in range(opt.epoch):

        train_loss = 0.0

        train_mse = 0.0
        train_l1 = 0.0

        dcg_train = []
        ecg_train = []
        loss_list = []

        if epoch == 0:
            loss_F = './result/' + now_time + str(opt.model) + '_' + str(int(1000 / s_rate)) + 'hz_' + str(time) + 's_' + str(cr) + '_' + 'loss.csv'
            loss_write = open(loss_F, 'w', newline='')
            wLoss = csv.writer(loss_write)

            loss_title = ["Train Loss", "Train MSE", "Train L1", "Val Loss", "Val MSE", "Val L1"]
            wLoss.writerow(loss_title)

        elif (epoch + 1) % 50 == 0:
            train_ecg = './result/' + now_time + str(opt.model) + '_' + s_type[1] + '_' + s_hz + 'hz_' + str(time) + 's_' + str(cr) + '_train_' + str(epoch + 1) + '.csv'
            train_ecg = open(train_ecg, 'w', newline='')
            wTrainECG = csv.writer(train_ecg)
            val_ecg = './result/' + now_time + str(opt.model) + '_' + s_type[1] + '_' + s_hz + 'hz_' + str(time) + 's_' + str(cr) + '_val_' + str(epoch + 1) + '.csv'
            val_ecg = open(val_ecg, 'w', newline='')
            wValECG = csv.writer(val_ecg)

            train_z = './result/' + now_time + str(opt.model) + '_' + s_type[1] + '_' + s_hz + 'hz_' + str(time) + 's_' + str(cr) + '_z_' + str(epoch + 1) + '.csv'
            train_z = open(train_z, 'w', newline='')
            wTrainZ = csv.writer(train_z)
            val_z = './result/' + now_time + str(opt.model) + '_' + s_type[1] + '_' + s_hz + 'hz_' + str(time) + 's_' + str(cr) + '_z_val_' + str(epoch + 1) + '.csv'
            val_z = open(val_z, 'w', newline='')
            wValZ = csv.writer(val_z)

        for (dcg, ecg) in zip(train_set_dcg, train_set_ecg):
            dcg = dcg.to(device)
            ecg = ecg.to(device)

            dim_t = list(ecg.size())
            if len(dim_t) == 3:
                ecg = torch.squeeze(ecg)

            optim_enc.zero_grad()
            optim_dec.zero_grad()

            mu, log_var, z = enc(dcg)
            re_dcg, re_ecg = dec(z)

            loss_e = loss_enc(mu, log_var, opt.epoch, epoch)
            loss_d = loss_dec(re_ecg, ecg, opt.input_size_ecg)
            loss_r = loss_reg(ecg, re_ecg)

            loss = loss_e + loss_d + loss_r

            loss.backward()

            train_loss += loss.item()
            train_mse += F.mse_loss(ecg, re_ecg, reduction='sum').item()
            train_l1 += F.l1_loss(ecg, re_ecg, reduction='sum').item()

            optim_enc.step()
            optim_dec.step()

            if epoch == 0:
                if dcg_exist == 0:
                    for i_dcg in dcg:
                        for d in i_dcg:
                            dcg_train.append(d.item())
                        wTrainDCGOrg.writerow(dcg_train)
                        dcg_train = []

                if ecg_exist == 0:
                    for i_ecg in ecg:
                        for e in i_ecg:
                            ecg_train.append(e.item())
                        wTrainECGOrg.writerow(ecg_train)
                        ecg_train = []

            elif (epoch + 1) % 50 == 0:
                # Savitzky-Golay filter
                for s_idx in range(re_ecg.size(0)):
                    re_ecg[s_idx] = torch.Tensor(
                        signal.savgol_filter(torch.squeeze(torch.Tensor(re_ecg.to('cpu')[s_idx])).tolist(),
                                             opt.win_size, opt.poly_order))

                ecg_list = re_ecg.tolist()
                z_list = z.tolist()
                mu_list = mu.tolist()
                log_var_list = log_var.tolist()

                for e in ecg_list:
                    wTrainECG.writerow(e)

                for (mu, lv, z) in zip(mu_list, log_var_list, z_list):
                    wTrainZ.writerow([mu, lv, z])

            if (epoch + 1) == opt.epoch:
                enc_path = './pth/' + now_time + str(opt.model) + '_' + s_hz + 'hz_' + str(epoch) + '_enc.pth'  # model save path
                dec_path = './pth/' + now_time + str(opt.model) + '_' + s_hz + 'hz_' + str(epoch) + '_dec.pth'  # model save path

        loss_list.append(train_loss / (len(train_set_dcg.dataset) * opt.input_size_ecg))
        loss_list.append(train_mse / (len(train_set_dcg.dataset) * opt.input_size_ecg))
        loss_list.append(train_l1 / (len(train_set_dcg.dataset) * opt.input_size_ecg))

        with torch.no_grad():

            val_loss = 0.0
            val_mse = 0.0
            val_l1 = 0.0

            val_dcg = []
            val_ecg = []

            enc.eval()
            dec.eval()

            for (dcg_v, ecg_v) in zip(val_set_dcg, val_set_ecg):
                dcg_v = dcg_v.to(device)
                ecg_v = ecg_v.to(device)

                dim_t = list(ecg_v.size())
                if len(dim_t) == 3:
                    ecg_v = torch.squeeze(ecg_v)

                mu, log_var, z = enc(dcg_v)
                re_dcg, re_ecg = dec(z)

                loss_e = loss_enc(mu, log_var, opt.epoch, epoch)
                loss_d = loss_dec(re_ecg, ecg_v, opt.input_size_ecg)
                loss_r = loss_reg(ecg_v, re_ecg)

                loss = loss_e + loss_d + loss_r

                val_loss += loss.item()
                val_mse += F.mse_loss(ecg_v, re_ecg, reduction='sum').item()
                val_l1 += F.l1_loss(ecg_v, re_ecg, reduction='sum').item()

                if epoch == 0:
                    if dcg_exist == 0:
                        for i_dcg in dcg_v:
                            for d in i_dcg:
                                val_dcg.append(d.item())
                            wValDCGOrg.writerow(val_dcg)
                            val_dcg = []
                    if ecg_exist == 0:
                        for i_ecg in ecg_v:
                            for e in i_ecg:
                                val_ecg.append(e.item())
                            wValECGOrg.writerow(val_ecg)
                            val_ecg = []

                elif (epoch + 1) % 50 == 0:
                    # Savitzky-Golay filter
                    for s_idx in range(re_ecg.size(0)):
                        re_ecg[s_idx] = torch.Tensor(
                            signal.savgol_filter(torch.squeeze(torch.Tensor(re_ecg.to('cpu')[s_idx])).tolist(),
                                                 opt.win_size, opt.poly_order))

                    ecg_list = re_ecg.tolist()
                    z_list = z.tolist()
                    mu_list = mu.tolist()
                    log_var_list = log_var.tolist()

                    for e in ecg_list:
                            wValECG.writerow(e)

                    for (mu, lv, z) in zip(mu_list, log_var_list, z_list):
                        wValZ.writerow([mu, lv, z])


        print('Epoch: {:5d} | Train loss: {:.4f} | Validation loss: {:.4f}'.format(epoch + 1, train_loss / (len(train_set_dcg.dataset) * opt.input_size_ecg), val_loss / (len(val_set_dcg.dataset) * opt.input_size_ecg)))

        loss_list.append(val_loss / (len(val_set_dcg.dataset) * opt.input_size_ecg))
        loss_list.append(val_mse / (len(val_set_dcg.dataset) * opt.input_size_ecg))
        loss_list.append(val_l1 / (len(val_set_dcg.dataset) * opt.input_size_ecg))

        enc.train()
        dec.train()

        wLoss.writerow(loss_list)

    save_checkpoint(enc, enc_path)
    save_checkpoint(dec, dec_path)


if __name__ == "__main__":
    Train()