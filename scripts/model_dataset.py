import numpy as np
import pandas as pd
import gzip
from torch.utils.data.dataset import Dataset
import torch
import torch.nn as nn

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

class CustomImageDataset(Dataset):
        def __init__(self, img_folder_path, normalized_cosmo_params_path, transform=None):
                self.normalized_cosmo_params_path = normalized_cosmo_params_path
                self.normalized_cosmo_params = pd.read_csv(self.normalized_cosmo_params_path)
                self.img_folder_path = img_folder_path
                self.transform = transform

        def __len__(self):
                return len(self.normalized_cosmo_params)

        def __getitem__(self, idx):
                img_path = self.normalized_cosmo_params.iloc[idx, 1]
                f = gzip.GzipFile(img_path, 'r')
                image = np.load(f)
                label = np.array(self.normalized_cosmo_params.iloc[idx, -5:], dtype=np.float32)
                if self.transform:
                image = self.transform(image)
                image = np.expand_dims(image, 0)
                return image, label, img_path


class model_o3_err(nn.Module):
    def __init__(self, hidden, dr, channels):
        super(model_o3_err, self).__init__()

        # input: 1x64x64 ---------------> output: 2*hiddenx32x32  # These dimensions are written assuming 64^3 density field.
        self.C01 = nn.Conv2d(channels,  2*hidden, kernel_size=3, stride=2, padding=1,
                            padding_mode='circular', bias=True)
#         self.C02 = nn.Conv2d(2*hidden,  2*hidden, kernel_size=3, stride=1, padding=1,
#                             padding_mode='circular', bias=True)
#         self.C03 = nn.Conv2d(2*hidden,  2*hidden, kernel_size=2, stride=2, padding=0,
#                             padding_mode='circular', bias=True)
        self.B01 = nn.BatchNorm2d(2*hidden)
#         self.B02 = nn.BatchNorm2d(2*hidden)
#         self.B03 = nn.BatchNorm2d(2*hidden)

        # input: 2*hiddenx32x32 ----------> output: 4*hiddenx16x16
        self.C11 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=3, stride=2, padding=1,
                            padding_mode='circular', bias=True)
#         self.C12 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
#                             padding_mode='circular', bias=True)
#         self.C13 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=2, stride=2, padding=0,
#                             padding_mode='circular', bias=True)
        self.B11 = nn.BatchNorm2d(4*hidden)
#         self.B12 = nn.BatchNorm2d(4*hidden)
#         self.B13 = nn.BatchNorm2d(4*hidden)

        # input: 4*hiddenx16x16 --------> output: 8*hiddenx8x8
        self.C21 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=3, stride=2, padding=1,
                            padding_mode='circular', bias=True)
#         self.C22 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
#                             padding_mode='circular', bias=True)
#         self.C23 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=2, stride=2, padding=0,
#                             padding_mode='circular', bias=True)
        self.B21 = nn.BatchNorm2d(8*hidden)
#         self.B22 = nn.BatchNorm2d(8*hidden)
#         self.B23 = nn.BatchNorm2d(8*hidden)

        # input: 8*hiddenx8x8 ----------> output: 16*hiddenx4x4
        self.C31 = nn.Conv2d(8*hidden,  16*hidden, kernel_size=3, stride=2, padding=1,
                            padding_mode='circular', bias=True)
#         self.C32 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=3, stride=1, padding=1,
#                             padding_mode='circular', bias=True)
#         self.C33 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=2, stride=2, padding=0,
#                             padding_mode='circular', bias=True)
        self.B31 = nn.BatchNorm2d(16*hidden)
#         self.B32 = nn.BatchNorm2d(16*hidden)
#         self.B33 = nn.BatchNorm2d(16*hidden)

        # input: 16*hiddenx4x4 ----------> output: 32*hiddenx1x1
        self.C41 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=4, stride=2, padding=0,
                            padding_mode='circular', bias=True)

        self.B41 = nn.BatchNorm2d(32*hidden)

        self.P0  = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.FC1  = nn.Linear(32*hidden, 16*hidden)
        self.FC2  = nn.Linear(16*hidden, 10)

        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):

        x = self.LeakyReLU(self.C01(image))
#         x = self.LeakyReLU(self.B02(self.C02(x)))
#         x = self.LeakyReLU(self.B03(self.C03(x)))

        x = self.LeakyReLU(self.B11(self.C11(x)))
#         x = self.LeakyReLU(self.B12(self.C12(x)))
#         x = self.LeakyReLU(self.B13(self.C13(x)))

        x = self.LeakyReLU(self.B21(self.C21(x)))
#         x = self.LeakyReLU(self.B22(self.C22(x)))
#         x = self.LeakyReLU(self.B23(self.C23(x)))

        x = self.LeakyReLU(self.B31(self.C31(x)))
#         x = self.LeakyReLU(self.B32(self.C32(x)))
#         x = self.LeakyReLU(self.B33(self.C33(x)))

        x = self.LeakyReLU(self.B41(self.C41(x)))

        x = x.view(image.shape[0], -1)
        x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.FC2(x)

        # enforce the errors to be positive
        y = torch.clone(x)
        y[:,5:10] = torch.square(x[:,5:10])

        return y


############################ Below is for ViT ############################
