import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import torch.optim as optim
from utils import get_rmse_score

from einops.layers.torch import Rearrange

params          = [0,1,2,3,4]    #Omega_m, Omega_b, h, n_s, sigma_8. The code will be trained to predict all these parameters.
g               = params           #g will contain the mean of the posterior
h               = [5+i for i in g] #h will contain the variance of the posterior


class model_o3_err(nn.Module): # TODO: This is not used in the notebooks currently.
    def __init__(self, image_size, hidden, dr, channels):
        super(model_o3_err, self).__init__()

        img_size = image_size
        
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

        img_size = int((img_size+2*1-1*(3-1)-1)/2 +1)
        
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

        img_size = int((img_size+2*1-1*(3-1)-1)/2 +1)
        
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
        
        img_size = int((img_size+2*1-1*(3-1)-1)/2 +1)

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
        
        img_size = int((img_size+2*1-1*(3-1)-1)/2 +1)

        # input: 16*hiddenx4x4 ----------> output: 32*hiddenx1x1
        self.C41 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=4, stride=2, padding=0,
                            padding_mode='circular', bias=True)

        self.B41 = nn.BatchNorm2d(32*hidden)

        img_size = int((img_size+2*0-1*(4-1)-1)/2 +1)

#         self.FC1  = nn.Linear(32*hidden, 16*hidden)
#         self.FC2  = nn.Linear(16*hidden, 10)

        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        #self.mlp_head = nn.Sequential(
        #        nn.Linear(32*hidden*img_size*img_size, 16*hidden),
        #        self.LeakyReLU,
        #        self.dropout,
        #        nn.Linear(16*hidden, 10)
        #)
        self.mlp_head = nn.Sequential(
                nn.Linear(32*hidden*img_size*img_size, 10),
                # self.LeakyReLU,
                # self.dropout,
                # nn.Linear(16*hidden, 10)
        )

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
        
        # The MLP head implements the two commented lines below.
        x = self.mlp_head(x)
#         x = self.dropout(self.LeakyReLU(self.FC1(x)))
#         x = self.FC2(x)

        # enforce the errors to be positive
        y = torch.clone(x)
        y[:,5:10] = torch.square(x[:,5:10])

        return y


class CNN(pl.LightningModule):

    def __init__(self, model_kwargs, lr, wd, beta1, beta2, minimum, maximum):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_o3_err(**model_kwargs)
        #self.example_input_array = next(iter(train_loader))[0]

        self.maximum = maximum
        self.minimum = minimum

    def forward(self, x):
        # NOTE: See https://lightning.ai/docs/pytorch/2.1.3/starter/style_guide.html#forward-vs-training-step
        # forward is recommended to be used for prediction/inference, whereas for actual training, training_step is recommended.
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd, betas=(self.hparams.beta1, self.hparams.beta2))
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=5)
#         return [optimizer], [lr_scheduler]

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss"
            },
        }

    def _calculate_loss(self, batch, mode="train"):
        x, y, _ = batch
        batch_size = x.shape[0]

        p = self.model(x)
        y_NN = p[:,g]             #posterior mean
        e_NN = p[:,h]             #posterior std

        loss1 = torch.mean((y_NN - y)**2,                axis=0)
        loss2 = torch.mean(((y_NN - y)**2 - e_NN**2)**2, axis=0)
        loss  = torch.mean(torch.log(loss1) + torch.log(loss2))
        # NOTE: See logging for more information: https://lightning.ai/docs/pytorch/2.1.3/extensions/logging.html
        # Not sure if the below logic is even needed, but should be fine.
        if mode == "train" or mode == 'val':
            # To match the CNN training code, we need to log the train and val
            # loss after each batch, and also after each epoch.
            self.log(f'{mode}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        elif mode == 'test':
            # For testing, logging the loss after each step is not required.
            # So we only log after the epoch. For testing, there will be one epoch only.
            self.log(f'{mode}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

        # TODO: Where are the logs/logged values stored? Need to find and print manually at end of training outside this function.

        if mode == 'val' or mode == 'train':
            # Untransform the parameters for the sake of calculating RMSE and sigma_bar.
            # `minimum` and `maximum` must be defined globally.
            y = y.cpu().detach().numpy() * (self.maximum - self.minimum) + self.minimum
            y_NN   = y_NN.cpu().detach().numpy() * (self.maximum - self.minimum) + self.minimum
            e_NN   = e_NN.cpu().detach().numpy() * (self.maximum - self.minimum)

            # Also log RMSE and sigma_bar for all parameters.
            rmse = get_rmse_score(y, y_NN)
            sigma_bar = np.mean(y_NN, axis=0)
            # Only log at the end of epoch instead of each step.
            # Logging is only done for Omega_m and sigma_8 since only these are interesting for DM density/DM halo fields.
            # But more can easily be added here if and when needed.
            metrics_to_log = {
                f'{mode}_omegam_rmse': rmse[0],
                f'{mode}_omegam_sigma_bar': sigma_bar[0],
                f'{mode}_sigma8_rmse': rmse[-1],
                f'{mode}_sigma8_sigma_bar': sigma_bar[-1]
            }
            self.log_dict(metrics_to_log, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")



class CNN_FineTune(CNN):
    def __init__(self, PRETRAINED_FILENAME, model_kwargs, lr, wd, beta1, beta2, minimum, maximum):
        super(CNN_FineTune, self).__init__(model_kwargs=model_kwargs, lr=lr, wd=wd, beta1=beta1, 
        beta2=beta2, minimum=minimum, maximum=maximum)
        
        self.save_hyperparameters()

        self.model = CNN.load_from_checkpoint(PRETRAINED_FILENAME, model_kwargs, lr=lr, wd=wd, beta1=beta1, beta2=beta2, minimum=minimum, maximum=maximum)

        # Re-initialize the MLP head.
        # See, for example, https://pyimagesearch.com/2019/06/03/fine-tuning-with-keras-and-deep-learning/
        self.model.model.mlp_head = nn.Sequential(
            nn.Linear(32*model_kwargs["hidden"], 10),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(p=model_kwargs['dr']),
#             nn.Linear(16*hidden, 10)
        )

        #self.example_input_array = next(iter(train_loader))[0]

    def configure_optimizers(self):
        # The below way of setting the learning rates taken from https://discuss.pytorch.org/t/how-to-set-a-different-learning-rate-for-a-single-layer-in-a-network/48552/10
        mlp_head_params = list(map(lambda x: x[1],list(filter(lambda kv: 'model.mlp_head' in kv[0], self.model.named_parameters()))))
        feature_params = list(map(lambda x: x[1],list(filter(lambda kv: 'model.mlp_head' not in kv[0], self.model.named_parameters()))))
        assert len(mlp_head_params) > 0  # Because we know there exists a MLP head in our model.
        assert len(feature_params) > 0  # Because we know there exists parameters corresponding to the convolution layers.
        optimizer = torch.optim.AdamW(
            [
                {'params': mlp_head_params, 'lr': 1e-2},
                {'params': feature_params, 'lr': 1e-4}
            ],
            weight_decay=self.hparams.wd, betas=(self.hparams.beta1, self.hparams.beta2)
        )
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss"
            },
        }