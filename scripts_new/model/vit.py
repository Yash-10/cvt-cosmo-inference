import numpy as np
import pandas as pd
import gzip
from torch.utils.data.dataset import Dataset
import torch
import torch.nn as nn

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from vit_pytorch import SimpleViT
# from vit_pytorch.cross_vit import CrossViT
# from vit_pytorch.vit_for_small_dataset import ViT as ViTSmallDataset
import torch.optim as optim

from utils import get_rmse_score
from model.cnn import model_o3_err

from einops.layers.torch import Rearrange


params          = [0,1,2,3,4]    #Omega_m, Omega_b, h, n_s, sigma_8. The code will be trained to predict all these parameters.
g               = params           #g will contain the mean of the posterior
h               = [5+i for i in g] #h will contain the variance of the posterior

class ViT(pl.LightningModule):
    def __init__(self, model_kwargs, lr, wd, beta1, beta2, minimum, maximum):
        super().__init__()
        self.save_hyperparameters()
        #         self.model = VisionTransformer(**model_kwargs)  # If want to use the original ViT.
        self.model = SimpleViT(  # If want to use an improved version of the ViT, also proposed by the original authors.
            image_size = model_kwargs['image_size'],
            patch_size = model_kwargs['patch_size'],
            num_classes = model_kwargs['num_classes'],
            dim = model_kwargs['hidden_dim'],
            depth = model_kwargs['num_layers'],
            heads = model_kwargs['num_heads'],
            mlp_dim = model_kwargs['embed_dim'],
            channels=1
        )
        #         self.model = CrossViT(
        #             image_size = model_kwargs['image_size'],
        #             num_classes = model_kwargs['num_classes'],
        #             depth = model_kwargs['depth'],             # number of multi-scale encoding blocks
        #             sm_dim = model_kwargs['sm_dim'],            # high res dimension
        #             sm_patch_size = model_kwargs['sm_patch_size'],      # high res patch size (should be smaller than lg_patch_size)
        #             sm_enc_depth = model_kwargs['sm_enc_depth'],        # high res depth
        #             sm_enc_heads = model_kwargs['sm_enc_heads'],        # high res heads
        #             sm_enc_mlp_dim = model_kwargs['sm_enc_mlp_dim'],   # high res feedforward dimension
        #             lg_dim = model_kwargs['lg_dim'],            # low res dimension
        #             lg_patch_size = model_kwargs['lg_patch_size'],      # low res patch size
        #             lg_enc_depth = model_kwargs['lg_enc_depth'],        # low res depth
        #             lg_enc_heads = model_kwargs['lg_enc_heads'],        # low res heads
        #             lg_enc_mlp_dim = model_kwargs['lg_enc_mlp_dim'],   # low res feedforward dimensions
        #             cross_attn_depth = model_kwargs['cross_attn_depth'],    # cross attention rounds
        #             cross_attn_heads = model_kwargs['cross_attn_heads'],    # cross attention heads
        #             dropout = model_kwargs['dropout'],
        #             emb_dropout = model_kwargs['emb_dropout'],
        #             channels = model_kwargs['num_channels']
        #         )
        #         self.model = ViTSmallDataset(
        #             image_size = model_kwargs['image_size'],
        #             patch_size = model_kwargs['patch_size'],
        #             num_classes = model_kwargs['num_classes'],
        #             dim = model_kwargs['dim'],
        #             depth = model_kwargs['depth'],
        #             heads = model_kwargs['heads'],
        #             mlp_dim = model_kwargs['mlp_dim'],
        #             dropout = model_kwargs['dropout'],
        #             emb_dropout = model_kwargs['emb_dropout'],
        #             channels = model_kwargs['channels']
        #         )
        #self.example_input_array = next(iter(train_loader))[0]

        self.maximum = maximum
        self.minimum = minimum

    def forward(self, x):
        # NOTE: See https://lightning.ai/docs/pytorch/2.1.3/starter/style_guide.html#forward-vs-training-step
        # forward is recommended to be used for prediction/inference, whereas for actual training, training_step is recommended.
#         return self.model(x)

        out = self.model(x)
        # enforce the errors to be positive
        y = torch.clone(out)
        y[:,5:10] = torch.square(out[:,5:10])
        return y

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

        out = self.model(x)
        # enforce the errors to be positive
        p = torch.clone(out)
        p[:,5:10] = torch.square(out[:,5:10])

        y_NN = p[:,g]             #posterior mean
        e_NN = p[:,h]             #posterior std
        #y_NN = p[:,[0,4]]
        #e_NN = p[:,[5,9]]
        
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
            sigma_bar = np.mean(e_NN, axis=0)
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


def load_model_for_torch_summary(model_name, model_kwargs):
    #     model_torch_summary = ViT(**kwargs)
    #     return model_torch_summary
    if model_name == "ViT":
        model_torch_summary = SimpleViT(  # If want to use an improved version of the ViT, also proposed by the original authors.
            image_size = model_kwargs['image_size'],
            patch_size = model_kwargs['patch_size'],
            num_classes = model_kwargs['num_classes'],
            dim = model_kwargs['hidden_dim'],
            depth = model_kwargs['num_layers'],
            heads = model_kwargs['num_heads'],
            mlp_dim = model_kwargs['embed_dim'],
            channels=1
        )
    elif model_name == "CNN":
        model_torch_summary = model_o3_err(**model_kwargs)
    else:
        raise ValueError(f"Model name {model_name} not recognized.")
#     model_torch_summary = CrossViT(
#             image_size = model_kwargs['image_size'],
#             num_classes = model_kwargs['num_classes'],
#             depth = model_kwargs['depth'],             # number of multi-scale encoding blocks
#             sm_dim = model_kwargs['sm_dim'],            # high res dimension
#             sm_patch_size = model_kwargs['sm_patch_size'],      # high res patch size (should be smaller than lg_patch_size)
#             sm_enc_depth = model_kwargs['sm_enc_depth'],        # high res depth
#             sm_enc_heads = model_kwargs['sm_enc_heads'],        # high res heads
#             sm_enc_mlp_dim = model_kwargs['sm_enc_mlp_dim'],   # high res feedforward dimension
#             lg_dim = model_kwargs['lg_dim'],            # low res dimension
#             lg_patch_size = model_kwargs['lg_patch_size'],      # low res patch size
#             lg_enc_depth = model_kwargs['lg_enc_depth'],        # low res depth
#             lg_enc_heads = model_kwargs['lg_enc_heads'],        # low res heads
#             lg_enc_mlp_dim = model_kwargs['lg_enc_mlp_dim'],   # low res feedforward dimensions
#             cross_attn_depth = model_kwargs['cross_attn_depth'],    # cross attention rounds
#             cross_attn_heads = model_kwargs['cross_attn_heads'],    # cross attention heads
#             dropout = model_kwargs['dropout'],
#             emb_dropout = model_kwargs['emb_dropout'],
#             channels = model_kwargs['num_channels']
#         )
#     model_torch_summary = ViTSmallDataset(
#         image_size = model_kwargs['image_size'],
#         patch_size = model_kwargs['patch_size'],
#         num_classes = model_kwargs['num_classes'],
#         dim = model_kwargs['dim'],
#         depth = model_kwargs['depth'],
#         heads = model_kwargs['heads'],
#         mlp_dim = model_kwargs['mlp_dim'],
#         dropout = model_kwargs['dropout'],
#         emb_dropout = model_kwargs['emb_dropout'],
#         channels = model_kwargs['channels']
#     )
    return model_torch_summary
    

class ViT_FineTune(ViT):
    def __init__(self, PRETRAINED_FILENAME, model_kwargs, lr, wd, beta1, beta2, minimum, maximum, freeze_layers=False):
        super().__init__(model_kwargs=model_kwargs, lr=lr, wd=wd, beta1=beta1, beta2=beta2, minimum=minimum, maximum=maximum)
        self.save_hyperparameters()

        self.model = ViT.load_from_checkpoint(
            PRETRAINED_FILENAME, model_kwargs, lr=lr, wd=wd, beta1=beta1, beta2=beta2, minimum=minimum, maximum=maximum
        )

        if freeze_layers:
            # Freeze the parameters
            for param in self.model.model.parameters():
                param.requires_grad = False

            # Assign new Linear layer to the patch embedding layer
            patch_dim = model_kwargs['patch_size'] ** 2
            dim = model_kwargs['hidden_dim']
            self.model.model.to_patch_embedding[2] = nn.Linear(patch_dim, dim)

        # Even if freeze_layers=True, we re-initialize the MLP head.
        # See, for example, https://pyimagesearch.com/2019/06/03/fine-tuning-with-keras-and-deep-learning/
        # NOTE: This reinitialization must come after `freeze_layers`, otherwise the reinitialized layer will again become frozen.
        self.model.model.mlp_head = nn.Sequential(
            nn.LayerNorm(model_kwargs['embed_dim']),
            nn.Linear(model_kwargs['embed_dim'], model_kwargs['num_classes'])
        )
        
#         self.model.model.mlp_head = nn.Sequential(
#             nn.LayerNorm(model_kwargs['embed_dim']),
#             nn.Linear(model_kwargs['embed_dim'], model_kwargs['embed_dim']//2),
#             nn.GELU(),
#             nn.Dropout(p=0.1),
#             nn.Linear(model_kwargs['embed_dim']//2, model_kwargs['embed_dim']//4),
#             nn.GELU(),
#             nn.Dropout(p=0.1),
#             nn.Linear(model_kwargs['embed_dim']//4, model_kwargs['num_classes'])
#         )

        #self.example_input_array = next(iter(train_loader))[0]
 
    def configure_optimizers(self):
        mlp_head_params = list(map(lambda x: x[1],list(filter(lambda kv: 'model.mlp_head' in kv[0], self.model.named_parameters()))))
        feature_params = list(map(lambda x: x[1],list(filter(lambda kv: 'model.mlp_head' not in kv[0], self.model.named_parameters()))))
        assert len(mlp_head_params) > 0  # Because we know there exists a MLP head in our model.
        assert len(feature_params) > 0  # Because we know there exists parameters corresponding to the transformer and input layers.
        optimizer = torch.optim.AdamW(
            [
                {'params': mlp_head_params, 'lr': 1e-2},
                {'params': feature_params, 'lr': 1e-4}
            ],
            weight_decay=self.hparams.wd, betas=(self.hparams.beta1, self.hparams.beta2)
        )
#         optimizer = torch.optim.AdamW(
#             [
#                 # Set a small learning rate for all pre-trained layers, but a larger
#                 # learning rate for the MLP head layers.
#                 {"params": self.model.model.input_layer.parameters(), "lr": 1e-4},
#                 {"params": self.model.model.transformer.parameters(), "lr": 1e-4},
#                 {"params": self.model.model.mlp_head.parameters(), "lr": 1e-2},
#             ],
#             weight_decay=self.hparams.wd, betas=(self.hparams.beta1, self.hparams.beta2)
#         )
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss"
            },
        }


class ViT_FineTune_CvT(ViT):
    def __init__(self, PRETRAINED_FILENAME, model_kwargs, lr, wd, beta1, beta2, minimum, maximum, freeze_layers=False):
        super().__init__(model_kwargs=model_kwargs, lr=lr, wd=wd, beta1=beta1, beta2=beta2, minimum=minimum, maximum=maximum)
        self.save_hyperparameters()

        self.model = ViT.load_from_checkpoint(
            PRETRAINED_FILENAME, model_kwargs, lr=lr, wd=wd, beta1=beta1, beta2=beta2, minimum=minimum, maximum=maximum
        )

        if freeze_layers:
            # Freeze the parameters
            for param in self.model.model.parameters():
                param.requires_grad = False

        # Even if freeze_layers=True, we re-initialize the `to_logits` head (all other layers are kept frozen if freeze_layers=True).
        # See, for example, https://pyimagesearch.com/2019/06/03/fine-tuning-with-keras-and-deep-learning/
        # NOTE: This reinitialization must come after `freeze_layers`, otherwise the reinitialized layer will again become frozen.
        s3_emb_dim = self.model.model.layers[-1][0].out_channels  # hacky way to get the value of s3_emb_dim.
        self.model.model.to_logits = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...'),
            nn.Linear(s3_emb_dim, model_kwargs['num_classes'])
        )
 
    def configure_optimizers(self):
        to_logits_params = list(map(lambda x: x[1],list(filter(lambda kv: 'model.to_logits' in kv[0], self.model.named_parameters()))))
        feature_params = list(map(lambda x: x[1],list(filter(lambda kv: 'model.to_logits' not in kv[0], self.model.named_parameters()))))
        assert len(to_logits_params) > 0  # Because we know there exists a MLP head in our model.
        assert len(feature_params) > 0  # Because we know there exists parameters corresponding to the transformer and input layers.
        optimizer = torch.optim.AdamW(
            [
                {'params': to_logits_params, 'lr': 1e-2},
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
