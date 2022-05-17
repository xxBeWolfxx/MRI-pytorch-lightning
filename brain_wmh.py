import os
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
from albumentations import Compose, Resize, PadIfNeeded
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from typing import List
from pytorch_lightning.callbacks import LearningRateMonitor
from torch import einsum
from torch.distributions.constraints import simplex, one_hot
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import torchmetrics
from segmentation_models_pytorch import Unet

max_epochs = 1000


class SkullstripperDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, allowed_names: List[str], augment: bool = False):
        self._images_path = path / 'img'
        self._labels_path = path / 'mask'
        self._allowed_names = allowed_names

        self.wantedSize = (96, 160)

        if augment:
            self._transforms = Compose([
                Resize(83, 132),
                PadIfNeeded(*self.wantedSize),
                ToTensorV2()
            ])

    def __getitem__(self, index: int):
        image_path = self._images_path / self._allowed_names[index]
        mask_path = self._labels_path / self._allowed_names[index]

        image = cv2.imread(os.path.join(image_path))
        mask = cv2.imread(os.path.join(mask_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        result = self._transforms(image=image, mask=mask)
        return result['image'], result['mask'].type(torch.float32) / 255.0

    def __len__(self):
        return len(self._allowed_names)


base_path = Path('nii_data')
train_file_names = sorted([path.name for path in (base_path / 'I' / 'mask').iterdir()])
train_file_names, test_file_names = train_test_split(train_file_names, test_size=0.2, random_state=42)
test_file_names, val_file_names = train_test_split(test_file_names, test_size=0.25, random_state=42)

train_dataset = SkullstripperDataset(base_path / 'I', train_file_names, augment=True)
val_dataset = SkullstripperDataset(base_path / 'I', val_file_names, augment=True)
test_dataset = SkullstripperDataset(base_path / 'I', test_file_names, augment=True)


def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum().float()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


class DiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, target):
        preds = torch.sigmoid(preds)
        preds = preds.flatten()
        target = target.flatten()

        intersection = (preds * target).sum()
        return 1 - ((2 * intersection + 1) / (preds.sum() + target.sum() + 1))


class Segmenter(pl.LightningModule):
    global max_epochs
    alphaBL = (1 / max_epochs)

    def __init__(self):
        super().__init__()

        # self.network = Unet('resnet18', encoder_weights='imagenet', activation='sigmoid', in_channels=1)
        # self.network = Unet('efficientnet-b0', encoder_weights='imagenet', activation='sigmoid', in_channels=1)
        self.network = Unet('efficientnet-b3', encoder_weights='imagenet', activation='sigmoid', in_channels=1)

        self.loss_function = DiceLoss()

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float().view(-1, 1, 96, 160)
        mask = mask.float().view(-1, 1, 96, 160)
        out = self(img)

        loss = self.loss_function(out, mask)
        self.log('train_loss', loss, prog_bar=True)
        accuracy = torchmetrics.functional.accuracy(out, mask.type(torch.int64))
        precision = torchmetrics.functional.precision(out, mask.type(torch.int64))
        recall = torchmetrics.functional.recall(out, mask.type(torch.int64))
        dice_score = dice_coeff(out, mask.type(torch.int64))
        self.log('train_acc', accuracy, prog_bar=True)
        self.log('train_precision', precision, prog_bar=True)
        self.log('train_recall', recall, prog_bar=True)
        self.log('train_dice', dice_score, prog_bar=True)

        self.alphaBL = (1 / max_epochs) + (1 / max_epochs) * self.current_epoch
        loss = (1 - self.alphaBL) * loss + self.alphaBL * self.bounderLoss(precision, recall)

        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float().view(-1, 1, 96, 160)
        mask = mask.float().view(-1, 1, 96, 160)
        out = self(img)

        loss = self.loss_function(out, mask)

        accuracy = torchmetrics.functional.accuracy(out, mask.type(torch.int64))
        precision = torchmetrics.functional.precision(out, mask.type(torch.int64))
        recall = torchmetrics.functional.recall(out, mask.type(torch.int64))
        dice_score = dice_coeff(out, mask.type(torch.int64))

        self.alphaBL = (1 / max_epochs) + (1 / max_epochs) * self.current_epoch
        loss = (1 - self.alphaBL) * loss + self.alphaBL * self.bounderLoss(precision, recall)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', accuracy, prog_bar=True)
        self.log('val_precision', precision, prog_bar=True)
        self.log('val_recall', recall, prog_bar=True)
        self.log('val_dice', dice_score, prog_bar=True)

    def test_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float().view(-1, 1, 96, 160)
        mask = mask.float().view(-1, 1, 96, 160)
        out = self(img)

        loss = self.loss_function(out, mask)

        accuracy = torchmetrics.functional.accuracy(out, mask.type(torch.int64))
        precision = torchmetrics.functional.precision(out, mask.type(torch.int64))
        recall = torchmetrics.functional.recall(out, mask.type(torch.int64))
        dice_score = dice_coeff(out, mask.type(torch.int64))

        self.alphaBL = (1 / max_epochs) + (1 / max_epochs) * self.current_epoch
        loss = (1 - self.alphaBL) * loss + self.alphaBL * self.bounderLoss(precision, recall)

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', accuracy, prog_bar=True)
        self.log('test_precision', precision, prog_bar=True)
        self.log('test_recall', recall, prog_bar=True)
        self.log('test_dice', dice_score, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        sch = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=500, gamma=0.1)
        return [opt], [sch]

        # return torch.optim.Adam(self.parameters(), lr=1e-4)

    def bounderLoss(self, precision, recall):
        bounder = 2 * precision * recall / (precision + recall + 1e-7)
        return bounder


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8)

segmenter = Segmenter()

model_checkpoint = pl.callbacks.ModelCheckpoint(dirpath='checkpoints/')
# early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss', patience=3)

logger = pl.loggers.NeptuneLogger(
    api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlMmM5ZDE1Mi1jYTBmLTQzNjMtYWZiNy0zYmI1MjI1OGExMmUifQ==',
    project='aniakettner/WMH'
)

trainer = pl.Trainer(
    logger=logger,
    callbacks=[model_checkpoint],
    gpus=1,

    max_epochs=max_epochs,
    resume_from_checkpoint="checkpoints/epoch=999-step=511999.ckpt"
)

trainer.fit(segmenter, train_dataloaders=train_loader, val_dataloaders=val_loader)
# trainer.test(ckpt_path='checkpoints/epoch=999-step=511999.ckpt', test_dataloaders=test_loader)


with torch.no_grad():
    image, mask = test_dataset[0]
    image = image.float()
    result = segmenter(image[None, ...])
    result[result < 0.5] = 0
    result[result != 0] = 1

    plt.imshow(image.permute(1, 2, 0), cmap='gray')
    # plt.imsave('brain1.png', image.squeeze(), cmap='gray')
    plt.figure()
    plt.imshow(result.squeeze(), cmap='gray')
    # plt.imsave('WMH_epochs1000_efficientnet.png', result.squeeze(), cmap='gray')
    plt.figure()
    plt.imshow(mask.squeeze(), cmap='gray')
    # plt.imsave('mask1.png', mask.squeeze(), cmap='gray')

    plt.show()
