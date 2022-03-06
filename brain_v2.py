import os
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
from albumentations import Compose, Resize
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from typing import List
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import torchmetrics
from segmentation_models_pytorch import Unet


class SkullstripperDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, allowed_names: List[str], augment: bool = False):
        self._images_path = path / 'img/subdir_required_by_keras'
        self._labels_path = path / 'mask/subdir_required_by_keras'
        self._allowed_names = allowed_names

        if augment:
            self._transforms = Compose([
                Resize(192, 256),
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


base_path = Path('skullstripper_data')
train_file_names = sorted([path.name for path in (base_path / 'z_train' / 'mask/subdir_required_by_keras').iterdir()])
val_file_names = sorted(
    [path.name for path in (base_path / 'z_validation' / 'mask/subdir_required_by_keras').iterdir()])

train_file_names, test_file_names = train_test_split(train_file_names, test_size=0.15, random_state=42)

train_dataset = SkullstripperDataset(base_path / 'z_train', train_file_names, augment=True)
val_dataset = SkullstripperDataset(base_path / 'z_validation', val_file_names, augment=True)
test_dataset = SkullstripperDataset(base_path / 'z_train', test_file_names, augment=True)


def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum().float()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


class Segmenter(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # self.network = Unet('resnet18', encoder_weights='imagenet', activation='sigmoid', in_channels=1)
        # self.network = Unet('efficientnet-b0', encoder_weights='imagenet', activation='sigmoid', in_channels=1)
        self.network = Unet('efficientnet-b3', encoder_weights='imagenet', activation='sigmoid', in_channels=1)
        self.loss_function = F.binary_cross_entropy_with_logits

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float().view(-1, 1, 192, 256)
        mask = mask.float().view(-1, 1, 192, 256)
        out = self(img)

        loss = self.loss_function(out, mask)
        accuracy = torchmetrics.functional.accuracy(out, mask.type(torch.int64))
        precision = torchmetrics.functional.precision(out, mask.type(torch.int64))
        recall = torchmetrics.functional.recall(out, mask.type(torch.int64))
        dice_score = dice_coeff(out, mask.type(torch.int64))
        self.log('train_acc', accuracy, prog_bar=True)
        self.log('train_precision', precision, prog_bar=True)
        self.log('train_recall', recall, prog_bar=True)
        self.log('train_dice', dice_score, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float().view(-1, 1, 192, 256)
        mask = mask.float().view(-1, 1, 192, 256)
        out = self(img)

        loss = self.loss_function(out, mask)
        self.log('val_loss', loss, prog_bar=True)
        accuracy = torchmetrics.functional.accuracy(out, mask.type(torch.int64))
        precision = torchmetrics.functional.precision(out, mask.type(torch.int64))
        recall = torchmetrics.functional.recall(out, mask.type(torch.int64))
        dice_score = dice_coeff(out, mask.type(torch.int64))
        self.log('val_acc', accuracy, prog_bar=True)
        self.log('val_precision', precision, prog_bar=True)
        self.log('val_recall', recall, prog_bar=True)
        self.log('val_dice', dice_score, prog_bar=True)


    def test_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float().view(-1, 1, 192, 256)
        mask = mask.float().view(-1, 1, 192, 256)
        out = self(img)

        loss = self.loss_function(out, mask)
        self.log('test_loss', loss, prog_bar=True)
        accuracy = torchmetrics.functional.accuracy(out, mask.type(torch.int64))
        precision = torchmetrics.functional.precision(out, mask.type(torch.int64))
        recall = torchmetrics.functional.recall(out, mask.type(torch.int64))
        dice_score = dice_coeff(out, mask.type(torch.int64))
        self.log('test_acc', accuracy, prog_bar=True)
        self.log('test_precision', precision, prog_bar=True)
        self.log('test_recall', recall, prog_bar=True)
        self.log('test_dice', dice_score, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        sch = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=500, gamma=0.1)
        return [opt], [sch]

        # return torch.optim.Adam(self.parameters(), lr=1e-4)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8)

segmenter = Segmenter()

model_checkpoint = pl.callbacks.ModelCheckpoint(dirpath='checkpoints/')
# early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss', patience=3)

logger = pl.loggers.NeptuneLogger(
    api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlMmM5ZDE1Mi1jYTBmLTQzNjMtYWZiNy0zYmI1MjI1OGExMmUifQ==',
    project='aniakettner/brain'
)

trainer = pl.Trainer(
    logger=logger,
    callbacks=[model_checkpoint],
    gpus=1,
    max_epochs=10,
    # resume_from_checkpoint="checkpoints/epoch=4-step=11084.ckpt"
)

trainer.fit(segmenter, train_dataloaders=train_loader, val_dataloaders=val_loader)
# trainer.test(ckpt_path='checkpoints/epoch=4-step=11084.ckpt', test_dataloaders=test_loader)

with torch.no_grad():
    image, mask = test_dataset[0]
    image = image.float()
    result = segmenter(image[None, ...])
    result[result < 0.5] = 0
    result[result != 0] = 1

    plt.imshow(image.permute(1, 2, 0))
    plt.figure()
    plt.imshow(result.squeeze())
    plt.figure()
    plt.imshow(mask.squeeze())
    plt.show()
