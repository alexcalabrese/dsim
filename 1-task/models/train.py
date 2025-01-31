"""
Training pipeline with PyTorch Lightning integration
"""

import pytorch_lightning as pl
from torch import optim
from torchmetrics import Accuracy

class EmotionTrainingSystem(pl.LightningModule):
    """PyTorch Lightning training system for emotion classification
    
    Features:
        - Automatic mixed precision
        - Learning rate scheduling
        - Early stopping
        - Model checkpointing
    """
    
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.lr = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task='multiclass', num_classes=6)
        self.val_acc = Accuracy(task='multiclass', num_classes=6)
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        features = batch['features']
        labels = batch['labels']
        outputs = self(features)
        loss = self.criterion(outputs, labels)
        self.train_acc(outputs, labels)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        features = batch['features']
        labels = batch['labels']
        outputs = self(features)
        loss = self.criterion(outputs, labels)
        self.val_acc(outputs, labels)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_acc',
                'interval': 'epoch',
                'frequency': 1
            }
        } 