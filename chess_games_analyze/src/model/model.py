import os
import pandas as pd
import numpy as np
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_1 = nn.Conv2d(in_channels=18, out_channels=64, kernel_size=3, padding=1, bias=True)
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.bn_1 = nn.BatchNorm2d(64)
        self.bn_2 = nn.BatchNorm2d(128)

        self.dropout = nn.Dropout(0.2)

        self.fc_1 = nn.Linear(in_features=128*8*8, out_features=512)
        self.fc_2 = nn.Linear(in_features=512, out_features=256)

        self.head_from = nn.Linear(in_features=256, out_features=64)
        self.head_to = nn.Linear(in_features=256, out_features=64)
        self.head_prom = nn.Linear(in_features=256, out_features=5)


    def forward(self, t):
        t = F.relu(self.bn_1(self.conv_1(t)))
        t = F.relu(self.bn_2(self.conv_2(t)))

        t = t.reshape(t.size(0), -1)

        t = F.relu(self.fc_1(t))
        t = self.dropout(t)
        t = F.relu(self.fc_2(t))
        t = self.dropout(t)

        from_logits = self.head_from(t)
        to_logits = self.head_to(t)
        prom_logits = self.head_prom(t)

        return from_logits, to_logits, prom_logits


def save_model(model, optimizer, epoch, loss, path="models/chess_model.pth"):
    """
    Сохраняет модель, оптимизатор и метаданные
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    
    print(f"Модель сохранена: {path}")