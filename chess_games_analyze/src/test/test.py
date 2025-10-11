import os
import pandas as pd
import numpy as np
import chess
import torch
import torch.nn.functional as F
import torchmetrics
from src.training.trainer import apply_masks
from src.model.model import ChessModel
from src.model.utils import load_model


def test_model(dataloader, device):
    """
    Функция теста
    """
    model = ChessModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model = load_model(model, device, "models/best_model.pth")
    model.eval()

    # метрики
    # accuracy
    acc_from = torchmetrics.Accuracy(task="multiclass", num_classes=64).to(device)
    acc_to = torchmetrics.Accuracy(task="multiclass", num_classes=64).to(device)
    acc_prom = torchmetrics.Accuracy(task="multiclass", num_classes=5).to(device)
    # top 3
    top_k_from = torchmetrics.Accuracy(task="multiclass", num_classes=64, top_k=3).to(
        device
    )
    top_k_to = torchmetrics.Accuracy(task="multiclass", num_classes=64, top_k=3).to(
        device
    )
    top_k_prom = torchmetrics.Accuracy(task="multiclass", num_classes=5, top_k=3).to(
        device
    )

    total_loss = 0
    total_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):

            board_tensor = batch["board_tensor"].to(device).float()
            from_target = batch["from_target"].to(device)
            to_target = batch["to_target"].to(device)
            prom_target = batch["prom_target"].to(device)
            from_mask = batch["from_mask"].to(device)
            legal_moves_matrix = batch["legal_moves_matrix"].to(device)

            board_tensor = board_tensor.permute(0, 3, 1, 2)
            # предсказания модели
            from_logits, to_logits, prom_logits = model(board_tensor)

            loss = apply_masks(
                from_logits,
                to_logits,
                prom_logits,
                from_mask,
                legal_moves_matrix,
                from_target,
                to_target,
                prom_target,
            )    
            total_loss += loss.item()
            total_batches += 1        

            # обновление
            # accuracy
            acc_from.update(from_logits, from_target.argmax(dim=1))
            acc_to.update(to_logits, to_target.argmax(dim=1))
            acc_prom.update(prom_logits, prom_target.argmax(dim=1))
            # top 3
            top_k_from.update(from_logits, from_target.argmax(dim=1))
            top_k_to.update(to_logits, to_target.argmax(dim=1))
            top_k_prom.update(prom_logits, prom_target.argmax(dim=1))

    # финальные метрики
    avg_loss = total_loss / total_batches
    # accuracy
    final_acc_from = acc_from.compute()
    final_acc_to = acc_to.compute()
    final_acc_prom = acc_prom.compute()
    # top 3
    final_top_k_from = top_k_from.compute()
    final_top_k_to = top_k_to.compute()
    final_top_k_prom = top_k_prom.compute()

    print(f"Средний Loss: {avg_loss:.4f}")
    print (f'Accuracy from {final_acc_from:.3f}')
    print (f'Accuracy to {final_acc_to:.3f}')
    print (f'Accuracy prom {final_acc_prom:.3f}')
    print (f'Top-3 from {final_top_k_from:.3f}')
    print (f'Top-3 to {final_top_k_to:.3f}')
    print (f'Top-3 prom {final_top_k_prom:.3f}')

    return {
        'loss': avg_loss,
        'accuracy_from': final_acc_from,
        'accuracy_to': final_acc_to,
        'accuracy_prom': final_acc_prom,
        'top3_from': final_top_k_from,
        'top3_to': final_top_k_to,
        'top3_prom': final_top_k_prom
    }