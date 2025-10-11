import os
import pandas as pd
import numpy as np
import chess
import torch
import torch.nn.functional as F
import torchmetrics
from src.model.model import ChessModel
from src.model.utils import save_model


def apply_masks(
    from_logits,
    to_logits,
    prom_logits,
    from_mask,
    legal_moves_matrix,
    from_target,
    to_target,
    prom_target,
):
    """
    Функция применяет маски

    Args:
        from_logits: предсказание "откуда ход"
        to_logits: предсказание "куда ход"
        prom_logits: предсказания превращений
        from_mask: маска доступных для хода фигур
        legal_moves_matrix: маска легальных ходов
        from_target: таргет "откуда ход"
        to_target: таргет "куда ход"
        prom_target: таргет превращений

    Returns:
        loss
    """

    batch_size = from_logits.size(0)

    # Маскируем from_logits для всего батча
    from_logits_masked = from_logits * from_mask
    from_logits_masked[from_mask == 0] = -(10**9)

    # вычисляем лосс
    loss_from = F.cross_entropy(from_logits_masked, from_target)

    true_from_sq = torch.argmax(from_target, dim=1)
    true_to_sq = torch.argmax(to_target, dim=1)

    # получение масок для to_logits
    batch_indices = torch.arange(batch_size, device=from_logits.device)
    to_masks = legal_moves_matrix[batch_indices, true_from_sq].max(dim=2)[0]

    to_logits_masked = to_logits * to_masks
    to_logits_masked[to_masks == 0] = -(10**9)
    # вычисляем лосс
    loss_to = F.cross_entropy(to_logits_masked, to_target)

    # получение масок для to_logits
    prom_masks = legal_moves_matrix[batch_indices, true_from_sq, true_to_sq]

    # маскируем  prom_logits
    prom_logits_masked = prom_logits * prom_masks
    prom_logits_masked[prom_masks == 0] = -(10**9)
    loss_prom = F.cross_entropy(prom_logits_masked, prom_target)


    return loss_from + loss_to + loss_prom


def train_epoch(model, dataloader, optimizer, device):
    """
    Функция обучения

    Args:
        model: инициализированная модель
        dataloader: загрузчик данных
        optimizer, device

    Returns:
        total_loss / len(dataloader)
    """

    model.train()
    total_loss = 0

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

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # обновление
        # accuracy
        acc_from.update(from_logits, from_target.argmax(dim=1))
        acc_to.update(to_logits, to_target.argmax(dim=1))
        acc_prom.update(prom_logits, prom_target.argmax(dim=1))
        # top 3
        top_k_from.update(from_logits, from_target.argmax(dim=1))
        top_k_to.update(to_logits, to_target.argmax(dim=1))
        top_k_prom.update(prom_logits, prom_target.argmax(dim=1))

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.3f}")

    # финальные метрики
    # accuracy
    final_acc_from = acc_from.compute()
    final_acc_to = acc_to.compute()
    final_acc_prom = acc_prom.compute()
    # top 3
    final_top_k_from = top_k_from.compute()
    final_top_k_to = top_k_to.compute()
    final_top_k_prom = top_k_prom.compute()

    avg_loss = total_loss / len(dataloader)

    return (
        avg_loss,
        final_acc_from,
        final_acc_to,
        final_acc_prom,
        final_top_k_from,
        final_top_k_to,
        final_top_k_prom,
    )


def train_model(dataloader, device):
    """
    Функция для тренировки модели
    """

    model = ChessModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_loss = float('inf')

    for epoch in range(10):
        (   avg_loss,
            final_acc_from,
            final_acc_to,
            final_acc_prom,
            final_top_k_from,
            final_top_k_to,
            final_top_k_prom,
        ) = train_epoch(model, dataloader, optimizer, device)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, optimizer, epoch, avg_loss, "models/best_model.pth")
            print(f"Новая лучшая модель. Loss: {avg_loss:.4f}")

        print(f"Epoch {epoch}, Train Loss: {avg_loss:.4f}")
        print(
            f"Accuracy: from {final_acc_from:.4f}, to {final_acc_to:.4f}, prom {final_acc_prom:.4f}"
        )
        print(
            f"Top-K: from {final_top_k_from:.4f}, to {final_top_k_to:.4f}, prom {final_top_k_prom:.4f}"
        )
