import os
import torch


def save_model(model, optimizer, epoch, loss, path="models/chess_model.pth"):
    """
    Сохраняет модель, оптимизатор и метаданные
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Сохраняем checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    
    print(f"Модель сохранена: {path}")


def load_model(model, device=None, path="models/chess_model.pth"):
    """
    Загружает модель и оптимизатор
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    
    print(f"Модель загружена: {path}")
    
    return model