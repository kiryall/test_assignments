from torch.utils.data import Dataset
import torch
import chess
from src.data.preprocessor import fen_to_tensor, move_to_target
from src.data.masks import get_position_masks

class ChessDataset(Dataset):
    """
    Класс загружает данные батчами.
    Применяет функции преобразования ходов в тензоры.
    Создает маски легальных ходов.

    Returns:
        Dict (словарь тензоров 
        'board_tensor' (8х8х18)
        'from_target' (64)
        'to_target' (64)
        'promotion_target' (5)
        'from_mask' (64)
        'legal_moves_matrix' (64х64х5))
    """

    def __init__(self, df):
        """
        Args:
            df: DataFrame
        """
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        fen = self.df.iloc[idx]['fen']
        move_uci = self.df.iloc[idx]['move']

        board = chess.Board(fen)

        # позиции и ходы
        board_tensor = fen_to_tensor(fen)
        from_target, to_target, promotion_target = move_to_target(move_uci)
        # маски
        masks = get_position_masks(board)

        return {
            'board_tensor': torch.tensor(board_tensor, dtype=torch.float32),
            'from_target': torch.tensor(from_target, dtype=torch.float32),
            'to_target': torch.tensor(to_target, dtype=torch.float32),
            'prom_target': torch.tensor(promotion_target, dtype=torch.float32),
            'from_mask': torch.tensor(masks['from_mask'], dtype=torch.float32),
            'legal_moves_matrix': torch.tensor(masks['legal_moves_matrix'], dtype=torch.float32)
        }