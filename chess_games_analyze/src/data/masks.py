import os
import pandas as pd
import numpy as np
import chess
import torch


def get_position_masks(board):
    """
    Возвращает словарь масок легальных ходов
    
    Args:
        board (chess.Board): Позиция, из которой делается ход
    
    Returns:
        masks (dict): 'from_mask' - вектор [64] мест откуда можно ходить
                    'legal_moves_matrix' - тензор 64х64х5 всех легальных ходов
                    4 плоскости показывают превращения пешек
    """

    masks = {}

    masks['from_mask'] = np.zeros(64, dtype=np.float32)
    masks['legal_moves_matrix'] = np.zeros((64, 64, 5), dtype=np.float32)
    
    for move in board.legal_moves:
        from_sq = move.from_square
        to_sq = move.to_square
        
        masks['from_mask'][from_sq] = 1.0
        
        # Определяем тип фигуры и цвет
        piece = board.piece_at(from_sq)
        if not piece:
            continue
            
        is_pawn_promotion = (piece.piece_type == chess.PAWN and 
                            ((piece.color == chess.WHITE and chess.square_rank(to_sq) == 7) or
                             (piece.color == chess.BLACK and chess.square_rank(to_sq) == 0)))
        
        if is_pawn_promotion:
            # Пешка на последней горизонтали - ВСЕ 4 превращения
            for prom_idx in range(1, 5):
                masks['legal_moves_matrix'][from_sq, to_sq, prom_idx] = 1.0
        else:
            # Все остальные ходы - только "нет превращения"
            masks['legal_moves_matrix'][from_sq, to_sq, 0] = 1.0
    
    return masks