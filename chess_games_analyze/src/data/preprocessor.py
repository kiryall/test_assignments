import os
import pandas as pd
import numpy as np
import chess
import torch


def fen_to_tensor(fen_str):
    """
    Преобразует строку формате FEN в тензор (8, 8, 18)

    Args:
        fen_str (str): строка в формате FEN

    Returns:
        torch.tensor
    """

    try:
        board = chess.Board(fen_str)
    except Exception as e:
        print(f"Invalid FEN: {fen_str[:50]}... Error: {e}")
        return None

    # определяем тензор 8х8х18
    tensor = torch.zeros((8, 8, 18), dtype=torch.int8)
    # типы фигур
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

    # заполняем фигуры
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            # индекс типа фигуры
            piece_type_idx = piece_types.index(piece.piece_type)
            # сдвиг по цвету
            color = 0 if piece.color == chess.WHITE else 6
            # определение плоскости
            plane = color + piece_type_idx
            # определение строки и столбца
            row = 7 - (square // 8)
            col = square % 8
            tensor[row, col, plane] = 1

    # плоскость с очередбю хода
    turn = 1 if board.turn == chess.WHITE else -1
    tensor[:, :, 12] = turn

    # плоскости рокировок
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[:, :, 13] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[:, :, 14] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[:, :, 15] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[:, :, 16] = 1.0

    # взятие на проходе
    if board.ep_square is not None:
        ep_square = board.ep_square
        row = 7 - (ep_square // 8)
        col = ep_square % 8
        tensor[row, col, 17] = 1    

    return tensor


def move_to_target(uci, board=None):
    """
    Преобразует ход в формате UCI
    в три целевых вектора для многоголовой модели
    
    Args:
        uci (str): Ход в строковом формате
        board (chess.Board): Позиция, из которой делается ход
    
    Returns:
        tuple: (from_target, to_target, promotion_target)
    """
    
    # объект хода из строки
    try:
        move = chess.Move.from_uci(uci)
    except Exception as e:
        print(f"Invalid move... Error: {e}")
        return None    

    # вектор "откуда"
    from_target = np.zeros(64, dtype=np.int8)
    from_target[move.from_square] = 1

    # вектор "куда"
    to_target = np.zeros(64, dtype=np.int8)
    to_target[move.to_square] = 1

    # вектор "превращение"
    promotion_target = np.zeros(5, dtype=np.int8)
    promotion_pieces = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]

    if move.promotion:
        promotion_idx = promotion_pieces.index(move.promotion)
        promotion_target[promotion_idx] = 1
    else:
        promotion_target[0] = 1

    return from_target, to_target, promotion_target