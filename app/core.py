import numpy as np
from .utils import *
from scipy.signal import convolve2d

class System:
    def check_horizontal(self, board, player):
        mask = (board == player).astype(int)
        return np.any(convolve2d(mask, [[1, 1, 1, 1]], mode='valid') >= 4)
    
    def check_vertical(self, board, player):
        return self.check_horizontal(board.T, player)
        
    def check_diagonals(self, board, player):
        mask = (board == player).astype(int)
        for flipped in [mask, np.fliplr(mask)]:
            for offset in range(-2, 4):
                diag = np.diagonal(flipped, offset=offset)
                if len(diag) >= 4 and np.any(np.convolve(diag, [1, 1, 1, 1], mode='valid') >= 4):
                    return True
        return False
        
    def check_win(self, board, player):
        return (
            self.check_horizontal(board, player)
            or self.check_vertical(board, player)
            or self.check_diagonals(board, player)
        )    
        
class MiniMax:
    def __init__(self, depth=5, player=-1):
        self.depth = depth 
        self.player = player
        self.system = System()
        self.directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
      
    def get_best_move(self, board):
        valid_moves = [c for c in range(board.shape[1]) if board[0, c] == 0]
        if not valid_moves:
            return None
            
        _, col = self.minimax(board, self.depth, -np.inf, np.inf, True)
        return col
        
    def minimax(self, board, depth, alpha, beta, maximizing):
        current_player = self.player if maximizing else -self.player
        opponent = -current_player
        if depth == 0 or self.system.check_win(board, current_player) or \
        self.system.check_win(board, opponent):
            return self.evaluate(board), None
        
        valid_moves = [c for c in range(board.shape[1]) if board[0, c] == 0]
        if not valid_moves:
            return 0, None
        
        best_col = valid_moves[0]
        
        if maximizing:
            max_eval = -np.inf
            for col in valid_moves:
                child = self.make_move(board, col, self.player)
                eval, _ = self.minimax(child, depth-1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_col = col
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_col
        else:
            min_eval = np.inf
            for col in valid_moves:
                child = self.make_move(board, col, -self.player)
                eval, _ = self.minimax(child, depth-1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_col = col
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_col
    
    def evaluate(self, board):
        scores = {self.player: 0, -self.player: 0}
        
        for player in [self.player, -self.player]:
            for dx, dy in self.directions:
                kernel = (
                    np.ones((4, 4)) if (dx, dy) in [(1, 1), (1, -1)] else
                    np.ones((4, 1)) if (dx, dy) == (1, 0) else
                    np.ones((1, 4))
                    )
                
                mask = (board == player).astype(int)
                conv = convolve2d(mask, kernel, mode='valid')
                scores[player] += np.sum(conv == 3) * 100 + np.sum(conv == 2) * 10
        
        return scores[self.player] - scores[-self.player]
    
    def make_move(self, board, col, player):
        new_board = np.copy(board)
        for row in reversed(range(board.shape[0])):
            if new_board[row, col] == 0:
                new_board[row, col] = player
                break
        return new_board

# board.shape[0] = rows
# board.shape[1] = cols
    