import numpy as np
from .utils import *
from scipy.signal import convolve2d

best_moves = []
score = {
    4: 10000,
    3: 500,
    2: 50,
    1: 5
}

bonus = {
    2: 10,
    3: 15,
    4: 10,
}

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
    def __init__(self, depth: int = 5, player: int = -1):
        self.depth = depth 
        self.player = player
        self.system = System()
        self.directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    def get_kernel(self, x, y):
        kernel = (
            np.ones((4, 4)) if (x, y) in [(1, 1), (1, -1)] else
            np.ones((4, 1)) if (x, y) == (1, 0) else
            np.ones((1, 4))
            )
        return kernel
    
    def make_move(self, board, col, player):
        new_board = np.copy(board)
        for row in reversed(range(board.shape[0])):
            if new_board[row, col] == 0:
                new_board[row, col] = player
                break
        return new_board
    
    def valid_moves(self, board):
        valid_moves = [c for c in range(board.shape[1]) if board[0, c] == 0]
        return valid_moves
    
    def get_best_move(self, board):
        if not self.valid_moves(board):
            return None
            
        forced_move = self.get_forced_move(board)
        if forced_move is not None:
            return forced_move    
          
        adaptive_depth = self.get_adaptive_depth(board)  
        _, col = self.minimax(board, adaptive_depth, -np.inf, np.inf, True)
        
        b = ({_}, {col})
        print(f'Лучший ход: ({b})')
        best_moves.append(b)
        return col
        
    def minimax(self, board, depth, alpha, beta, maximizing):
        current_player = self.player if maximizing else -self.player
        opponent = -current_player
        if depth == 0 or self.system.check_win(board, current_player) or \
        self.system.check_win(board, opponent):
            return self.evaluate(board), None
        
        valid_moves = self.valid_moves(board)
        if not valid_moves:
            return 0, None
        
        valid_moves = sorted(valid_moves, key=lambda x: abs(x - 3))
        best_col = valid_moves[0]
        # ходы AI
        if maximizing:
            max_eval = -np.inf # alpha
            for col in valid_moves:
                child = self.make_move(board, col, self.player) 
                eval, _ = self.minimax(child, depth-1, alpha, beta, False)
                print(f'Ход AI в колонку {col}: оценка = {eval}')
                if eval > max_eval:
                    max_eval = eval
                    best_col = col
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_col
        else:
            # ходы Human
            min_eval = np.inf # beta
            for col in valid_moves:
                child = self.make_move(board, col, -self.player) 
                eval, _ = self.minimax(child, depth-1, alpha, beta, True)
                print(f'Ход Human в колонку {col}: оценка = {eval}')
                if eval < min_eval:
                    min_eval = eval
                    best_col = col
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_col
    
    def evaluate(self, board):
        ai_score = 0
        human_score = 0
        if self.system.check_win(board, self.player):
            return 10000
        if self.system.check_win(board, -self.player):
            return -10000
        
        if self.depth % 2 == 0:
            for col in self.valid_moves(board):
                test_board = self.make_move(board, col, -self.player)
                if self.detect_potential_double_threats(test_board, -self.player):
                    return -9500
                else:
                    for col2 in self.valid_moves(test_board):
                        test_board_2 = self.make_move(test_board, col2, -self.player)
                        if self.detect_potential_double_threats(test_board_2, -self.player):
                            return -9500
        
        ai_threats, ai_double, _ = self.detect_threats(board, self.player)
        human_threats, human_double, _ = self.detect_threats(board, -self.player)
        
        if ai_double:
            return 9500
        if human_double:
            return -9500
        
        if self.detect_potential_double_threats(board, self.player):
            return 9000
        if self.detect_potential_double_threats(board, -self.player):
            return -9000
        
        for player in [self.player, -self.player]:
            player_score = 0
            for dx, dy in self.directions:
                kernel = self.get_kernel(dx, dy)
                mask = (board == player).astype(int)
                conv = convolve2d(mask, kernel, mode='valid')
                
                for chip, sc in score.items():
                    player_score += np.sum(conv == chip) * sc

            center_bonus = 0
            for col, sc in bonus.items():       
                center_bonus += np.sum(board[:, col] == player) * sc
            player_score += center_bonus
        
            if player == self.player:
                ai_score = player_score + (ai_threats * 2000)
            else:
                human_score = player_score + (human_threats * -2000)
                
        return ai_score - human_score
    
    def detect_threats(self, board, opponent):
        threats = 0
        threat_cols = []
        for col in range(board.shape[1]):
            if board[0, col] == 0:
                test_board = self.make_move(board, col, opponent)
                if self.system.check_win(test_board, opponent):
                    threats += 1  
                    threat_cols.append(col)
                    
        double_threat = threats >= 2
        return threats, double_threat, threat_cols

    def detect_potential_double_threats(self, board, player):
        for col in self.valid_moves(board):
            test_board = self.make_move(board, col, player)
            threats, _, _ = self.detect_threats(test_board, player)
            if threats >= 2:
                return True
        return False 

    def get_forced_move(self, board):
        _, _, threat_cols = self.detect_threats(board, -self.player)
        if threat_cols:
            return threat_cols[0]
        
        for col in self.valid_moves(board):
            test_board = self.make_move(board, col, self.player)
            if self.system.check_win(test_board, self.player):
                return col
    
        for col in self.valid_moves(board):
            test_board = self.make_move(board, col, -self.player)
            if self.system.check_win(test_board, -self.player):
                return col
        return None
    
    def get_adaptive_depth(self, board):
        filled_cells = np.count_nonzero(board)
        if filled_cells < 10:
            return self.depth
        elif filled_cells < 20:
            return 6
        elif filled_cells < 30:
            return 8
        else:
            return 10
