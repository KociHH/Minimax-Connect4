import pygame
import sys
from .utils import *
from .core import System, MiniMax, best_moves
import numpy as np


class Game:
    def __init__(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=int)
        self.human = 1
        self.ai = -1
        self.turn = self.human
        self.system = System()
        self.minimax = MiniMax(depth=6, player=self.ai)
        self.game_over = False
        self.winner = None
        self.font = pygame.font.SysFont("Arial", 36, bold=True)
        
    def draw_board(self, screen):
        screen.fill(COLORS['BLUE'])  
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                cell_value = self.board[row, col]
                if cell_value == self.human:
                    color = COLORS['YELLOW']
                elif cell_value == self.ai:
                    color = COLORS['RED']
                else:
                    color = COLORS['WHITE']  
                pygame.draw.circle(screen, color, 
                    (col * CELL_SIZE + CELL_SIZE // 2, (row + 1) * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 2 - 5)
        
        pygame.draw.rect(screen, COLORS['WHITE'], (0, 0, WIDTH, CELL_SIZE))
        status = "Your Turn" if self.turn == self.human else "AI Thinking"
        text = self.font.render(status, True, COLORS['BLACK'])
        screen.blit(text, (10, 10))

    def make_move(self, col, player):
        print(self.board, end="\n\n")
        if self.board[0, col] != 0: 
            return False
        for row in reversed(range(BOARD_ROWS)):
            if self.board[row, col] == 0:
                self.board[row, col] = player
                if self.system.check_win(self.board, player):
                    self.game_over = True
                    self.winner = 'You' if player == self.human else 'AI'
                return True
        return False
    

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Connect Four")
    game = Game()
    clock = pygame.time.Clock()
        
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            if event.type == pygame.MOUSEBUTTONDOWN and not game.game_over:
                x, y = event.pos
                col = x // CELL_SIZE
                if game.turn == game.human:
                    if game.make_move(col, game.human):
                        print('Сделан ход игрока')
                        game.draw_board(screen)
                        pygame.display.flip()
                        game.turn = game.ai

        if not game.game_over and game.turn == game.ai:
            col = game.minimax.get_best_move(game.board)
            if col is not None:
                game.make_move(col, game.ai)
                game.draw_board(screen)
                pygame.display.flip()
                game.turn = game.human 
        
        if game.game_over:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill(COLORS['LIGHT_WHITE'])
            screen.blit(overlay, (0, 0))
            
            text_winner = game.font.render(f'{game.winner} Winner', True, COLORS['BLACK'])
            rect = text_winner.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            screen.blit(text_winner, rect)
            
            pygame.display.flip()
            print(game.board)
            print(f'Лучшие ходы за партию:')
            for i in range(len(best_moves)):
                print(f'{i + 1}. {best_moves[i]}')
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
        else:
            game.draw_board(screen)
                        
        pygame.display.update()
        clock.tick(30)
        
        
