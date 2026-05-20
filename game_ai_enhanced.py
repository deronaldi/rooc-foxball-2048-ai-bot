"""
2048 Game AI Bot - Main Script (Enhanced Version)
Reads screen, detects tiles, and plays optimally
"""

import mss
import numpy as np
from PIL import Image
import pyautogui
import time
from typing import List, Tuple, Optional
import keyboard
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Board2048:
    """Represents a 2048 game board"""
    
    def __init__(self, grid: Optional[np.ndarray] = None):
        if grid is None:
            self.grid = np.zeros((4, 4), dtype=np.int32)
        else:
            self.grid = grid.copy()
    
    def move(self, direction: int) -> bool:
        """
        Move in a direction: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        Returns True if board changed
        """
        original = self.grid.copy()
        
        if direction == 0:  # UP
            self.grid = self._compress(self._merge(self._compress(
                np.rot90(self.grid, 3)
            ), np.rot90(self.grid, 3)))
        elif direction == 1:  # RIGHT
            self.grid = self._compress(self._merge(self._compress(
                np.rot90(self.grid, 2)[:, ::-1]
            ), np.rot90(self.grid, 2)[:, ::-1]))[:, ::-1]
        elif direction == 2:  # DOWN
            rotated = np.rot90(self.grid, 1)
            self.grid = self._compress(self._merge(self._compress(rotated), rotated))[::-1]
        elif direction == 3:  # LEFT
            self.grid = self._compress(self._merge(self._compress(self.grid), self.grid))
        
        return not np.array_equal(original, self.grid)
    
    def _compress(self, grid: np.ndarray) -> np.ndarray:
        """Move all non-zero values to the left"""
        new_grid = np.zeros_like(grid)
        for i in range(4):
            non_zero = grid[i, grid[i] > 0]
            new_grid[i, :len(non_zero)] = non_zero
        return new_grid
    
    def _merge(self, grid: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Merge equal adjacent tiles"""
        new_grid = grid.copy()
        for i in range(4):
            for j in range(3):
                if new_grid[i, j] != 0 and new_grid[i, j] == new_grid[i, j + 1]:
                    new_grid[i, j] *= 2
                    new_grid[i, j + 1] = 0
        return new_grid
    
    def get_empty_cells(self) -> np.ndarray:
        """Get coordinates of empty cells"""
        return np.argwhere(self.grid == 0)
    
    def get_score(self) -> int:
        """Calculate score (sum of all tiles)"""
        return np.sum(self.grid)
    
    def get_max_tile(self) -> int:
        """Get the maximum tile value"""
        return int(np.max(self.grid)) if np.max(self.grid) > 0 else 0
    
    def is_game_over(self) -> bool:
        """Check if no moves are possible"""
        for direction in range(4):
            test_board = Board2048(self.grid)
            if test_board.move(direction):
                return False
        return True


class Game2048AI:
    """AI player for 2048 game"""
    
    def __init__(self, x0: int, y0: int, x1: int, y1: int, move_interval: float = 1.0, use_ocr: bool = False):
        """
        Initialize the AI
        x0, y0: top-left corner of game board on screen
        x1, y1: bottom-right corner of game board on screen
        move_interval: seconds between moves
        use_ocr: whether to use OCR for tile detection
        """
        self.x0, self.y0 = x0, y0
        self.x1, self.y1 = x1, y1
        self.move_interval = move_interval
        self.running = True
        self.board = Board2048()
        self.use_ocr = use_ocr
        self.move_count = 0
        
    def capture_screen(self) -> np.ndarray:
        """Capture specified region of screen"""
        try:
            with mss.mss() as sct:
                monitor = {
                    "top": self.y0,
                    "left": self.x0,
                    "width": self.x1 - self.x0,
                    "height": self.y1 - self.y0
                }
                screenshot = sct.grab(monitor)
                return np.array(screenshot)
        except Exception as e:
            logger.error(f"Failed to capture screen: {e}")
            return None
    
    def extract_board_from_image(self, image: np.ndarray) -> Optional[Board2048]:
        """
        Extract 2048 board state from screenshot
        Attempts simple color-based detection
        """
        if image is None:
            return None
        
        try:
            board_grid = self._detect_tiles_color(image)
            return Board2048(board_grid)
        except Exception as e:
            logger.warning(f"Board extraction error: {e}")
            return self.board  # Return previous board state
    
    def _detect_tiles_color(self, image: np.ndarray) -> np.ndarray:
        """Detect tiles by analyzing grid pattern"""
        board = np.zeros((4, 4), dtype=np.int32)
        height, width = image.shape[:2]
        
        cell_height = height // 4
        cell_width = width // 4
        
        for row in range(4):
            for col in range(4):
                y1 = row * cell_height + 5
                y2 = (row + 1) * cell_height - 5
                x1 = col * cell_width + 5
                x2 = (col + 1) * cell_width - 5
                
                if y1 < y2 and x1 < x2:
                    cell = image[y1:y2, x1:x2]
                    
                    # Check if cell is mostly empty (light color)
                    if len(cell.shape) == 3 and cell.shape[2] >= 3:
                        avg_brightness = np.mean(cell[:, :, :3])
                        
                        if avg_brightness > 200:  # Empty cell
                            board[row, col] = 0
                        else:
                            # Occupied cell - color analysis for value
                            board[row, col] = 2  # Default: could improve with OCR
        
        return board
    
    def minimax(self, board: Board2048, depth: int = 3, is_maximizing: bool = True, 
                alpha: float = -float('inf'), beta: float = float('inf')) -> float:
        """
        Minimax with alpha-beta pruning
        """
        if depth == 0 or board.is_game_over():
            return self.evaluate(board)
        
        if is_maximizing:
            max_eval = -float('inf')
            for direction in range(4):
                test_board = Board2048(board.grid)
                if test_board.move(direction):
                    eval_score = self.minimax(test_board, depth - 1, False, alpha, beta)
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break
            return max_eval if max_eval != -float('inf') else self.evaluate(board)
        else:
            min_eval = float('inf')
            empty_cells = board.get_empty_cells()
            if len(empty_cells) == 0:
                return self.evaluate(board)
            
            # Sample random new tiles
            sample_size = min(2, len(empty_cells))
            for _ in range(sample_size):
                if len(empty_cells) > 0:
                    idx = np.random.randint(len(empty_cells))
                    test_board = Board2048(board.grid)
                    test_board.grid[empty_cells[idx, 0], empty_cells[idx, 1]] = 2 if np.random.random() > 0.1 else 4
                    eval_score = self.minimax(test_board, depth - 1, True, alpha, beta)
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break
            
            return min_eval if min_eval != float('inf') else self.evaluate(board)
    
    def evaluate(self, board: Board2048) -> float:
        """Heuristic evaluation of board state"""
        score = 0
        
        # Prefer higher tiles
        score += board.get_score() * 2
        
        # Prefer large tiles in corners
        corners = [
            board.grid[0, 0], board.grid[0, 3],
            board.grid[3, 0], board.grid[3, 3]
        ]
        score += max(corners) * 3 if max(corners) > 0 else 0
        
        # Prefer more empty cells
        empty_cells = len(board.get_empty_cells())
        score += empty_cells * 20
        
        # Prefer smooth boards
        unique_tiles = len(np.unique(board.grid[board.grid > 0]))
        score -= unique_tiles * 3
        
        return score
    
    def find_best_move(self) -> int:
        """Find the best move using minimax"""
        best_move = -1
        best_score = -float('inf')
        moves_evaluated = []
        
        for direction in range(4):
            test_board = Board2048(self.board.grid)
            if test_board.move(direction):
                score = self.minimax(test_board, depth=3, is_maximizing=False)
                moves_evaluated.append((direction, score))
                if score > best_score:
                    best_score = score
                    best_move = direction
        
        if best_move == -1 and len(moves_evaluated) > 0:
            best_move = moves_evaluated[0][0]
        
        return best_move
    
    def execute_move(self, direction: int):
        """Execute a move by simulating key press"""
        keys = ['up', 'right', 'down', 'left']
        try:
            pyautogui.press(keys[direction])
            self.board.move(direction)
            self.move_count += 1
            logger.info(f"[Move {self.move_count}] {keys[direction].upper()} | Score: {self.board.get_score()} | Max: {self.board.get_max_tile()}")
        except Exception as e:
            logger.error(f"Failed to execute move: {e}")
    
    def run(self):
        """Main game loop"""
        logger.info("="*60)
        logger.info("🤖 2048 AI Bot Starting")
        logger.info("="*60)
        logger.info(f"Board region: ({self.x0}, {self.y0}) to ({self.x1}, {self.y1})")
        logger.info(f"Move interval: {self.move_interval}s")
        logger.info(f"Screen capture resolution: {self.x1 - self.x0}x{self.y1 - self.y0}")
        logger.info("Press Ctrl+C to stop")
        logger.info("="*60 + "\n")
        
        time.sleep(2)  # Wait before starting
        
        try:
            while self.running:
                # Capture and update board
                screenshot = self.capture_screen()
                if screenshot is not None:
                    new_board = self.extract_board_from_image(screenshot)
                    if new_board is not None:
                        self.board = new_board
                
                # Find and execute best move
                best_move = self.find_best_move()
                if best_move != -1:
                    self.execute_move(best_move)
                
                # Wait for next move
                time.sleep(self.move_interval)
                
        except KeyboardInterrupt:
            logger.info("\n⏹️ Bot stopped by user")
        except Exception as e:
            logger.error(f"❌ Error: {e}")
        finally:
            logger.info("="*60)
            logger.info(f"📊 Final Statistics:")
            logger.info(f"   Moves made: {self.move_count}")
            logger.info(f"   Final Score: {self.board.get_score()}")
            logger.info(f"   Max Tile: {self.board.get_max_tile()}")
            logger.info("="*60)


def load_coordinates() -> Optional[Tuple[int, int, int, int]]:
    """Try to load coordinates from config file"""
    config_file = 'board_coordinates.txt'
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                coords = f.readline().strip().split(',')
                if len(coords) == 4:
                    return tuple(map(int, coords))
        except Exception as e:
            logger.warning(f"Could not load coordinates: {e}")
    return None


if __name__ == "__main__":
    # Try to load saved coordinates
    coords = load_coordinates()
    
    if coords:
        x0, y0, x1, y1 = coords
        logger.info(f"✅ Loaded coordinates from board_coordinates.txt: ({x0}, {y0}) to ({x1}, {y1})")
    else:
        logger.warning("⚠️  No saved coordinates found!")
        logger.info("Run: python calibrate.py")
        logger.info("Or manually set coordinates below:")
        
        # Manual defaults (CHANGE THESE!)
        x0, y0 = 100, 100      # ← UPDATE: TOP-LEFT corner
        x1, y1 = 500, 500      # ← UPDATE: BOTTOM-RIGHT corner
    
    # Create and run bot
    bot = Game2048AI(x0, y0, x1, y1, move_interval=1.0, use_ocr=False)
    bot.run()
