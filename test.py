import datetime
import math
from pathlib import Path
import shutil
import time

import cv2
import mss
import numpy as np
from PIL import Image
import pyautogui
import pytesseract
from typing import Iterable, List, Sequence, Tuple, Optional
import logging

from board_reader import BoardReader

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
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
            self.grid = self._move_left(self.grid.T).T
        elif direction == 1:  # RIGHT
            self.grid = self._move_left(self.grid[:, ::-1])[:, ::-1]
        elif direction == 2:  # DOWN
            self.grid = self._move_left(self.grid.T[:, ::-1])[:, ::-1].T
        elif direction == 3:  # LEFT
            self.grid = self._move_left(self.grid)
        
        return not np.array_equal(original, self.grid)
    
    def _move_left(self, grid: np.ndarray) -> np.ndarray:
        """Merge tiles to the left"""
        new_grid = np.zeros_like(grid)
        
        for i in range(4):
            row = grid[i, grid[i] > 0]
            merged = []
            j = 0
            while j < len(row):
                if j + 1 < len(row) and row[j] == row[j + 1]:
                    merged.append(row[j] * 2)
                    j += 2
                else:
                    merged.append(row[j])
                    j += 1
            
            new_grid[i, :len(merged)] = merged
        
        return new_grid
    
    def get_empty_cells(self) -> np.ndarray:
        """Get coordinates of empty cells"""
        return np.argwhere(self.grid == 0)
    
    def get_score(self) -> int:
        """Calculate score (sum of all tiles)"""
        return np.sum(self.grid)
    
    def get_max_tile(self) -> int:
        """Get the maximum tile value"""
        return np.max(self.grid)

    @staticmethod
    def log2_tile(tile: int) -> int:
        """Return the log2 rank of a tile value; empty cells have rank 0."""
        return 0 if tile == 0 else int(math.log2(tile))
    
    def is_game_over(self) -> bool:
        """Check if no moves are possible"""
        for direction in range(4):
            test_board = Board2048(self.grid)
            if test_board.move(direction):
                return False
        return True
    
    def evaluate(self) -> float:
        """
        Estimate board quality using the standalone expectimax heuristic.

        Larger scores are better. The score rewards empty cells, immediately
        possible merges, monotonic/smooth ordering, and keeping the max tile in a
        corner.
        """
        empty = len(self.get_empty_cells())
        ranks = [[self.log2_tile(int(value)) for value in row] for row in self.grid]

        merges = 0
        smoothness_penalty = 0.0
        for row in range(4):
            for col in range(4):
                if self.grid[row, col] == 0:
                    continue
                if col + 1 < 4 and self.grid[row, col + 1] != 0:
                    if self.grid[row, col] == self.grid[row, col + 1]:
                        merges += 1
                    smoothness_penalty += abs(ranks[row][col] - ranks[row][col + 1])
                if row + 1 < 4 and self.grid[row + 1, col] != 0:
                    if self.grid[row, col] == self.grid[row + 1, col]:
                        merges += 1
                    smoothness_penalty += abs(ranks[row][col] - ranks[row + 1][col])

        monotonicity_penalty = 0.0
        lines: Iterable[Sequence[int]] = list(ranks) + [list(col) for col in zip(*ranks)]
        for line in lines:
            increasing = sum(max(0, line[idx] - line[idx + 1]) for idx in range(3))
            decreasing = sum(max(0, line[idx + 1] - line[idx]) for idx in range(3))
            monotonicity_penalty += min(increasing, decreasing)

        max_tile = int(self.get_max_tile())
        corners = (self.grid[0, 0], self.grid[0, 3], self.grid[3, 0], self.grid[3, 3])
        corner_bonus = self.log2_tile(max_tile) * 30.0 if max_tile in corners else 0.0

        return (
            empty * 270.0
            + merges * 700.0
            + corner_bonus
            - smoothness_penalty * 12.0
            - monotonicity_penalty * 47.0
        )


def load_region_coordinates(path: str) -> Optional[Tuple[int, int, int, int]]:
    """Load region coordinates from a comma-separated text file."""
    config_file = Path(path)
    if not config_file.exists():
        return None

    try:
        text = config_file.read_text().strip()
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if len(parts) != 4:
            return None
        return tuple(map(int, parts))
    except Exception as e:
        logger.warning(f"Could not load coordinates from {config_file}: {e}")
        return None


class FastExpectimax2048:
    """
    Optimized expectimax engine using a 64-bit board representation.

    Each tile is stored as a 4-bit exponent: empty=0, tile 2=1, tile 4=2,
    tile 8=3, and so on. A row is therefore 16 bits, allowing all row movement
    and row scoring to be precomputed in 65,536-entry lookup tables.
    """

    BOARD_CELLS = 16
    ROW_MASK = 0xFFFF
    MAX_RANK = 15  # 4-bit nybble limit: 2^15 = 32768

    def __init__(self):
        self.row_left_table = [0] * 65536
        self.row_right_table = [0] * 65536
        self.row_score_table = [0.0] * 65536
        self._build_tables()

    def _build_tables(self):
        for row in range(65536):
            ranks = self._decode_row(row)
            moved_left = self._move_row_left_ranks(ranks)
            moved_right = list(reversed(self._move_row_left_ranks(list(reversed(ranks)))))

            self.row_left_table[row] = self._encode_row(moved_left)
            self.row_right_table[row] = self._encode_row(moved_right)
            self.row_score_table[row] = self._score_row(ranks)

    def _decode_row(self, row: int) -> List[int]:
        return [(row >> (4 * idx)) & 0xF for idx in range(4)]

    def _encode_row(self, ranks: List[int]) -> int:
        row = 0
        for idx, rank in enumerate(ranks):
            row |= (int(rank) & 0xF) << (4 * idx)
        return row

    def _move_row_left_ranks(self, ranks: List[int]) -> List[int]:
        nonzero = [rank for rank in ranks if rank]
        merged: List[int] = []
        idx = 0
        while idx < len(nonzero):
            if idx + 1 < len(nonzero) and nonzero[idx] == nonzero[idx + 1]:
                merged.append(min(nonzero[idx] + 1, self.MAX_RANK))
                idx += 2
            else:
                merged.append(nonzero[idx])
                idx += 1
        return merged + [0] * (4 - len(merged))

    def _score_row(self, ranks: List[int]) -> float:
        """Precompute row/column heuristic score for one 4-cell line."""
        empty = ranks.count(0)

        # Potential merges encourage boards with adjacent equal ranks.
        merge_score = 0.0
        for idx in range(3):
            if ranks[idx] and ranks[idx] == ranks[idx + 1]:
                merge_score += 1 << ranks[idx]

        # Smoothness penalizes large jumps between neighboring non-empty cells.
        smoothness_penalty = 0.0
        for idx in range(3):
            if ranks[idx] and ranks[idx + 1]:
                smoothness_penalty += abs(ranks[idx] - ranks[idx + 1])

        # Monotonicity: a good 2048 board tends to consistently increase or
        # decrease along each row/column. Penalize the weaker of both directions.
        increasing_penalty = 0.0
        decreasing_penalty = 0.0
        for idx in range(3):
            if ranks[idx] > ranks[idx + 1]:
                increasing_penalty += ranks[idx] - ranks[idx + 1]
            else:
                decreasing_penalty += ranks[idx + 1] - ranks[idx]
        monotonicity_penalty = min(increasing_penalty, decreasing_penalty)

        # Edge bonus keeps large tiles near board edges, matching the original
        # high-performing expectimax heuristic family.
        edge_bonus = max(ranks[0], ranks[3]) * 2.0

        return (
            270000.0 * empty +
            700.0 * merge_score +
            2500.0 * edge_bonus -
            47000.0 * monotonicity_penalty -
            700.0 * smoothness_penalty
        )

    def grid_to_board(self, grid: np.ndarray) -> int:
        board = 0
        for row in range(4):
            for col in range(4):
                value = int(grid[row, col])
                if value <= 0:
                    rank = 0
                else:
                    rank = int(np.log2(value))
                    rank = min(max(rank, 0), self.MAX_RANK)
                board |= rank << (4 * (row * 4 + col))
        return board

    def board_to_grid(self, board: int) -> np.ndarray:
        grid = np.zeros((4, 4), dtype=np.int32)
        for row in range(4):
            for col in range(4):
                rank = (board >> (4 * (row * 4 + col))) & 0xF
                grid[row, col] = 0 if rank == 0 else (1 << rank)
        return grid

    def _get_row(self, board: int, row: int) -> int:
        return (board >> (row * 16)) & self.ROW_MASK

    def _set_row(self, board: int, row: int, value: int) -> int:
        shift = row * 16
        return (board & ~(self.ROW_MASK << shift)) | ((value & self.ROW_MASK) << shift)

    def _get_col(self, board: int, col: int) -> int:
        result = 0
        for row in range(4):
            result |= ((board >> (4 * (row * 4 + col))) & 0xF) << (4 * row)
        return result

    def _set_col(self, board: int, col: int, value: int) -> int:
        for row in range(4):
            shift = 4 * (row * 4 + col)
            board = (board & ~(0xF << shift)) | (((value >> (4 * row)) & 0xF) << shift)
        return board

    def execute_move(self, board: int, direction: int) -> int:
        """Move board: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT."""
        result = board
        if direction == 3:  # LEFT
            for row in range(4):
                result = self._set_row(result, row, self.row_left_table[self._get_row(result, row)])
        elif direction == 1:  # RIGHT
            for row in range(4):
                result = self._set_row(result, row, self.row_right_table[self._get_row(result, row)])
        elif direction == 0:  # UP
            for col in range(4):
                result = self._set_col(result, col, self.row_left_table[self._get_col(result, col)])
        elif direction == 2:  # DOWN
            for col in range(4):
                result = self._set_col(result, col, self.row_right_table[self._get_col(result, col)])
        return result

    def score_board(self, board: int) -> float:
        score = 0.0
        for idx in range(4):
            score += self.row_score_table[self._get_row(board, idx)]
            score += self.row_score_table[self._get_col(board, idx)]
        score += self._score_corner_and_snake(board)
        return score

    def _board_ranks(self, board: int) -> List[int]:
        return [(board >> (4 * pos)) & 0xF for pos in range(self.BOARD_CELLS)]

    def _score_corner_and_snake(self, board: int) -> float:
        """
        Board-level positional heuristic that row/column tables cannot express.

        The fast row heuristic rewards edges, monotonicity, merges, and empty
        spaces, but it does not explicitly say "keep the largest tile in a
        corner". This bonus/penalty adds that preference while still allowing
        expectimax to choose a temporary escape move if every corner-preserving
        line is genuinely worse.
        """
        ranks = self._board_ranks(board)
        max_rank = max(ranks)
        if max_rank == 0:
            return 0.0

        corners = (0, 3, 12, 15)
        max_in_corner = any(ranks[pos] == max_rank for pos in corners)

        # Strongly prefer the largest tile in a corner. Penalize it heavily when
        # it drifts into an edge/interior cell, especially at high tile values.
        score = 0.0
        max_tile_weight = float(1 << max_rank)
        if max_in_corner:
            score += 4500.0 * max_tile_weight
        else:
            score -= 9000.0 * max_tile_weight

        # Reward snake layouts for all four corners and use the best orientation.
        # This avoids hard-coding top-left only, while still pushing high tiles to
        # form an ordered chain around whichever corner currently looks best.
        snake_paths = (
            (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12),   # TL
            (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15),   # TR
            (12, 13, 14, 15, 11, 10, 9, 8, 4, 5, 6, 7, 3, 2, 1, 0),   # BL
            (15, 14, 13, 12, 8, 9, 10, 11, 7, 6, 5, 4, 0, 1, 2, 3),   # BR
        )
        weights = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        best_snake_score = max(
            sum(ranks[pos] * weight for pos, weight in zip(path, weights))
            for path in snake_paths
        )
        score += 18000.0 * best_snake_score

        # Extra ordered-chain heuristic: the highest tile should be followed by
        # the second highest, then the third highest, etc. along the selected
        # snake path. A weighted snake score alone can still tolerate a scattered
        # second-largest tile, so this explicitly rewards adjacent descending
        # ranks near the corner and penalizes inversions where a larger tile sits
        # behind a smaller tile in the chain.
        best_chain_score = -float('inf')
        for path in snake_paths:
            sequence = [ranks[pos] for pos in path]
            chain_score = 0.0

            for idx in range(15):
                current_rank = sequence[idx]
                next_rank = sequence[idx + 1]
                if current_rank == 0 or next_rank == 0:
                    continue

                # Adjacent high tiles in descending/equal order are valuable,
                # especially near the front of the snake next to the max tile.
                front_weight = 16 - idx
                if current_rank >= next_rank:
                    chain_score += front_weight * (next_rank ** 2)
                    if current_rank - next_rank <= 1:
                        chain_score += front_weight * 8.0 * next_rank
                else:
                    # Inversion: a larger tile appears after a smaller tile,
                    # breaking the desired max -> second max -> third max chain.
                    chain_score -= front_weight * 35.0 * (next_rank - current_rank) * next_rank

            # Directly encourage the second-highest tile to sit very close to the
            # max-tile corner. This is the behavior you described: after the
            # highest tile, the next biggest should be beside/near it.
            if sequence[0] == max_rank:
                remaining = [rank for rank in sequence[1:] if rank > 0]
                if remaining:
                    second_rank = max(remaining)
                    second_index = sequence.index(second_rank, 1)
                    chain_score += max(0, 12 - second_index) * 250.0 * second_rank
                    if second_index == 1:
                        chain_score += 2500.0 * second_rank
                    elif second_index == 2:
                        chain_score += 1200.0 * second_rank

            best_chain_score = max(best_chain_score, chain_score)

        score += 9000.0 * best_chain_score

        return score

    def _empty_positions(self, board: int) -> List[int]:
        return [pos for pos in range(self.BOARD_CELLS) if ((board >> (4 * pos)) & 0xF) == 0]

    def _max_rank(self, board: int) -> int:
        max_rank = 0
        for pos in range(self.BOARD_CELLS):
            max_rank = max(max_rank, (board >> (4 * pos)) & 0xF)
        return max_rank

    def expectimax(self, board: int, depth: int, is_player_turn: bool, cache: dict, probability: float = 1.0) -> float:
        # Cut branches that are exceedingly unlikely, following the referenced
        # optimized brute-force approach.
        if depth <= 0 or probability < 0.0001:
            return self.score_board(board)

        key = (board, depth, is_player_turn)
        cached = cache.get(key)
        if cached is not None:
            return cached

        if is_player_turn:
            best_value = -float('inf')
            moved = False
            for direction in range(4):
                moved_board = self.execute_move(board, direction)
                if moved_board != board:
                    moved = True
                    best_value = max(best_value, self.expectimax(moved_board, depth - 1, False, cache, probability))
            value = best_value if moved else self.score_board(board)
        else:
            empty_positions = self._empty_positions(board)
            if not empty_positions:
                value = self.score_board(board)
            else:
                value = 0.0
                cell_probability = 1.0 / len(empty_positions)
                for pos in empty_positions:
                    shift = 4 * pos
                    # Standard 2048 spawn expectation: 90% tile 2, 10% tile 4.
                    for rank, tile_probability in ((1, 0.9), (2, 0.1)):
                        child = board | (rank << shift)
                        branch_probability = probability * cell_probability * tile_probability
                        value += cell_probability * tile_probability * self.expectimax(
                            child, depth - 1, True, cache, branch_probability
                        )

        cache[key] = value
        return value

    def choose_depth(self, board: int) -> int:
        empty_count = len(self._empty_positions(board))
        max_rank = self._max_rank(board)
        if empty_count >= 8:
            return 4
        if empty_count >= 5:
            return 5
        if max_rank >= 11:  # 2048+
            return 6
        return 5

    def find_best_move(self, grid: np.ndarray) -> Tuple[int, List[Tuple[str, Optional[float]]], int]:
        board = self.grid_to_board(grid)
        depth = self.choose_depth(board)
        cache = {}
        move_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        best_move = -1
        best_score = -float('inf')
        move_scores: List[Tuple[str, Optional[float]]] = []

        for direction, move_name in enumerate(move_names):
            moved_board = self.execute_move(board, direction)
            if moved_board == board:
                move_scores.append((move_name, None))
                continue
            score = self.expectimax(moved_board, depth - 1, False, cache)
            move_scores.append((move_name, score))
            if score > best_score:
                best_score = score
                best_move = direction

        return best_move, move_scores, depth


class Game2048AI:
    """AI player for 2048 game"""
    
    def __init__(self, x0: int, y0: int, x1: int, y1: int, next_tile_coords: Optional[Tuple[int, int, int, int]] = None, move_interval: float = 0.3, use_ocr: bool = False, debug: bool = False):
        """
        Initialize the AI
        x0, y0: top-left corner of game board on screen
        x1, y1: bottom-right corner of game board on screen
        move_interval: seconds between moves
        use_ocr: if True, use OCR to read tile numbers
        debug: if True, save captured screenshots for inspection
        """
        self.x0, self.y0 = x0, y0
        self.x1, self.y1 = x1, y1
        self.next_tile_coords = next_tile_coords
        self.move_interval = move_interval
        self.running = True
        self.board = Board2048()
        self.reader = BoardReader(use_ocr=use_ocr)
        self.move_count = 0
        self.debug = debug
        self.debug_folder = Path("capture_debug")
        if self.debug:
            self.debug_folder.mkdir(exist_ok=True)
    
    def save_debug_image(self, image: np.ndarray, name: str):
        """Save board debug images plus individual 1-16 tile crops."""
        if not self.debug:
            return
        try:
            path = self.debug_folder / name
            raw_path = self.debug_folder / name.replace("screenshot_", "raw_screenshot_", 1)

            raw_image = self.reader._normalize_capture_to_rgb(image)
            raw_pil_image = Image.fromarray(raw_image)
            raw_pil_image.save(raw_path)

            # Save each full tile crop beside the screenshot using the same base
            # filename plus _1 ... _16. Example:
            # screenshot_20260514_212500.png
            # screenshot_20260514_212500_1.png ... _16.png
            tile_boxes = self.reader.detector._get_tile_boxes(raw_pil_image)
            for row_idx, row_boxes in enumerate(tile_boxes):
                for col_idx, box in enumerate(row_boxes):
                    tile_number = row_idx * 4 + col_idx + 1
                    tile_path = path.with_name(f"{path.stem}_{tile_number}{path.suffix}")
                    raw_pil_image.crop(box).save(tile_path)

            processed_image = self.reader.preprocess_debug_image(image)
            processed_image.save(path)
            logger.info(f"Saved raw debug image: {raw_path}")
            logger.info(f"Saved processed debug image: {path}")
        except Exception as e:
            logger.warning(f"Failed to save debug image {name}: {e}")
    
    def capture_screen(self) -> np.ndarray:
        """Capture specified region of screen"""
        with mss.mss() as sct:
            monitor = {"top": self.y0, "left": self.x0, "width": self.x1 - self.x0, "height": self.y1 - self.y0}
            screenshot = sct.grab(monitor)
            screenshot_np = np.array(screenshot)
            if self.debug:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                self.save_debug_image(screenshot_np, f"screenshot_{timestamp}.png")
            return screenshot_np

    def capture_region(self, coords: Tuple[int, int, int, int]) -> np.ndarray:
        """Capture an arbitrary screen region."""
        x0, y0, x1, y1 = coords
        with mss.mss() as sct:
            monitor = {"top": y0, "left": x0, "width": x1 - x0, "height": y1 - y0}
            return np.array(sct.grab(monitor))

    def read_next_tile(self) -> int:
        """Read the calibrated next-tile preview as either 2 or 4."""
        if self.next_tile_coords is None:
            return self.ask_next_tile()

        image = self.capture_region(self.next_tile_coords)
        image = self.reader._normalize_capture_to_rgb(image)
        processed = self.reader.detector._preprocess_tile(Image.fromarray(image))

        if self.debug:
            self.debug_folder.mkdir(exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            Image.fromarray(image).save(self.debug_folder / f"next_tile_raw_{timestamp}.png")
            processed.save(self.debug_folder / f"next_tile_processed_{timestamp}.png")

        if not shutil.which("tesseract"):
            return self.ask_next_tile()

        configs = (
            '--psm 10 --oem 1 -c tessedit_char_whitelist=24',
            '--psm 8 --oem 1 -c tessedit_char_whitelist=24',
            '--psm 7 --oem 1 -c tessedit_char_whitelist=24',
        )

        for config in configs:
            text = pytesseract.image_to_string(processed, config=config).strip()
            digits = "".join(ch for ch in text if ch in "24")
            if digits:
                # The next tile preview should contain one digit. If OCR returns
                # extra characters, use the first valid 2/4 it saw.
                return int(digits[0])

        # Fallback to a lightweight shape/color heuristic. This keeps the loop
        # usable even if OCR occasionally fails on the tiny preview.
        gray = np.array(processed.convert("L"))
        black_pixels = gray < 128
        if np.any(black_pixels):
            ys, xs = np.where(black_pixels)
            digit = black_pixels[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            h, w = digit.shape
            # A 4 usually has more dark pixels in the right half because of the
            # vertical stem. A 2 is typically more balanced/curved.
            right_half = int(np.sum(digit[:, w // 2:]))
            left_half = int(np.sum(digit[:, :w // 2]))
            if right_half > left_half * 1.25:
                return 4

        return self.ask_next_tile()
    
    def extract_board_from_image(self, image: np.ndarray) -> Board2048:
        """
        Extract 2048 board state from screenshot
        This uses the board reader to detect tiles by color or OCR.
        """
        if image is None:
            return self.board

        save_csv_path = None
        if self.debug:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_csv_path = self.debug_folder / f"board_{timestamp}.csv"

        try:
            board_grid = self.reader.read_board(image, csv_path=save_csv_path)
            if board_grid is None:
                return self.board

            # If detection fails and returns only zeros, keep the previous board
            if np.sum(board_grid) == 0:
                logger.warning("Board extraction returned an empty board. Keeping previous board state.")
                return self.board

            return Board2048(board_grid)
        except Exception as e:
            logger.warning(f"Board extraction error: {e}")
            return self.board
    
    def expectimax(self, board: Board2048, depth: int, is_player_turn: bool, cache: dict, next_tile: int) -> float:
        """
        Expectimax search using the OCR-read next tile as a deterministic spawn.

        Player turns maximize across all legal moves. Chance turns average over
        every empty spawn location, but the tile value itself is fixed from the
        next-tile preview: if OCR reads 2, P(2)=1 and P(4)=0; if OCR reads 4,
        P(2)=0 and P(4)=1.
        """
        key = (tuple(int(value) for value in board.grid.flatten()), depth, is_player_turn, next_tile)
        if key in cache:
            return cache[key]

        if depth <= 0:
            value = board.evaluate()
            cache[key] = value
            return value

        if is_player_turn:
            best_value = -float('inf')
            for direction in range(4):
                next_board = Board2048(board.grid)
                if next_board.move(direction):
                    best_value = max(best_value, self.expectimax(next_board, depth - 1, False, cache, next_tile))
            value = best_value if best_value != -float('inf') else board.evaluate()
        else:
            empty_cells = board.get_empty_cells()
            if len(empty_cells) == 0:
                value = self.expectimax(board, depth - 1, True, cache, next_tile)
            else:
                total = 0.0
                probability_per_cell = 1.0 / len(empty_cells)
                for row, col in empty_cells:
                    next_board = Board2048(board.grid)
                    next_board.grid[row, col] = next_tile
                    total += probability_per_cell * self.expectimax(
                        next_board, depth - 1, True, cache, next_tile
                    )
                value = total

        cache[key] = value
        return value

    def _corner_for_max_tile(self, board: Board2048) -> Optional[Tuple[int, int]]:
        """Return the corner containing the current max tile, if any."""
        max_tile = board.get_max_tile()
        for corner in ((0, 0), (0, 3), (3, 0), (3, 3)):
            if board.grid[corner] == max_tile:
                return corner
        return None

    def _orient_to_corner(self, grid: np.ndarray, corner: Tuple[int, int]) -> np.ndarray:
        """
        Mirror the board so the chosen corner becomes top-left.

        This lets one monotonic/snake heuristic support all four possible max-tile
        corners without duplicating logic.
        """
        oriented = grid
        row, col = corner
        if row == 3:
            oriented = oriented[::-1, :]
        if col == 3:
            oriented = oriented[:, ::-1]
        return oriented

    def _calculate_merge_stats(self, before: np.ndarray, direction: int) -> Tuple[int, int]:
        """
        Calculate total and largest merge produced by one move.

        The normal board sum does not reveal merge quality because 512+512 and
        two separate 512 tiles both sum to 1024. This method inspects the moved
        lines directly and rewards the actual merged tile values.
        """
        if direction == 0:      # UP: read columns top-to-bottom
            lines = [before[:, col] for col in range(4)]
        elif direction == 1:    # RIGHT: read rows right-to-left
            lines = [before[row, ::-1] for row in range(4)]
        elif direction == 2:    # DOWN: read columns bottom-to-top
            lines = [before[::-1, col] for col in range(4)]
        else:                   # LEFT: read rows left-to-right
            lines = [before[row, :] for row in range(4)]

        total_merge = 0
        largest_merge = 0
        for line in lines:
            values = [int(value) for value in line if value > 0]
            idx = 0
            while idx + 1 < len(values):
                if values[idx] == values[idx + 1]:
                    merged_value = values[idx] * 2
                    total_merge += merged_value
                    largest_merge = max(largest_merge, merged_value)
                    idx += 2
                else:
                    idx += 1

        return total_merge, largest_merge

    def _corner_structure_score(self, board: Board2048, corner: Tuple[int, int]) -> float:
        """
        Score how cleanly the board is arranged around the chosen max-tile corner.

        Higher values reward a stable snake shape where large tiles stay close to
        the corner and values generally decrease away from it.
        """
        oriented = self._orient_to_corner(board.grid.astype(np.float64), corner)
        log_grid = np.zeros_like(oriented, dtype=np.float64)
        nonzero = oriented > 0
        log_grid[nonzero] = np.log2(oriented[nonzero])

        snake_weights = np.array([
            [15, 14, 13, 12],
            [8, 9, 10, 11],
            [7, 6, 5, 4],
            [0, 1, 2, 3],
        ], dtype=np.float64)
        snake_score = float(np.sum(log_grid * snake_weights))

        monotonic_bonus = 0.0
        inversion_penalty = 0.0
        for row in range(4):
            for col in range(3):
                diff = log_grid[row, col] - log_grid[row, col + 1]
                monotonic_bonus += diff
                inversion_penalty += max(-diff, 0.0)
        for col in range(4):
            for row in range(3):
                diff = log_grid[row, col] - log_grid[row + 1, col]
                monotonic_bonus += diff
                inversion_penalty += max(-diff, 0.0)

        smoothness_penalty = 0.0
        for row in range(4):
            for col in range(3):
                if log_grid[row, col] and log_grid[row, col + 1]:
                    smoothness_penalty += abs(log_grid[row, col] - log_grid[row, col + 1])
                if log_grid[col, row] and log_grid[col + 1, row]:
                    smoothness_penalty += abs(log_grid[col, row] - log_grid[col + 1, row])

        return (
            350.0 * snake_score +
            900.0 * monotonic_bonus -
            3500.0 * inversion_penalty -
            300.0 * smoothness_penalty
        )

    def _greedy_corner_score(
        self,
        before: Board2048,
        after: Board2048,
        direction: int,
        locked_corner: Optional[Tuple[int, int]],
    ) -> float:
        """Score one immediate move using max-tile-in-corner greedy rules."""
        current_max = before.get_max_tile()
        after_max = after.get_max_tile()
        after_corner = self._corner_for_max_tile(after)
        total_merge, largest_merge = self._calculate_merge_stats(before.grid, direction)
        empty_before = len(before.get_empty_cells())
        empty_after = len(after.get_empty_cells())

        score = 0.0

        # Rule 1 & 2: once the largest tile is in a corner, treat that corner as
        # locked. Moving it out is almost never worth it, so apply a huge penalty.
        if locked_corner is not None:
            if after.grid[locked_corner] != current_max:
                return -1_000_000_000.0
            active_corner = locked_corner
            score += 2_000_000.0
        else:
            # If the max tile is not in a corner yet, strongly prefer moves that
            # place/keep it in any corner, then use that corner for structure.
            active_corner = after_corner if after_corner is not None else (0, 0)
            if after_corner is not None and after_max == current_max:
                score += 1_000_000.0
            else:
                score -= 250_000.0

        # Rule 3: greedy merge priority. Largest merge matters most, then total
        # merge value. Empty-cell gain is a secondary signal that a merge happened.
        score += 5000.0 * np.log2(largest_merge) if largest_merge > 0 else 0.0
        score += 1200.0 * np.log2(total_merge) if total_merge > 0 else 0.0
        score += 8000.0 * (empty_after - empty_before)
        score += 5000.0 * empty_after

        # Rule 4: keep numbers tidy around the corner with snake/monotonic order.
        score += self._corner_structure_score(after, active_corner)

        # Small preference against throwing the second-largest tile far from the
        # locked corner, helping the board build around the max tile.
        oriented = self._orient_to_corner(after.grid.astype(np.float64), active_corner)
        top_area = oriented[:2, :].sum()
        bottom_area = oriented[2:, :].sum()
        score += 0.05 * top_area - 0.05 * bottom_area

        return score
    
    def find_best_move(self, next_tile: int) -> int:
        """Find the best move using expectimax and the OCR-read next tile."""
        if next_tile not in (2, 4):
            raise ValueError(f"next_tile must be 2 or 4, got {next_tile!r}")

        best_move = -1
        best_score = -float('inf')
        empty_count = len(self.board.get_empty_cells())
        search_depth = 4 if empty_count >= 6 else 5
        cache = {}
        move_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        move_scores = []

        for direction, move_name in enumerate(move_names):
            test_board = Board2048(self.board.grid)
            if test_board.move(direction):
                score = self.expectimax(
                    test_board,
                    depth=search_depth - 1,
                    is_player_turn=False,
                    cache=cache,
                    next_tile=next_tile,
                )
                move_scores.append((move_name, score))
                if score > best_score:
                    best_score = score
                    best_move = direction
            else:
                move_scores.append((move_name, None))

        print(f"EXPECTIMAX DEPTH: {search_depth} | NEXT TILE OCR: {next_tile}")
        print("EXPECTIMAX SCORES:")
        for move_name, score in move_scores:
            if score is None:
                print(f"  {move_name}: invalid/no board movement")
            else:
                print(f"  {move_name}: {score:.2f}")
        
        return best_move
    
    def print_board_csv(self):
        """Print the current board as CSV rows."""
        print("BOARD CSV:")
        for row in self.board.grid:
            print(",".join(str(int(value)) for value in row))

    def print_best_move(self, direction: int):
        """Print only the selected best move."""
        keys = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        if direction == -1:
            print("BEST MOVE: NONE")
            print("")
            return

        print(f"BEST MOVE: {keys[direction]}")
        print("")

    def execute_move(self, direction: int):
        """Execute a move by pressing the matching keyboard arrow key."""
        keys = ['up', 'right', 'down', 'left']
        if direction < 0 or direction >= len(keys):
            logger.warning(f"Invalid move direction: {direction}")
            return

        pyautogui.press(keys[direction])
        self.move_count += 1
        print(f"PERFORMED MOVE: {keys[direction].upper()}")

        if self.move_count % 20 == 0:
            pyautogui.click()
            print("MOUSE CLICK: current position")

        time.sleep(0.5)

    def ask_next_tile(self) -> int:
        """Ask the user which tile will spawn next: 2 or 4."""
        while True:
            answer = input("NEXT TILE (2/4): ").strip()
            if answer in ("2", "4"):
                return int(answer)
            print("Please enter 2 or 4.")
    
    def run(self):
        """Main game loop"""
        try:
            while self.running:
                next_tile = self.read_next_tile()
                print(f"NEXT TILE: {next_tile}")

                # Capture and update board
                screenshot = self.capture_screen()
                self.board = self.extract_board_from_image(screenshot)
                self.print_board_csv()
                
                # Find and execute best move using keyboard arrow keys.
                best_move = self.find_best_move(next_tile)
                self.print_best_move(best_move)
                if best_move == -1:
                    print("NO VALID MOVES LEFT. Stopping without pressing any key.")
                    self.running = False
                    break

                self.execute_move(best_move)
                
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.error(f"Error: {e}")


# ==== CONFIGURATION ====
# You need to find your game board coordinates
# Run this script with --calibrate flag to help find coordinates

def calibrate():
    """Helper to find board coordinates"""
    logger.info("Calibration mode: Move your mouse to game board corners and note the coordinates")
    logger.info("To find coordinates, run: python -c \"import pyautogui; print(pyautogui.position())\"")
    logger.info("Or just hover your mouse and check VS Code's debug output")
    
    print("\n1. Move your mouse to TOP-LEFT corner of the 2048 board and note (x, y)")
    print("2. Move your mouse to BOTTOM-RIGHT corner of the 2048 board and note (x, y)")
    print("\nThen update the coordinates below in the __main__ section")
    

def load_coordinates() -> Optional[Tuple[int, int, int, int]]:
    """Load saved board coordinates from board_coordinates.txt."""
    return load_region_coordinates("board_coordinates.txt")


if __name__ == "__main__":
    import sys
    
    use_ocr = "--ocr" in sys.argv
    debug = "--debug" in sys.argv

    if len(sys.argv) > 1 and sys.argv[1] == "--calibrate":
        calibrate()
    else:
        coords = load_coordinates()
        if coords:
            x0, y0, x1, y1 = coords
        else:
            logger.warning("No valid saved coordinates found in board_coordinates.txt. Using hardcoded defaults.")
            # ===== CUSTOMIZE THESE COORDINATES =====
            # x0, y0 = top-left corner of game board
            # x1, y1 = bottom-right corner of game board
            
            x0, y0 = 100, 100      # START HERE: Update these coordinates
            x1, y1 = 500, 500      # END HERE: Update these coordinates

        next_tile_coords = load_region_coordinates("next_tile_coordinates.txt")
        bot = Game2048AI(x0, y0, x1, y1, next_tile_coords=next_tile_coords, move_interval=0.3, use_ocr=use_ocr, debug=debug)
        bot.run()
