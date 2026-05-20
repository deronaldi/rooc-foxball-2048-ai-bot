"""
Advanced Board Reader for 2048
Detects tile values from game screenshots using OCR and color detection
"""

import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytesseract
from typing import Tuple, List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class TileDetector:
    """Detects 2048 tiles from screenshot"""
    
    # Color ranges for classic 2048 tiles in RGB order.
    # Important: BoardReader normalizes screenshots to RGB before detection.
    # Customize these based on your game's actual colors
    # TILE_COLORS = {
    #     0: ((205, 193, 180), (220, 210, 200)),   # empty
    #     2: ((238, 228, 218), (245, 235, 225)),
    #     4: ((237, 224, 200), (242, 229, 206)),
    #     8: ((242, 177, 121), (246, 189, 135)),
    #     16: ((245, 149, 99), (249, 161, 110)),
    #     32: ((246, 124, 95), (250, 135, 105)),
    #     64: ((246, 94, 59), (250, 110, 75)),
    #     128: ((237, 207, 114), (242, 214, 125)),
    #     256: ((237, 204, 97), (242, 210, 110)),
    #     512: ((237, 200, 80), (242, 206, 95)),
    #     1024: ((237, 197, 63), (242, 203, 80)),
    #     2048: ((237, 194, 46), (242, 200, 65)),
    # }

    TILE_COLORS = {
        0: ((160, 95, 80), (190, 130, 115)),       # empty
        2: ((220, 200, 185), (245, 230, 215)),
        4: ((225, 195, 170), (250, 220, 195)),
        8: ((235, 180, 145), (255, 210, 180)),
        16: ((235, 190, 130), (255, 220, 165)),
        32: ((235, 125, 100), (255, 155, 125)),
        64: ((245, 190, 105), (255, 220, 150)),
        128: ((245, 170, 80), (255, 205, 130)),
        256: ((245, 160, 70), (255, 195, 120)),
        512: ((245, 150, 65), (255, 185, 115)),
        1024: ((245, 140, 80), (255, 175, 125)),
        2048: ((245, 130, 100), (255, 165, 145)),
        4096: ((235, 120, 120), (255, 155, 155)),
    }

    def _get_background_color(self, cell: np.ndarray) -> np.ndarray:
        h, w = cell.shape[:2]

        samples = [
            cell[int(h * 0.20), int(w * 0.20)],
            cell[int(h * 0.20), int(w * 0.80)],
            cell[int(h * 0.80), int(w * 0.20)],
            cell[int(h * 0.80), int(w * 0.80)],
        ]

        return np.mean(samples, axis=0)

    def _crop_digit_area(self, cell_image: Image.Image) -> Image.Image:
        w, h = cell_image.size
        # return cell_image.crop((
        #     int(w * 0.10),
        #     int(h * 0.12),
        #     int(w * 0.90),
        #     int(h * 0.88),
        # ))
        # return cell_image.crop((
        #     int(w * 0.005),
        #     int(h * 0.22),
        #     int(w * 0.995),
        #     int(h * 0.78),
        # ))
        return cell_image.crop((
            int(w * 0.05),
            int(h * 0.25),
            int(w * 0.985),
            int(h * 0.75),
        ))

    
    def __init__(self, use_ocr: bool = False):
        """
        Initialize detector
        use_ocr: If True, use OCR to read numbers (slower but more accurate)
        """
        self.use_ocr = use_ocr
        # Fast path for the Ragnarok-style board: once a tile value has been
        # read by OCR, keep a normalized image template for that value. Future
        # frames can then classify the same-looking digits with cheap in-memory
        # image comparison instead of launching Tesseract 16 times per move.
        self.templates: Dict[int, List[np.ndarray]] = {}
        self.template_size = (64, 64)
        self.template_match_threshold = 0.86
        self.strong_template_match_threshold = 0.94
        self.tesseract_available = shutil.which("tesseract") is not None

        if self.use_ocr and not self.tesseract_available:
            logger.warning(
                "OCR mode is enabled, but tesseract.exe was not found in PATH. "
                "Install Tesseract OCR and add it to PATH, or set "
                "pytesseract.pytesseract.tesseract_cmd to the full tesseract.exe path."
            )
    
    def detect_tiles_by_color(self, image: np.ndarray, tile_size: Optional[int] = None) -> np.ndarray:
        """
        Detect tiles by color matching
        Faster but less accurate if colors vary
        """
        board = np.zeros((4, 4), dtype=np.int32)
        height, width = image.shape[:2]
        
        cell_height = height // 4
        cell_width = width // 4
        
        for row in range(4):
            for col in range(4):
                # Extract cell region
                y1 = row * cell_height
                y2 = (row + 1) * cell_height
                x1 = col * cell_width
                x2 = (col + 1) * cell_width
                
                cell = image[y1:y2, x1:x2]
                
                # Find dominant color and match to tile value
                board[row, col] = self._match_color_to_tile(cell)
        
        return board
    
    def _match_color_to_tile(self, cell: np.ndarray) -> int:
        """Match tile background color to tile value."""
        bg_color = self._get_background_color(cell)

        best_tile = 0
        best_distance = float("inf")

        for tile_value, (lower, upper) in self.TILE_COLORS.items():
            lower = np.array(lower, dtype=np.float32)
            upper = np.array(upper, dtype=np.float32)

            center_color = (lower + upper) / 2.0
            distance = np.linalg.norm(bg_color.astype(np.float32) - center_color)

            if distance < best_distance:
                best_distance = distance
                best_tile = tile_value

        if best_distance > 45:
            return 0

        return best_tile
    
    def _get_tile_boxes(self, image: Image.Image) -> List[List[Tuple[int, int, int, int]]]:
        """Split the board capture into 16 full 4x4 tile boxes."""
        width, height = image.size
        cell_width = width // 4
        cell_height = height // 4

        boxes: List[List[Tuple[int, int, int, int]]] = []

        for row in range(4):
            row_boxes = []
            for col in range(4):
                x1 = col * cell_width
                y1 = row * cell_height
                # Let the last row/column absorb any remainder pixels so no part
                # of the board is lost when width/height are not divisible by 4.
                x2 = width if col == 3 else (col + 1) * cell_width
                y2 = height if row == 3 else (row + 1) * cell_height

                row_boxes.append((x1, y1, x2, y2))
            boxes.append(row_boxes)

        return boxes

    def _prepare_cell_for_reading(self, cell_image: Image.Image) -> Image.Image:
        """Crop inside one full tile box to isolate the number before OCR/template matching."""
        return self._crop_digit_area(cell_image)

    def _preprocess_tile(self, tile: Image.Image) -> Image.Image:
        arr = np.array(tile)
        img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        # angka putih: tile 8 ke atas
        white_mask = cv2.inRange(
            img,
            np.array([220, 220, 220]),
            np.array([255, 255, 255])
        )

        # angka coklat/gelap: tile 2 dan 4
        dark_mask = cv2.inRange(
            img,
            np.array([60, 70, 70]),      # BGR lower
            np.array([140, 140, 150])    # BGR upper
        )

        mask = cv2.bitwise_or(white_mask, dark_mask)

        # Jangan terlalu agresif, supaya digit 2/4 tidak hilang
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Remove grid/square outline artifacts. Real digits are inside the tile;
        # outlines usually touch the crop border or form very long horizontal /
        # vertical components. Filtering here keeps the OCR/debug image as a
        # clean white background with only number strokes in black.
        # mask = self._remove_outline_artifacts(mask)
        mask = self._keep_digit_components(mask)

        # Tipiskan sedikit agar 4 tidak jadi blob terlalu tebal.
        mask = cv2.erode(mask, np.ones((2, 2), np.uint8), iterations=1)

        result = np.full(mask.shape, 255, dtype=np.uint8)
        result[mask > 0] = 0

        result = cv2.resize(result, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

        return Image.fromarray(result)

    def _keep_digit_components(self, mask: np.ndarray) -> np.ndarray:
        cleaned = np.zeros_like(mask)
        height, width = mask.shape[:2]

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        min_area = width * height * 0.01

        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            cx, cy = centroids[label]

            too_small = area < min_area
            far_from_center = (
                cx < width * 0.15 or cx > width * 0.85 or
                cy < height * 0.10 or cy > height * 0.90
            )

            if too_small and far_from_center:
                continue

            cleaned[labels == label] = 255

        return cleaned

    def _remove_outline_artifacts(self, mask: np.ndarray) -> np.ndarray:
        """Remove connected components that look like tile/grid borders."""
        cleaned = mask.copy()
        height, width = cleaned.shape[:2]
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

        border_margin = max(2, min(width, height) // 40)
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            touches_border = (
                x <= border_margin or
                y <= border_margin or
                x + w >= width - border_margin or
                y + h >= height - border_margin
            )
            long_horizontal_line = w >= width * 0.65 and h <= height * 0.18
            long_vertical_line = h >= height * 0.65 and w <= width * 0.18

            if touches_border or long_horizontal_line or long_vertical_line:
                cleaned[labels == label] = 0

        return cleaned

    def _normalize_digit_template(self, processed_tile: Image.Image) -> Optional[np.ndarray]:
        """Return a centered, fixed-size binary digit image for template matching."""
        arr = np.array(processed_tile.convert("L"))
        foreground = arr < 128

        # Ignore empty/noisy cells. The threshold is intentionally small because
        # a single digit can be thin after preprocessing, but true empty cells
        # should have almost no black pixels.
        if int(np.sum(foreground)) < 80:
            return None

        ys, xs = np.where(foreground)
        if len(xs) == 0 or len(ys) == 0:
            return None

        pad = 12
        x1 = max(int(xs.min()) - pad, 0)
        x2 = min(int(xs.max()) + pad + 1, arr.shape[1])
        y1 = max(int(ys.min()) - pad, 0)
        y2 = min(int(ys.max()) + pad + 1, arr.shape[0])

        digit = foreground[y1:y2, x1:x2].astype(np.uint8) * 255
        if digit.size == 0:
            return None

        normalized = cv2.resize(digit, self.template_size, interpolation=cv2.INTER_AREA)
        normalized = (normalized > 127).astype(np.float32)
        return normalized

    def _match_template(self, candidate: Optional[np.ndarray]) -> Tuple[int, float]:
        """Match a normalized candidate against learned templates."""
        if candidate is None or not self.templates:
            return 0, 0.0

        best_value = 0
        best_score = 0.0
        for value, templates in self.templates.items():
            for template in templates:
                # Similarity in [0, 1], where 1 means identical binary shapes.
                score = 1.0 - float(np.mean(np.abs(candidate - template)))
                if score > best_score:
                    best_score = score
                    best_value = value

        if best_score >= self.template_match_threshold:
            return best_value, best_score
        return 0, best_score

    def _remember_template(self, value: int, candidate: Optional[np.ndarray]):
        """Store a template for a confirmed tile value, avoiding duplicates."""
        if candidate is None or not self._is_valid_2048_tile(value):
            return

        templates = self.templates.setdefault(value, [])
        for template in templates:
            score = 1.0 - float(np.mean(np.abs(candidate - template)))
            if score >= 0.98:
                return

        # Keep only a few variants per value to avoid unbounded growth while
        # still allowing slight anti-aliasing/crop differences.
        if len(templates) < 4:
            templates.append(candidate)

    def _ocr_processed_tile(self, processed_tile: Image.Image) -> int:
        configs = (
            '--dpi 300 --psm 10 --oem 1 -c tessedit_char_whitelist=0123456789 -c classify_bln_numeric_mode=1',
            '--dpi 300 --psm 8 --oem 1 -c tessedit_char_whitelist=0123456789 -c classify_bln_numeric_mode=1',
            '--dpi 300 --psm 7 --oem 1 -c tessedit_char_whitelist=0123456789 -c classify_bln_numeric_mode=1',
        )

        for config in configs:
            raw = pytesseract.image_to_string(processed_tile, config=config)
            text = "".join(ch for ch in raw if ch.isdigit())

            logger.info(f"OCR raw={raw!r}, digits={text!r}, config={config}")

            if not text:
                continue

            value = int(text)

            if self._is_valid_2048_tile(value):
                return value

            logger.info(f"OCR produced invalid 2048 tile value: {value}")

        return 0

    def _is_probably_empty_tile(self, candidate: Optional[np.ndarray]) -> bool:
        """Return True when preprocessing found no meaningful digit shape."""
        return candidate is None

    def _save_csv(self, rows: List[List[str]], csv_path: Path):
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def _save_debug_tiles(self, boxes: List[List[Tuple[int, int, int, int]]], pil_image: Image.Image, debug_path: Optional[Path] = None):
        """Save each of the 16 split tile boxes plus OCR-ready crops for debugging."""
        if debug_path is None:
            return
        
        debug_dir = debug_path.parent / "tiles"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        for row_idx, row_boxes in enumerate(boxes):
            for col_idx, box in enumerate(row_boxes):
                box_number = row_idx * 4 + col_idx + 1
                full_cell_image = pil_image.crop(box)
                digit_image = self._prepare_cell_for_reading(full_cell_image)
                processed_tile = self._preprocess_tile(digit_image)

                full_cell_image.save(debug_dir / f"box_{box_number:02d}_r{row_idx}_c{col_idx}_full.png")
                digit_image.save(debug_dir / f"box_{box_number:02d}_r{row_idx}_c{col_idx}_digit.png")
                processed_tile.save(debug_dir / f"box_{box_number:02d}_r{row_idx}_c{col_idx}_processed.png")

                # Keep the old names too, so existing workflows/scripts that look
                # for cell_r*_c* or tile_r*_c* continue to work.
                full_cell_image.save(debug_dir / f"cell_r{row_idx}_c{col_idx}.png")
                processed_tile.save(debug_dir / f"tile_r{row_idx}_c{col_idx}.png")

    def _is_valid_2048_tile(self, value: int) -> bool:
        """Check if value is a valid 2048 tile from 2 up to 131072."""
        return value >= 2 and value <= 131072 and (value & (value - 1)) == 0

    def detect_tiles_by_ocr(self, image: np.ndarray, save_csv_path: Optional[Path] = None) -> np.ndarray:
        """
        Detect tiles using OCR (text recognition)
        More accurate but slower
        """
        board = np.zeros((4, 4), dtype=np.int32)

        if not self.tesseract_available:
            logger.warning(
                "Cannot read tile numbers: Tesseract OCR engine is not installed or not in PATH. "
                "Returning an empty board."
            )
            return board

        # image must be RGB here. BoardReader.read_board() normalizes mss BGRA
        # captures before calling this OCR path.
        pil_image = Image.fromarray(image)

        # Step 1: divide the board image into 16 boxes. Step 2 below reads the
        # number inside each individual box.
        boxes = self._get_tile_boxes(pil_image)
        csv_rows: List[List[str]] = []

        if save_csv_path is not None:
            self._save_debug_tiles(boxes, pil_image, debug_path=save_csv_path)

        for row_idx, row_boxes in enumerate(boxes):
            row_values: List[str] = []
            for col_idx, box in enumerate(row_boxes):
                full_cell_image = pil_image.crop(box)
                digit_image = self._prepare_cell_for_reading(full_cell_image)
                processed_tile = self._preprocess_tile(digit_image)
                candidate = self._normalize_digit_template(processed_tile)

                value = 0
                if not self._is_probably_empty_tile(candidate):
                    # Fast path: use learned templates when the match is strong.
                    # This avoids launching Tesseract for already-known digits.
                    template_value, score = self._match_template(candidate)
                    if template_value and score >= self.strong_template_match_threshold:
                        value = template_value
                    else:
                        try:
                            value = self._ocr_processed_tile(processed_tile)
                        except Exception as e:
                            logger.warning(f"OCR failed for box {row_idx * 4 + col_idx + 1} ({row_idx}, {col_idx}): {e}")

                        # If OCR could not read it, fall back to normal template
                        # threshold. This keeps speed after templates are learned,
                        # while avoiding weak template matches before OCR runs.
                        if not value:
                            value = template_value

                if value:
                    board[row_idx, col_idx] = value
                    row_values.append(str(value))
                    self._remember_template(value, candidate)
                else:
                    logger.debug(f"Could not classify box {row_idx * 4 + col_idx + 1} ({row_idx}, {col_idx}). Setting to 0.")
                    row_values.append("")

            csv_rows.append(row_values)

        if save_csv_path is not None:
            self._save_csv(csv_rows, save_csv_path)

        return board
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """Detect board from image"""
        if self.use_ocr:
            return self.detect_tiles_by_ocr(image)
        else:
            return self.detect_tiles_by_color(image)


class BoardReader:
    """Reads full board state from screenshot"""
    
    def __init__(self, use_ocr: bool = False):
        self.detector = TileDetector(use_ocr=use_ocr)

    def _normalize_capture_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize screen captures to RGB.

        mss returns BGRA screenshots, while PIL/numpy screenshots are often RGB.
        The detector and OCR preprocessing both expect RGB. For 4-channel images
        we can safely identify mss BGRA and convert it. For 3-channel images we
        keep the data unchanged because blindly applying BGR->RGB will corrupt
        screenshots that are already RGB.
        """
        if len(image.shape) == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        return image
    
    def preprocess_debug_image(self, image: np.ndarray) -> Image.Image:
        """Return a clean black-on-white board image containing numbers only."""
        image = self._normalize_capture_to_rgb(image)
        pil_image = Image.fromarray(image)
        boxes = self.detector._get_tile_boxes(pil_image)
        clean_board = Image.new("L", pil_image.size, 255)

        for row_boxes in boxes:
            for box in row_boxes:
                full_cell_image = pil_image.crop(box)
                digit_image = self.detector._prepare_cell_for_reading(full_cell_image)
                processed_tile = self.detector._preprocess_tile(digit_image)

                # Paste the processed digit crop back into the same relative
                # position inside a white tile-sized canvas. This prevents the
                # board grid and tile outlines from ever appearing in the debug
                # capture while preserving where each number belongs.
                cell_width = box[2] - box[0]
                cell_height = box[3] - box[1]
                digit_width, digit_height = digit_image.size
                processed_tile = processed_tile.resize((digit_width, digit_height), Image.Resampling.LANCZOS)

                tile_canvas = Image.new("L", (cell_width, cell_height), 255)
                offset_x = int(cell_width * 0.04)
                offset_y = int(cell_height * 0.20)
                tile_canvas.paste(processed_tile, (offset_x, offset_y))
                clean_board.paste(tile_canvas, (box[0], box[1]))

        return clean_board
    
    def read_board(self, image: np.ndarray, csv_path: Optional[Path] = None) -> np.ndarray:
        """Read board from screenshot"""
        image = self._normalize_capture_to_rgb(image)

        if self.detector.use_ocr:
            return self.detector.detect_tiles_by_ocr(image, save_csv_path=csv_path)

        # Color detection uses the RGB ranges defined in TileDetector.TILE_COLORS.
        board = self.detector.detect(image)
        return board


def test_board_reader():
    """Test the board reader"""
    # This would require actual 2048 game screenshots to test properly
    logger.info("BoardReader module loaded. Use with Game2048AI for automatic board detection.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_board_reader()
