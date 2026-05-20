import pyautogui
import sys


def _capture_region(label: str):
    """Capture top-left and bottom-right coordinates for a screen region."""
    print("\n" + "="*60)
    print(f"{label.upper()} CALIBRATION")
    print("="*60)
    
    print(f"\n📍 STEP 1: Find TOP-LEFT corner of {label}")
    print(f"   Move your mouse to the TOP-LEFT corner of {label}")
    print("   Once positioned, press ENTER to capture...")
    input()
    x0, y0 = pyautogui.position()
    print(f"   ✓ Captured: ({x0}, {y0})")
    
    print(f"\n📍 STEP 2: Find BOTTOM-RIGHT corner of {label}")
    print(f"   Move your mouse to the BOTTOM-RIGHT corner of {label}")
    print("   Once positioned, press ENTER to capture...")
    input()
    x1, y1 = pyautogui.position()
    print(f"   ✓ Captured: ({x1}, {y1})")

    return x0, y0, x1, y1


def calibrate_coordinates():
    """Interactive calibration to find game board coordinates"""
    try:
        x0, y0, x1, y1 = _capture_region("2048 game board")
        
        print("\n" + "="*60)
        print("✅ CALIBRATION COMPLETE!")
        print("="*60)
        print(f"\nAdd these coordinates to test.py:")
        print(f"\n    x0, y0 = {x0}, {y0}      # TOP-LEFT")
        print(f"    x1, y1 = {x1}, {y1}      # BOTTOM-RIGHT")
        
        width = x1 - x0
        height = y1 - y0
        print(f"\nBoard size: {width} x {height} pixels")
        
        with open('board_coordinates.txt', 'w') as f:
            f.write(f"{x0},{y0},{x1},{y1}\n")
        
        print(f"\n💾 Coordinates saved to: board_coordinates.txt")
        print("\n✅ You can now run: python test.py")
    except Exception as e:
        print(f"   ✗ Calibration failed: {e}")


def calibrate_next_tile_coordinates():
    """Interactive calibration to find the next-tile preview coordinates."""
    try:
        x0, y0, x1, y1 = _capture_region("next tile preview")
        
        print("\n" + "="*60)
        print("✅ NEXT TILE CALIBRATION COMPLETE!")
        print("="*60)
        
        width = x1 - x0
        height = y1 - y0
        print(f"\nNext tile preview size: {width} x {height} pixels")
        
        with open('next_tile_coordinates.txt', 'w') as f:
            f.write(f"{x0},{y0},{x1},{y1}\n")
        
        print(f"\n💾 Coordinates saved to: next_tile_coordinates.txt")
        print("\n✅ You can now run: python test.py --ocr")
    except Exception as e:
        print(f"   ✗ Next tile calibration failed: {e}")


if __name__ == "__main__":
    if "--next-tile" in sys.argv:
        calibrate_next_tile_coordinates()
    else:
        calibrate_coordinates()
