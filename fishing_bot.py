#!/usr/bin/env python3
"""
NTE Fishing Auto-Player v6 (Linux / Zorin OS)
===============================================
Handles the full fishing loop:
  Phase 1 — IDLE:     No bar, no glow → wait (--autocast to press F)
  Phase 2 — HOOKED:   Blue glow on F button → press F to reel in
  Phase 3 — MINIGAME: Cursor (thin vertical line) + colored zone → A/D tracking

KEY FIXES in v6:
  - Fixed BGR channel order (mss returns BGRA, not RGBA)
  - Rewrote cursor detection: detects thin vertical line via column projection
  - Rewrote scan approach: green zone found per-row, cursor found via vertical scan
  - Bar region refreshed every N frames to track movement

NO injection — purely screen-reading + xdotool keypresses.

Install:
    sudo apt install xdotool
    python3 -m venv ~/fishbot-venv
    ~/fishbot-venv/bin/pip install pillow mss numpy

Usage:
    ~/fishbot-venv/bin/python fishing_bot.py --calibrate --monitor 2
    ~/fishbot-venv/bin/python fishing_bot.py --monitor 2
    ~/fishbot-venv/bin/python fishing_bot.py --autocast --monitor 2
"""

import argparse
import time
import subprocess
import sys
import os
import signal

try:
    import mss
    import numpy as np
    from PIL import Image
except ImportError:
    print("Missing dependencies! Install with:")
    print("  sudo apt install xdotool")
    print("  pip install pillow mss numpy")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION — all colors specified as (R, G, B) but accessed via
#  BGR indices since mss returns BGRA format.
# ═══════════════════════════════════════════════════════════════════════

# BGR channel indices for mss captures
B_CH = 0
G_CH = 1
R_CH = 2

# Yellow/white cursor (thin vertical line) — high brightness, warm color
# The cursor line is typically bright yellow/white/orange
CURSOR_BRIGHT_MIN = 200     # minimum brightness (max of R,G channels)
CURSOR_MIN_ROWS = 5         # minimum rows the vertical line must span
CURSOR_MAX_WIDTH = 8        # max width of cursor column cluster

# Green/colored target zone thresholds (the zone we need to stay in)
# Adjusted for BGR order
TARGET_G_MIN = 150          # green channel minimum
TARGET_R_MAX = 140          # red channel maximum (in BGR: channel 2)
TARGET_B_MAX = 140          # blue channel maximum (in BGR: channel 0)

# Minimum width (in pixels) for a target zone cluster to be real
MIN_TARGET_CLUSTER = 30

# F-button glow detection (blue glow)
GLOW_B_MIN = 200
GLOW_R_MAX = 150
GLOW_G_MIN = 140
GLOW_THRESHOLD = 200

LOOP_DELAY = 0.015
IDLE_DELAY = 0.15
KEY_HOLD_MS = 30
DEADZONE = 8
CAST_COOLDOWN = 3.0

# How often to re-detect bar region (every N frames)
BAR_REFRESH_INTERVAL = 60

# ═══════════════════════════════════════════════════════════════════════

running = True


def signal_handler(sig, frame):
    global running
    print("\n[!] Stopping...")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def press_key(key: str):
    subprocess.Popen(
        ["xdotool", "key", "--delay", str(KEY_HOLD_MS), key],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


# ── Color helpers (BGR order) ─────────────────────────────────────────

def is_target_green(pixel_row: np.ndarray) -> np.ndarray:
    """Return boolean mask for pixels that match the target zone color. Input is BGR."""
    return (
        (pixel_row[:, G_CH] > TARGET_G_MIN) &
        (pixel_row[:, R_CH] < TARGET_R_MAX) &
        (pixel_row[:, B_CH] < TARGET_B_MAX)
    )


def is_bright_cursor(pixel_row: np.ndarray) -> np.ndarray:
    """
    Return boolean mask for pixels that could be the cursor line.
    The cursor is typically a bright vertical line (white, yellow, or bright orange).
    In BGR: high values in G and R channels (channels 1 and 2).
    """
    # High brightness in R and G, distinguishable from the green zone
    r = pixel_row[:, R_CH].astype(np.int16)
    g = pixel_row[:, G_CH].astype(np.int16)
    b = pixel_row[:, B_CH].astype(np.int16)
    brightness = np.maximum(r, g)
    # Cursor is bright AND not dominantly green (to avoid the target zone)
    return (
        (brightness > CURSOR_BRIGHT_MIN) &
        # Not pure green (cursor has high R or is white/yellow)
        ((r > 150) | ((r > 100) & (g > 100) & (b > 100)))
    )



# ── Detection ─────────────────────────────────────────────────────────

def find_target_zone_center(row: np.ndarray) -> int | None:
    """
    Find the largest green/colored target zone cluster center in a single row.
    Input row is BGR format.
    Ignores small clusters (< MIN_TARGET_CLUSTER px) to filter out icons.
    """
    mask = is_target_green(row)
    xs = np.where(mask)[0]
    if len(xs) < MIN_TARGET_CLUSTER:
        return None

    # Find largest contiguous cluster
    diffs = np.diff(xs)
    breaks = np.where(diffs > 5)[0]  # gap > 5px = different cluster
    best_cluster = None
    best_len = 0
    start = 0
    for b in breaks:
        c = xs[start:b + 1]
        if len(c) > best_len:
            best_len = len(c)
            best_cluster = c
        start = b + 1
    c = xs[start:]
    if len(c) > best_len:
        best_len = len(c)
        best_cluster = c

    if best_cluster is None or len(best_cluster) < MIN_TARGET_CLUSTER:
        return None
    return int((best_cluster[0] + best_cluster[-1]) // 2)


def find_cursor_vertical(frame: np.ndarray) -> int | None:
    """
    Find the cursor as a thin bright vertical line by projecting across rows.
    
    Strategy: For each column x, count how many rows have a bright cursor pixel
    at that x. A real vertical cursor line will have many rows lit up in a 
    narrow column range. Random noise won't.
    
    Input frame is BGR format, shape (H, W, 3).
    """
    h, w = frame.shape[:2]
    
    # Build a column histogram: for each x, how many rows have a bright pixel there
    # We check a band of rows in the middle of the bar (avoid edges/decorations)
    y_start = max(0, h // 4)
    y_end = min(h, h * 3 // 4)
    
    col_hits = np.zeros(w, dtype=np.int32)
    
    for y in range(y_start, y_end, 1):
        row = frame[y, :, :]
        bright_mask = is_bright_cursor(row)
        # Exclude pixels that are in the green zone (target zone is also bright)
        green_mask = is_target_green(row)
        cursor_mask = bright_mask & ~green_mask
        col_hits += cursor_mask.astype(np.int32)
    
    # The cursor should be a narrow spike in the histogram
    # Find columns with significant hits
    if col_hits.max() < CURSOR_MIN_ROWS:
        return None
    
    # Threshold: at least CURSOR_MIN_ROWS hits
    threshold = max(CURSOR_MIN_ROWS, col_hits.max() * 0.5)
    candidates = np.where(col_hits >= threshold)[0]
    
    if len(candidates) == 0:
        return None
    
    # Cluster the candidate columns (cursor is thin, 1-8px wide)
    diffs = np.diff(candidates)
    breaks = np.where(diffs > 3)[0]
    
    clusters = []
    start = 0
    for b in breaks:
        clusters.append(candidates[start:b + 1])
        start = b + 1
    clusters.append(candidates[start:])
    
    # Pick the narrowest cluster (most cursor-like) that has high column hits
    best_cluster = None
    best_score = 0
    
    for c in clusters:
        width = c[-1] - c[0] + 1
        if width > CURSOR_MAX_WIDTH * 3:
            continue  # Too wide, probably not cursor
        # Score = average hits in this cluster
        avg_hits = np.mean(col_hits[c[0]:c[-1]+1])
        if avg_hits > best_score:
            best_score = avg_hits
            best_cluster = c
    
    if best_cluster is None:
        return None
    
    # Return the center weighted by hit count
    cols = best_cluster
    weights = col_hits[cols[0]:cols[-1]+1].astype(np.float64)
    if weights.sum() == 0:
        return int((cols[0] + cols[-1]) // 2)
    center = np.average(np.arange(cols[0], cols[-1]+1), weights=weights)
    return int(center)


def find_target_zone_multi_row(frame: np.ndarray) -> int | None:
    """
    Find the target zone center by scanning multiple rows and averaging.
    More robust than single-row detection.
    Input frame is BGR, shape (H, W, 3).
    """
    h = frame.shape[0]
    centers = []
    
    # Scan middle portion of bar
    y_start = max(0, h // 4)
    y_end = min(h, h * 3 // 4)
    
    for y in range(y_start, y_end, 2):
        row = frame[y, :, :]
        center = find_target_zone_center(row)
        if center is not None:
            centers.append(center)
    
    if len(centers) == 0:
        return None
    
    # Return median center (robust to outliers)
    return int(np.median(centers))


def scan_bar_v6(frame: np.ndarray) -> tuple[int | None, int | None]:
    """
    Detect cursor and target zone positions in the bar frame.
    
    - Target zone: found by row-scanning for the large colored region
    - Cursor: found by vertical column projection (thin bright line)
    
    Returns (cursor_x, target_x) or (None, None).
    """
    target = find_target_zone_multi_row(frame)
    cursor = find_cursor_vertical(frame)
    return cursor, target



def row_has_real_target(row: np.ndarray) -> bool:
    """Check if a row has a target zone cluster wide enough to be the real bar."""
    mask = is_target_green(row)
    xs = np.where(mask)[0]
    if len(xs) < MIN_TARGET_CLUSTER:
        return False
    # Check largest cluster size
    diffs = np.diff(xs)
    breaks = np.where(diffs > 5)[0]
    start = 0
    for b in breaks:
        if (b - start + 1) >= MIN_TARGET_CLUSTER:
            return True
        start = b + 1
    return (len(xs) - start) >= MIN_TARGET_CLUSTER


def detect_f_glow(frame: np.ndarray) -> bool:
    """Detect the blue F-button glow in bottom-right. Frame is BGR."""
    h, w = frame.shape[:2]
    y1, y2 = int(h * 0.82), int(h * 0.98)
    x1, x2 = int(w * 0.85), int(w * 0.98)
    region = frame[y1:y2, x1:x2]
    # Blue glow: high B (channel 0), low R (channel 2), moderate G (channel 1)
    glow_mask = (
        (region[:, :, B_CH] > GLOW_B_MIN) &
        (region[:, :, R_CH] < GLOW_R_MAX) &
        (region[:, :, G_CH] > GLOW_G_MIN)
    )
    return np.sum(glow_mask) > GLOW_THRESHOLD


def bar_visible(frame: np.ndarray) -> bool:
    """Check if the real minigame bar (not icons) is on screen. Frame is BGR."""
    check_height = frame.shape[0] // 8
    for y in range(0, check_height, 2):
        if row_has_real_target(frame[y, :, :]):
            return True
    return False


# ── Monitor ───────────────────────────────────────────────────────────

def pick_monitor(sct, monitor_idx: int | None) -> dict:
    if monitor_idx is not None:
        if monitor_idx < 1 or monitor_idx >= len(sct.monitors):
            print(f"[!] Monitor {monitor_idx} not found. Available:")
            for i, m in enumerate(sct.monitors[1:], 1):
                print(f"    {i}: {m['width']}x{m['height']} ({m.get('name','')})")
            sys.exit(1)
        return sct.monitors[monitor_idx]

    # Auto-detect: try each monitor for bar elements
    print("[i] Auto-detecting game monitor...")
    for i, m in enumerate(sct.monitors[1:], 1):
        frame = np.array(sct.grab(m))[:, :, :3]
        if bar_visible(frame):
            print(f"[i] Found game on monitor {i}: {m['width']}x{m['height']}")
            return m

    best = max(sct.monitors[1:], key=lambda m: m["width"] * m["height"])
    print(f"[i] No bar found, using largest monitor: {best['width']}x{best['height']}")
    return best


def find_bar_region(sct, monitor: dict) -> dict | None:
    """Find the bar by looking for rows with real target zone clusters. Frame is BGR."""
    top_region = {
        "left": monitor["left"],
        "top": monitor["top"],
        "width": monitor["width"],
        "height": monitor["height"] // 5,
    }
    frame = np.array(sct.grab(top_region))[:, :, :3]

    first_row = None
    last_row = None

    for y in range(frame.shape[0]):
        row = frame[y, :, :]
        if row_has_real_target(row):
            if first_row is None:
                first_row = y
            last_row = y

    if first_row is None:
        return None

    # Add some padding around the detected rows
    top = max(0, first_row - 10)
    bottom = min(frame.shape[0], last_row + 15)
    height = max(bottom - top, 40)

    return {
        "left": monitor["left"],
        "top": monitor["top"] + top,
        "width": monitor["width"],
        "height": height,
    }



# ── Calibrate ─────────────────────────────────────────────────────────

def calibrate(monitor_idx: int | None):
    with mss.mss() as sct:
        print("[i] Available monitors:")
        for i, m in enumerate(sct.monitors[1:], 1):
            print(f"    {i}: {m['width']}x{m['height']} at left={m['left']} ({m.get('name','')})")

        monitor = pick_monitor(sct, monitor_idx)
        print(f"\n[i] Using: {monitor['width']}x{monitor['height']} ({monitor.get('name','')})")

        full = np.array(sct.grab(monitor))[:, :, :3]  # BGR
        print(f"[i] Captured: {full.shape[1]}x{full.shape[0]} (BGR format)")

        # Show a sample pixel to verify BGR
        mid_y, mid_x = full.shape[0] // 2, full.shape[1] // 2
        px = full[mid_y, mid_x]
        print(f"[i] Center pixel (BGR): B={px[0]} G={px[1]} R={px[2]}")

        glow = detect_f_glow(full)
        print(f"\n[Phase 2] F glow: {'YES' if glow else 'no'}")

        bar_region = find_bar_region(sct, monitor)
        if bar_region:
            bar_frame = np.array(sct.grab(bar_region))[:, :, :3]  # BGR
            print(f"\n[Phase 3] Bar region: top={bar_region['top']}, height={bar_region['height']}")
            print(f"  Bar frame: {bar_frame.shape[1]}x{bar_frame.shape[0]}")

            cursor, target = scan_bar_v6(bar_frame)
            print(f"  Cursor (vertical line): {cursor}")
            print(f"  Target zone center:     {target}")
            if cursor is not None and target is not None:
                diff = target - cursor
                print(f"  Offset: {diff:+d}px -> {'D' if diff > 0 else 'A'}")
                print("  [OK] Detection working!")
            else:
                if cursor is None:
                    print("  [!] Cursor not found — trying debug...")
                    _debug_cursor(bar_frame)
                if target is None:
                    print("  [!] Target zone not found — trying debug...")
                    _debug_target(bar_frame)

            # Save bar region as debug image
            bar_debug_path = os.path.expanduser("~/fishing_bar_debug.png")
            # Convert BGR to RGB for saving
            bar_rgb = bar_frame[:, :, ::-1]
            Image.fromarray(bar_rgb).save(bar_debug_path)
            print(f"\n[i] Bar debug image: {bar_debug_path}")
        else:
            print("\n[Phase 3] No bar detected")

        if not glow and not bar_region:
            print("\n[Phase 1] Idle — no glow, no bar (start fishing first)")

        debug_path = os.path.expanduser("~/fishing_debug.png")
        full_rgb = full[:, :, ::-1]
        Image.fromarray(full_rgb).save(debug_path)
        print(f"[i] Full screenshot: {debug_path}")

        # Live detection test: capture 10 frames rapidly to show if values change
        if bar_region:
            print("\n[i] Live test: 10 rapid captures...")
            for i in range(10):
                time.sleep(0.05)
                f = np.array(sct.grab(bar_region))[:, :, :3]
                c, t = scan_bar_v6(f)
                print(f"  Frame {i+1}: cursor={c} target={t}")


def _debug_cursor(frame: np.ndarray):
    """Debug helper: show column brightness histogram."""
    h, w = frame.shape[:2]
    y_start = max(0, h // 4)
    y_end = min(h, h * 3 // 4)
    
    col_hits = np.zeros(w, dtype=np.int32)
    for y in range(y_start, y_end):
        row = frame[y, :, :]
        bright_mask = is_bright_cursor(row)
        green_mask = is_target_green(row)
        cursor_mask = bright_mask & ~green_mask
        col_hits += cursor_mask.astype(np.int32)
    
    top_cols = np.argsort(col_hits)[-10:][::-1]
    print(f"    Top 10 columns by brightness hits:")
    for x in top_cols:
        print(f"      x={x}: {col_hits[x]} hits")
    print(f"    Max hits: {col_hits.max()}, needed: {CURSOR_MIN_ROWS}")
    
    # Also show what a middle row looks like
    mid_y = (y_start + y_end) // 2
    row = frame[mid_y, :, :]
    bright = is_bright_cursor(row)
    bright_xs = np.where(bright)[0]
    if len(bright_xs) > 0:
        print(f"    Middle row (y={mid_y}): {len(bright_xs)} bright pixels at x={bright_xs[:20]}...")
        # Show pixel values at those positions
        for x in bright_xs[:5]:
            px = row[x]
            print(f"      x={x}: B={px[0]} G={px[1]} R={px[2]}")


def _debug_target(frame: np.ndarray):
    """Debug helper: show green detection per row."""
    h = frame.shape[0]
    for y in range(0, h, max(1, h // 10)):
        row = frame[y, :, :]
        mask = is_target_green(row)
        xs = np.where(mask)[0]
        if len(xs) > 3:
            print(f"    y={y}: {len(xs)} green pixels, range x={xs[0]}-{xs[-1]}")
            # Show sample pixel values
            for x in xs[:3]:
                px = row[x]
                print(f"      x={x}: B={px[0]} G={px[1]} R={px[2]}")



# ── Main loop ─────────────────────────────────────────────────────────

def main_loop(monitor_idx: int | None, autocast: bool, deadzone: int, delay: float):
    print("[*] Starting in 3 seconds — focus the game window NOW!")
    time.sleep(3)
    print("[*] Running! Press Ctrl+C to stop.")
    if autocast:
        print("[*] Auto-cast enabled.")
    print()

    with mss.mss() as sct:
        monitor = pick_monitor(sct, monitor_idx)
        print(f"[i] Monitor: {monitor['width']}x{monitor['height']} at ({monitor['left']},{monitor['top']})\n")

        full_region = {
            "left": monitor["left"],
            "top": monitor["top"],
            "width": monitor["width"],
            "height": monitor["height"],
        }

        state = "IDLE"
        bar_region = None
        action_count = 0
        last_cast_time = 0
        frame_count = 0
        no_detect_count = 0

        while running:
            # ── MINIGAME ──────────────────────────────────────────
            if state == "MINIGAME" and bar_region:
                frame = np.array(sct.grab(bar_region))[:, :, :3]  # BGR
                
                # Periodically re-detect bar region to handle drift
                frame_count += 1
                if frame_count % BAR_REFRESH_INTERVAL == 0:
                    new_region = find_bar_region(sct, monitor)
                    if new_region is not None:
                        bar_region = new_region
                        frame = np.array(sct.grab(bar_region))[:, :, :3]
                
                cursor, target = scan_bar_v6(frame)

                if cursor is not None and target is not None:
                    no_detect_count = 0
                    diff = target - cursor
                    if abs(diff) > deadzone:
                        key = "d" if diff > 0 else "a"
                        press_key(key)
                        action_count += 1
                        if action_count % 20 == 0:
                            print(f"  [GAME] cursor={cursor:4d} target={target:4d} "
                                  f"offset={diff:+5d} -> {key.upper()}")
                    else:
                        if action_count % 50 == 0:
                            print(f"  [GAME] cursor={cursor:4d} target={target:4d} "
                                  f"offset={diff:+5d} -> (in deadzone)")
                    time.sleep(delay)
                    continue
                else:
                    no_detect_count += 1
                    if no_detect_count > 30:
                        # Lost detection for too long — re-check if minigame ended
                        bar_region = find_bar_region(sct, monitor)
                        if bar_region is None:
                            print("[OK] Minigame ended!")
                            state = "IDLE"
                            last_cast_time = time.time()
                            no_detect_count = 0
                            time.sleep(1.0)
                        else:
                            no_detect_count = 0
                    time.sleep(delay)
                    continue

            # ── FULL SCREEN ───────────────────────────────────────
            full = np.array(sct.grab(full_region))[:, :, :3]  # BGR

            # Check for bar (Phase 3)
            if bar_visible(full):
                bar_region = find_bar_region(sct, monitor)
                if bar_region:
                    if state != "MINIGAME":
                        print("[>] Minigame started!")
                    state = "MINIGAME"
                    frame_count = 0
                    no_detect_count = 0
                    time.sleep(delay)
                    continue

            # Check for F glow (Phase 2)
            if detect_f_glow(full):
                if state != "HOOKED":
                    print("[!] Fish hooked! Pressing F...")
                state = "HOOKED"
                press_key("f")
                action_count += 1
                time.sleep(0.3)
                continue

            # Phase 1: Idle
            if state != "IDLE":
                state = "IDLE"
                print("[~] Waiting for fish...")

            if autocast and (time.time() - last_cast_time) > CAST_COOLDOWN:
                print("[~] Casting (F)...")
                press_key("f")
                last_cast_time = time.time()
                action_count += 1

            time.sleep(IDLE_DELAY)

    print(f"\n[*] Done. Total actions: {action_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NTE Fishing Bot v6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 fishing_bot.py --calibrate --monitor 2
  python3 fishing_bot.py --monitor 2
  python3 fishing_bot.py --autocast --monitor 2
        """,
    )
    parser.add_argument("--calibrate", action="store_true",
                        help="Run calibration/detection test (no keypresses)")
    parser.add_argument("--autocast", action="store_true",
                        help="Auto-press F to cast when idle")
    parser.add_argument("--monitor", type=int, default=None,
                        help="Monitor number (1-based, use --calibrate to see list)")
    parser.add_argument("--deadzone", type=int, default=DEADZONE,
                        help=f"Pixel deadzone for A/D (default: {DEADZONE})")
    parser.add_argument("--delay", type=float, default=LOOP_DELAY,
                        help=f"Loop delay in seconds (default: {LOOP_DELAY})")
    args = parser.parse_args()

    if subprocess.run(["which", "xdotool"], capture_output=True).returncode != 0:
        print("[!] xdotool not found: sudo apt install xdotool")
        sys.exit(1)

    if args.calibrate:
        calibrate(args.monitor)
    else:
        main_loop(args.monitor, args.autocast, args.deadzone, args.delay)
