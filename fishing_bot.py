#!/usr/bin/env python3
"""
NTE Fishing Auto-Player v5 (Linux / Zorin OS)
===============================================
Handles the full fishing loop:
  Phase 1 — IDLE:     No bar, no glow → wait (--autocast to press F)
  Phase 2 — HOOKED:   Blue glow on F button → press F to reel in
  Phase 3 — MINIGAME: Yellow cursor + green zone → A/D tracking

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
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

YELLOW_R_MIN = 180
YELLOW_G_MIN = 150
YELLOW_B_MAX = 120

GREEN_G_MIN = 180
GREEN_R_MAX = 120
GREEN_B_MIN = 150

# Minimum width (in pixels) for a green cluster to count as the target zone.
# The Fishing Line icon ring is ~6px wide; the real green zone is 100-200+px.
MIN_GREEN_CLUSTER = 30

GLOW_B_MIN = 200
GLOW_R_MAX = 150
GLOW_G_MIN = 140
GLOW_THRESHOLD = 200

LOOP_DELAY = 0.015
IDLE_DELAY = 0.15
KEY_HOLD_MS = 30
DEADZONE = 8
CAST_COOLDOWN = 3.0

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


# ── Detection ─────────────────────────────────────────────────────────

def find_yellow_cursor_center(row: np.ndarray) -> int | None:
    """
    Find the yellow bracket cursor by clustering yellow pixels and
    picking the two closest clusters (left + right bracket).
    """
    mask = (
        (row[:, 0] > YELLOW_R_MIN) &
        (row[:, 1] > YELLOW_G_MIN) &
        (row[:, 2] < YELLOW_B_MAX)
    )
    xs = np.where(mask)[0]
    if len(xs) < 2:
        return None

    diffs = np.diff(xs)
    breaks = np.where(diffs > 10)[0]
    clusters = []
    start = 0
    for b in breaks:
        clusters.append(xs[start:b + 1])
        start = b + 1
    clusters.append(xs[start:])

    if len(clusters) >= 2:
        best_pair = None
        best_gap = 9999
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                gap = clusters[j][0] - clusters[i][-1]
                if gap < best_gap:
                    best_gap = gap
                    best_pair = (i, j)
        left = clusters[best_pair[0]]
        right = clusters[best_pair[1]]
        return int((left[0] + right[-1]) // 2)
    elif len(clusters) == 1 and len(clusters[0]) >= 2:
        c = clusters[0]
        return int((c[0] + c[-1]) // 2)
    return None


def find_green_zone_center(row: np.ndarray) -> int | None:
    """
    Find the largest green cluster's center.
    Ignores small clusters (< MIN_GREEN_CLUSTER px) to filter out icons.
    """
    mask = (
        (row[:, 1] > GREEN_G_MIN) &
        (row[:, 0] < GREEN_R_MAX) &
        (row[:, 2] > GREEN_B_MIN)
    )
    xs = np.where(mask)[0]
    if len(xs) < MIN_GREEN_CLUSTER:
        return None

    # Find largest contiguous cluster
    diffs = np.diff(xs)
    breaks = np.where(diffs > 15)[0]
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

    if best_cluster is None or len(best_cluster) < MIN_GREEN_CLUSTER:
        return None
    return int((best_cluster[0] + best_cluster[-1]) // 2)


def scan_bar(frame: np.ndarray) -> tuple[int | None, int | None]:
    """
    Scan rows to find the one with the biggest real green zone + yellow cursor.
    """
    h = frame.shape[0]
    best_cursor = None
    best_target = None
    best_green_count = 0

    for y in range(h - 1, 0, -2):  # step by 2 for speed
        row = frame[y, :, :]

        # Quick filter: count green pixels
        green_mask = (
            (row[:, 1] > GREEN_G_MIN) &
            (row[:, 0] < GREEN_R_MAX) &
            (row[:, 2] > GREEN_B_MIN)
        )
        green_count = np.sum(green_mask)
        if green_count < MIN_GREEN_CLUSTER:
            continue

        target = find_green_zone_center(row)
        if target is None:
            continue

        cursor = find_yellow_cursor_center(row)
        if cursor is None:
            continue

        if green_count > best_green_count:
            best_green_count = green_count
            best_cursor = cursor
            best_target = target

    return best_cursor, best_target


def row_has_real_green(row: np.ndarray) -> bool:
    """Check if a row has a green cluster wide enough to be the real bar zone."""
    mask = (
        (row[:, 1] > GREEN_G_MIN) &
        (row[:, 0] < GREEN_R_MAX) &
        (row[:, 2] > GREEN_B_MIN)
    )
    xs = np.where(mask)[0]
    if len(xs) < MIN_GREEN_CLUSTER:
        return False
    # Check largest cluster size
    diffs = np.diff(xs)
    breaks = np.where(diffs > 15)[0]
    start = 0
    for b in breaks:
        if (b - start + 1) >= MIN_GREEN_CLUSTER:
            return True
        start = b + 1
    return (len(xs) - start) >= MIN_GREEN_CLUSTER


def detect_f_glow(frame: np.ndarray) -> bool:
    h, w = frame.shape[:2]
    y1, y2 = int(h * 0.82), int(h * 0.98)
    x1, x2 = int(w * 0.85), int(w * 0.98)
    region = frame[y1:y2, x1:x2]
    glow_mask = (
        (region[:, :, 2] > GLOW_B_MIN) &
        (region[:, :, 0] < GLOW_R_MAX) &
        (region[:, :, 1] > GLOW_G_MIN)
    )
    return np.sum(glow_mask) > GLOW_THRESHOLD


def bar_visible(frame: np.ndarray) -> bool:
    """Check if the real minigame bar (not icons) is on screen."""
    check_height = frame.shape[0] // 8
    for y in range(0, check_height, 2):
        if row_has_real_green(frame[y, :, :]):
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
    """Find the bar by looking for rows with real green clusters."""
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
        # Check for real green zone OR yellow brackets
        has_real_green = row_has_real_green(row)

        has_yellow = False
        if not has_real_green:
            ym = (row[:, 0] > YELLOW_R_MIN) & (row[:, 1] > YELLOW_G_MIN) & (row[:, 2] < YELLOW_B_MAX)
            yxs = np.where(ym)[0]
            if len(yxs) >= 4:
                # Must have bracket-like structure: two clusters close together
                diffs = np.diff(yxs)
                breaks = np.where(diffs > 10)[0]
                if len(breaks) >= 1:  # at least 2 clusters
                    has_yellow = True

        if has_real_green or has_yellow:
            if first_row is None:
                first_row = y
            last_row = y

    if first_row is None:
        return None

    top = max(0, first_row - 5)
    bottom = min(frame.shape[0], last_row + 10)
    height = max(bottom - top, 60)

    return {
        "left": monitor["left"],
        "top": monitor["top"] + top,
        "width": monitor["width"],
        "height": height,
    }


# ── Calibrate ─────────────────────────────────────────────────────────

def calibrate(monitor_idx: int | None):
    with mss.MSS() as sct:
        print("[i] Available monitors:")
        for i, m in enumerate(sct.monitors[1:], 1):
            print(f"    {i}: {m['width']}x{m['height']} at left={m['left']} ({m.get('name','')})")

        monitor = pick_monitor(sct, monitor_idx)
        print(f"\n[i] Using: {monitor['width']}x{monitor['height']} ({monitor.get('name','')})")

        full = np.array(sct.grab(monitor))[:, :, :3]
        print(f"[i] Captured: {full.shape[1]}x{full.shape[0]}")

        glow = detect_f_glow(full)
        print(f"\n[Phase 2] F glow: {'YES ✓' if glow else 'no'}")

        bar_region = find_bar_region(sct, monitor)
        if bar_region:
            bar_frame = np.array(sct.grab(bar_region))[:, :, :3]
            print(f"\n[Phase 3] Bar region: top={bar_region['top']}, height={bar_region['height']}")
            print(f"  Bar frame: {bar_frame.shape[1]}x{bar_frame.shape[0]}")

            cursor, target = scan_bar(bar_frame)
            print(f"  Yellow cursor: {cursor}")
            print(f"  Green zone:    {target}")
            if cursor is not None and target is not None:
                diff = target - cursor
                print(f"  Offset: {diff:+d}px → {'D' if diff > 0 else 'A'}")
                print("  [✓] Detection working!")
            else:
                if cursor is None:
                    print("  [!] Yellow cursor not found")
                if target is None:
                    print("  [!] Green zone not found — need MIN_GREEN_CLUSTER={} px".format(MIN_GREEN_CLUSTER))
                    # Debug: show what green was found
                    for y in range(bar_frame.shape[0] - 1, 0, -5):
                        row = bar_frame[y, :, :]
                        gm = (row[:,1] > GREEN_G_MIN) & (row[:,0] < GREEN_R_MAX) & (row[:,2] > GREEN_B_MIN)
                        gxs = np.where(gm)[0]
                        if len(gxs) > 3:
                            diffs = np.diff(gxs)
                            breaks = np.where(diffs > 15)[0]
                            clusters = []
                            s = 0
                            for b in breaks:
                                clusters.append(f'{gxs[s]}-{gxs[b]}({b-s+1}px)')
                                s = b + 1
                            clusters.append(f'{gxs[s]}-{gxs[-1]}({len(gxs)-s}px)')
                            print(f"    y={y}: {' | '.join(clusters)}")
        else:
            print("\n[Phase 3] No bar detected")

        if not glow and not bar_region:
            print("\n[Phase 1] Idle — no glow, no bar (start fishing first)")

        debug_path = os.path.expanduser("~/fishing_debug.png")
        Image.fromarray(full).save(debug_path)
        print(f"\n[i] Screenshot: {debug_path}")


# ── Main loop ─────────────────────────────────────────────────────────

def main_loop(monitor_idx: int | None, autocast: bool, deadzone: int, delay: float):
    print("[*] Starting in 3 seconds — focus the game window NOW!")
    time.sleep(3)
    print("[*] Running! Press Ctrl+C to stop.")
    if autocast:
        print("[*] Auto-cast enabled.")
    print()

    with mss.MSS() as sct:
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

        while running:
            # ── MINIGAME ──────────────────────────────────────────
            if state == "MINIGAME" and bar_region:
                frame = np.array(sct.grab(bar_region))[:, :, :3]
                cursor, target = scan_bar(frame)

                if cursor is not None and target is not None:
                    diff = target - cursor
                    if abs(diff) > deadzone:
                        key = "d" if diff > 0 else "a"
                        press_key(key)
                        action_count += 1
                        if action_count % 30 == 0:
                            print(f"  [GAME] cursor={cursor:4d} target={target:4d} "
                                  f"offset={diff:+5d} → {key.upper()}")
                    time.sleep(delay)
                    continue
                else:
                    # Re-detect bar region in case it shifted
                    bar_region = find_bar_region(sct, monitor)
                    if bar_region is None:
                        print("[✓] Minigame ended!")
                        state = "IDLE"
                        last_cast_time = time.time()
                        time.sleep(1.0)
                    else:
                        time.sleep(delay)
                    continue

            # ── FULL SCREEN ───────────────────────────────────────
            full = np.array(sct.grab(full_region))[:, :, :3]

            # Check for bar (Phase 3)
            if bar_visible(full):
                bar_region = find_bar_region(sct, monitor)
                if bar_region:
                    if state != "MINIGAME":
                        print("[►] Minigame started!")
                    state = "MINIGAME"
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
        description="NTE Fishing Bot v5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 fishing_bot.py --calibrate --monitor 2
  python3 fishing_bot.py --monitor 2
  python3 fishing_bot.py --autocast --monitor 2
        """,
    )
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--autocast", action="store_true")
    parser.add_argument("--monitor", type=int, default=None)
    parser.add_argument("--deadzone", type=int, default=DEADZONE)
    parser.add_argument("--delay", type=float, default=LOOP_DELAY)
    args = parser.parse_args()

    if subprocess.run(["which", "xdotool"], capture_output=True).returncode != 0:
        print("[!] xdotool not found: sudo apt install xdotool")
        sys.exit(1)

    if args.calibrate:
        calibrate(args.monitor)
    else:
        main_loop(args.monitor, args.autocast, args.deadzone, args.delay)
