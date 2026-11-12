from __future__ import annotations

import atexit
import select
import sys
import termios
import time
import threading
import tty
from dataclasses import dataclass

import numpy as np


@dataclass
class KeyboardControllerConfig:
    vx_speed: float = 0.8
    vy_speed: float = 0.5
    wz_speed: float = 0.8
    min_vx: float = -0.8
    max_vx: float = 1.2
    min_vy: float = -0.7
    max_vy: float = 0.7
    min_wz: float = -1.0
    max_wz: float = 1.0
    release_timeout: float = 0.35


class TerminalKeyboardController:
    """Video-game-style velocity command controlled from the terminal.

    Keys:
      w/s: forward/backward
      a/d: strafe left/right
      q/e: yaw left/right
      space or x: stop
      r: reset

    Each axis is commanded at a fixed speed while its key is held, and
    snaps to zero the instant the key is released -- there is no
    increment/hold-between-presses state like a classic teleop tool.

    Terminals never report key-up events, only key-down bytes, so "held"
    is inferred from the OS's own key-repeat stream: as long as repeat
    bytes for a key keep arriving faster than `release_timeout`, the axis
    stays active; once they stop arriving, it zeros on the next poll tick.
    If a hold feels like it blips to zero right after the initial press
    (the gap before the OS's auto-repeat kicks in), raise
    --high-level.keyboard.release-timeout to match your terminal's key
    repeat delay.

    The terminal running the script must have keyboard focus.
    """

    _AXES: dict[int, tuple[str, str]] = {
        0: ("w", "s"),  # vx: forward, backward
        1: ("a", "d"),  # vy: left, right
        2: ("q", "e"),  # wz: left, right
    }

    def __init__(self, cfg: KeyboardControllerConfig) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError("Keyboard mode requires an interactive terminal (TTY).")
        self.cfg = cfg
        self._speeds = (cfg.vx_speed, cfg.vy_speed, cfg.wz_speed)
        self._mins = (cfg.min_vx, cfg.min_vy, cfg.min_wz)
        self._maxs = (cfg.max_vx, cfg.max_vy, cfg.max_wz)
        self._last_seen = {key: 0.0 for keys in self._AXES.values() for key in keys}
        self._command = np.zeros(3, dtype=np.float64)
        self._printed_command = np.zeros(3, dtype=np.float64)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        atexit.register(self.close)
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        self._print_help()

    def _print_help(self) -> None:
        print(
            "[KEYBOARD] w/s: vx  a/d: vy  q/e: wz  space/x: stop  r: reset\n"
            "[KEYBOARD] Video-game style: a command is active only while its "
            f"key is held (release_timeout={self.cfg.release_timeout:.2f}s), "
            "and snaps to zero on release."
        )

    def _reader_loop(self) -> None:
        poll_dt = min(0.05, self.cfg.release_timeout / 2.0)
        while not self._stop.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], poll_dt)
            if ready:
                char = sys.stdin.read(1)
                self._handle_key(char)
            self._update_command()

    def _handle_key(self, char: str) -> None:
        now = time.monotonic()
        with self._lock:
            if char in self._last_seen:
                self._last_seen[char] = now
            elif char in {" ", "x", "r"}:
                for key in self._last_seen:
                    self._last_seen[key] = 0.0

    def _update_command(self) -> None:
        now = time.monotonic()
        with self._lock:
            command = np.zeros(3, dtype=np.float64)
            for axis, (pos_key, neg_key) in self._AXES.items():
                pos_active = now - self._last_seen[pos_key] <= self.cfg.release_timeout
                neg_active = now - self._last_seen[neg_key] <= self.cfg.release_timeout
                if pos_active and neg_active:
                    # Both keys for this axis are "held"; the more recently
                    # pressed one wins.
                    sign = 1.0 if self._last_seen[pos_key] >= self._last_seen[neg_key] else -1.0
                elif pos_active:
                    sign = 1.0
                elif neg_active:
                    sign = -1.0
                else:
                    sign = 0.0
                value = sign * self._speeds[axis]
                command[axis] = np.clip(value, self._mins[axis], self._maxs[axis])

            self._command = command
            if not np.array_equal(command, self._printed_command):
                self._printed_command = command.copy()
                print(
                    f"\r[KEYBOARD] command=({command[0]:+.2f}, "
                    f"{command[1]:+.2f}, {command[2]:+.2f})   ",
                    end="",
                    flush=True,
                )

    def command(self) -> np.ndarray:
        with self._lock:
            return self._command.copy()

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
        except Exception:
            pass
