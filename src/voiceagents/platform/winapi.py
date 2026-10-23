"""Thin ctypes helpers for the Win32 calls the app needs."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int = 0  # 0 = no foreground window
    title: str = ""


def foreground_window() -> WindowInfo:
    """The window that currently has keyboard focus (hwnd 0 if there is none)."""
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return WindowInfo()
    buf = ctypes.create_unicode_buffer(user32.GetWindowTextLengthW(hwnd) + 1)
    user32.GetWindowTextW(hwnd, buf, len(buf))
    return WindowInfo(hwnd=int(hwnd), title=buf.value)
