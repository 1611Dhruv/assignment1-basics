"""
pyspy_flame.py
==============

Profile a block of code in a Jupyter kernel with py-spy and render the
resulting speedscope profile as a flamegraph inline in the notebook,
no trip to speedscope.app required.

Usage in a notebook
--------------------

    from cs336_basics.perf import pyspy_profile, render_speedscope

    with pyspy_profile("/tmp/profile.speedscope.json", rate=200) as p:
        heavy_function()          # <- the code you want to profile

    render_speedscope(p.path)     # draws the flamegraph inline

For click-to-zoom and hover tooltips, switch to an interactive backend
first (run `%matplotlib widget` with ipympl installed, or `%matplotlib
notebook`). On the default inline backend you still get a static
flamegraph, just without the mouse interaction.
"""

import os
import json
import time
import signal
import hashlib
import colorsys
import subprocess
from contextlib import contextmanager

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# --------------------------------------------------------------------------
# 1. Profiling: run py-spy against the current kernel for a window of time
# --------------------------------------------------------------------------

class _ProfileHandle:
    """Tiny holder so the context manager can hand back the output path."""
    def __init__(self, path):
        self.path = path


@contextmanager
def pyspy_profile(output="/tmp/profile.speedscope.json", rate=200, subprocesses=True, warmup=0.5, sudo=False):
    """
    Context manager that records a py-spy speedscope profile of this
    Python process while the body runs.

    output       where to write the speedscope JSON
    rate         samples per second
    native       include native (C extension) frames, e.g. numpy / torch
    subprocesses also sample child processes
    warmup       seconds to wait for py-spy to attach before timing starts
    sudo         prefix the command with sudo (needed on macOS / strict ptrace)
    """
    pid = os.getpid()
    cmd = ["py-spy", "record",
           "--pid", str(pid),
           "--format", "speedscope",
           "--output", output,
           "--rate", str(rate)]

    if subprocesses:
        cmd.append("--subprocesses")
    if sudo:
        cmd = ["sudo"] + cmd

    proc = subprocess.Popen(cmd, stderr=subprocess.DEVNULL, text=True)

    # py-spy writes the file only on exit, so we can't poll it for readiness.
    # A short warmup lets it attach before the workload starts.
    time.sleep(warmup)
    if proc.poll() is not None:
        err = proc.stderr.read() if proc.stderr else ""
        print("Process ended before py-spy could attatch")
        raise RuntimeError(
            "py-spy exited before profiling started. Common cause is "
            "ptrace permissions. Try sudo=True or relax ptrace_scope.\n\n"
            f"py-spy said:\n{err}"
        )

    try:
        yield _ProfileHandle(output)
    finally:
        # SIGINT makes py-spy flush the profile to disk cleanly.
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


# --------------------------------------------------------------------------
# 2. Parsing: speedscope JSON -> a merged call tree
# --------------------------------------------------------------------------

class _Node:
    __slots__ = ("name", "detail", "value", "children")

    def __init__(self, name, detail=""):
        self.name = name
        self.detail = detail          # file:line, shown in tooltips
        self.value = 0.0              # total weight in this subtree
        self.children = {}            # frame name -> _Node


def _frame_label(frame):
    name = frame.get("name", "?")
    f = frame.get("file")
    line = frame.get("line")
    detail = ""
    if f:
        detail = f"{os.path.basename(f)}:{line}" if line else os.path.basename(f)
    return name, detail


def _build_tree(profile, frames):
    """Merge every sample stack into a single call tree."""
    root = _Node("(root)")
    samples = profile.get("samples", [])
    weights = profile.get("weights") or [1] * len(samples)

    for stack, w in zip(samples, weights):
        root.value += w
        node = root
        for fidx in stack:                 # speedscope stacks are root-first
            name, detail = _frame_label(frames[fidx])
            child = node.children.get(name)
            if child is None:
                child = _Node(name, detail)
                node.children[name] = child
            child.value += w
            node = child
    return root


def _pick_profile(data):
    """py-spy emits one profile per thread. Pick the heaviest by default."""
    profiles = data["profiles"]
    if len(profiles) == 1:
        return profiles[0]
    totals = [sum(p.get("weights") or [1] * len(p.get("samples", [])))
              for p in profiles]
    best = max(range(len(profiles)), key=lambda i: totals[i])
    names = [p.get("name", f"profile {i}") for i, p in enumerate(profiles)]
    print(f"{len(profiles)} profiles found: {names}. "
          f"Showing the heaviest ({names[best]!r}). "
          f"Pass thread=<index> to choose another.")
    return profiles[best]


# --------------------------------------------------------------------------
# 3. Time formatting based on the profile's declared unit
# --------------------------------------------------------------------------

_UNIT_TO_SECONDS = {
    "nanoseconds": 1e-9, "microseconds": 1e-6,
    "milliseconds": 1e-3, "seconds": 1.0,
}


def _make_formatter(unit):
    factor = _UNIT_TO_SECONDS.get(unit)
    if factor is None:
        return lambda v: f"{v:.0f} samples"

    def fmt(v):
        s = v * factor
        if s >= 1:
            return f"{s:.2f} s"
        if s >= 1e-3:
            return f"{s * 1e3:.1f} ms"
        if s >= 1e-6:
            return f"{s * 1e6:.1f} us"
        return f"{s * 1e9:.0f} ns"
    return fmt


# --------------------------------------------------------------------------
# 4. Rendering: draw the tree as a flamegraph with matplotlib
# --------------------------------------------------------------------------

def _color_for(name):
    """Stable warm flamegraph color derived from the function name."""
    h = int(hashlib.md5(name.encode()).hexdigest(), 16)
    hue = (h % 100) / 100.0 * 0.12          # red -> orange-yellow band
    sat = 0.55 + (h // 100 % 100) / 100.0 * 0.25
    val = 0.85 + (h // 10000 % 100) / 100.0 * 0.10
    return colorsys.hsv_to_rgb(hue, sat, val)


class _Flamegraph:
    def __init__(self, root, fmt, max_depth, min_frac, figsize, interactive):
        self.root = root
        self.fmt = fmt
        self.max_depth = max_depth
        self.min_frac = min_frac
        self.interactive = interactive
        self.rects = []                      # (x0, x1, depth, node) for hit-testing
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.annot = None
        self._draw(self.root)
        if interactive:
            self.fig.canvas.mpl_connect("button_press_event", self._on_click)
            self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)

    def _draw(self, focus):
        self.ax.clear()
        self.rects.clear()
        total = focus.value or 1.0
        self.focus = focus
        self._max_depth_seen = 0

        def recurse(node, depth, x0):
            if self.max_depth is not None and depth > self.max_depth:
                return
            width = node.value
            if width / total < self.min_frac:
                return
            self._max_depth_seen = max(self._max_depth_seen, depth)
            color = (0.85, 0.85, 0.85) if node is self.root else _color_for(node.name)
            self.ax.add_patch(Rectangle(
                (x0, depth), width, 0.9,
                facecolor=color, edgecolor="white", linewidth=0.5))
            self.rects.append((x0, x0 + width, depth, node))

            # Label if the box is wide enough to fit readable text.
            frac = width / total
            if frac > 0.012:
                budget = max(1, int(frac * 110))
                label = node.name if len(node.name) <= budget else node.name[:budget - 1] + "\u2026"
                self.ax.text(x0 + width / 2, depth + 0.45, label,
                             ha="center", va="center", fontsize=8, clip_on=True)

            cx = x0
            for child in sorted(node.children.values(),
                                key=lambda c: c.value, reverse=True):
                recurse(child, depth + 1, cx)
                cx += child.value

        recurse(focus, 0, 0)

        self.ax.set_xlim(0, total)
        self.ax.set_ylim(self._max_depth_seen + 1, 0)   # invert: root on top
        self.ax.set_yticks([])
        self.ax.set_xticks([])
        pct = focus.value / (self.root.value or 1.0) * 100
        title = f"{focus.name}  |  {self.fmt(focus.value)}  ({pct:.1f}% of total)"
        if focus is not self.root:
            title += "   [click the top row to reset]"
        self.ax.set_title(title, fontsize=10)
        self.annot = self.ax.annotate(
            "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
            bbox=dict(boxstyle="round", fc="black", alpha=0.85),
            color="white", fontsize=8, visible=False, zorder=10)
        self.fig.canvas.draw_idle()

    def _hit(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return None
        depth = int(event.ydata)
        for x0, x1, d, node in self.rects:
            if d == depth and x0 <= event.xdata <= x1:
                return node
        return None

    def _on_click(self, event):
        node = self._hit(event)
        if node is None:
            return
        # Clicking the focused root row resets to the full graph.
        self._draw(self.root if node is self.focus else node)

    def _on_move(self, event):
        node = self._hit(event)
        if node is None:
            if self.annot and self.annot.get_visible():
                self.annot.set_visible(False)
                self.fig.canvas.draw_idle()
            return
        pct = node.value / (self.root.value or 1.0) * 100
        text = f"{node.name}\n{self.fmt(node.value)}  ({pct:.1f}%)"
        if node.detail:
            text += f"\n{node.detail}"
        self.annot.set_text(text)
        self.annot.xy = (event.xdata, event.ydata)
        self.annot.set_visible(True)
        self.fig.canvas.draw_idle()


def render_speedscope(path, thread=None, max_depth=None, min_frac=0.0,
                      figsize=(14, 8), interactive=True):
    """
    Parse a speedscope JSON file and draw it as a flamegraph inline.

    path        path to the speedscope .json written by py-spy
    thread      pick a specific profile/thread by index (default: heaviest)
    max_depth   clip the stack depth that gets drawn
    min_frac    skip frames smaller than this fraction of total (e.g. 0.005)
    figsize     matplotlib figure size
    interactive enable click-to-zoom and hover (needs an interactive backend)

    Returns the matplotlib Figure.
    """
    with open(path) as fh:
        data = json.load(fh)

    frames = data["shared"]["frames"]
    profile = data["profiles"][thread] if thread is not None else _pick_profile(data)
    fmt = _make_formatter(profile.get("unit", "none"))
    root = _build_tree(profile, frames)

    fg = _Flamegraph(root, fmt, max_depth, min_frac, figsize, interactive)
    return fg.fig


import psutil
import time
import os
import threading

def tree_rss_pss_mb(pid = None):
    parent = psutil.Process(pid or os.getpid())
    procs = [parent] + parent.children(recursive=True)
    total_rss = 0
    total_pss = 0
    for pr in procs:
        try:
            total_rss += pr.memory_info().rss
            total_pss += pr.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total_rss / 1e6, total_pss / 1e6

class PeakMoniter(threading.Thread):
    def __init__(self, interval=0.1):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_rss = 0
        self.peak_pss = 0
        self._run = True

    def run(self):
        while self._run:
            curr_rss, curr_pss = tree_rss_pss_mb()
            self.peak_rss = max(self.peak_rss, curr_rss)
            self.peak_pss = max(self.peak_pss, curr_pss)
            time.sleep(self.interval)
    
    def stop(self):
        self._run = False
        return self.peak_rss, self.peak_pss