"""Render a deterministic palette comparison from ConsoleRender's real ANSI output.

Run with: python output/tui-color-comparison/render_comparison.py
Requires the project's tui dependencies and Pillow. No model calls are made.
The before panel simulates tmux's RGB-to-256 conversion, using the default xterm
palette; it is not a capture of the user's terminal or its theme.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageFont
from rich.console import Console
from rich.text import Text
from wcwidth import wcwidth

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))
from netflux.core import AgentFunction, ModelTextPart, NodeState, NodeView
from netflux.providers import Provider
from netflux.tui import console as tui

OUT = Path(__file__).resolve().parent
BG, PANEL, FG, MUTED = "#0c1119", "#111923", "#d8e0ec", "#91a0b5"
FONT_DIR = Path("C:/Windows/Fonts")


def font(size: int, *, mono: bool = False, bold: bool = False):
    name = ("consolab.ttf" if bold else "consola.ttf") if mono else (
        "segoeuib.ttf" if bold else "segoeui.ttf"
    )
    if (FONT_DIR / name).exists():
        return ImageFont.truetype(str(FONT_DIR / name), size)
    return ImageFont.truetype(
        ("DejaVuSansMono" if mono else "DejaVuSans") + ("-Bold" if bold else "") + ".ttf",
        size,
    )


def rgb(index: int) -> tuple[int, int, int]:
    if index >= 232:
        return (8 + 10 * (index - 232),) * 3
    levels = [0, 95, 135, 175, 215, 255]
    index -= 16
    return levels[index // 36], levels[(index // 6) % 6], levels[index % 6]


def tmux_index(value: tuple[int, int, int]) -> int:
    # tmux colour_find_rgb (3.5a): compare the rounded colour cube with grey.
    # https://github.com/tmux/tmux/blob/3.5a/colour.c#L40-L80
    cube = [0 if c < 48 else 1 if c < 114 else (c - 35) // 40 for c in value]
    colour = 16 + 36 * cube[0] + 6 * cube[1] + cube[2]
    average = sum(value) // 3
    grey = 23 if average > 238 else (average - 3) // 10
    distance = lambda other: sum((a - b) ** 2 for a, b in zip(value, other))
    return 232 + grey if distance(rgb(232 + grey)) < distance(rgb(colour)) else colour


def original_palette() -> dict[Provider, str]:
    source = subprocess.check_output(
        ["git", "show", "fecc95fe5a02bc7aeacec45c629aabbec295b3a9:tui/console.py"],
        cwd=ROOT, text=True, encoding="utf-8",
    )
    assignment = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "_PROVIDER_FG" for t in n.targets))
    return {Provider[key.attr]: value.value for key, value in
            zip(assignment.value.keys, assignment.value.values)}


def node(number, provider, name, text, children=()):
    return NodeView(
        id=number,
        fn=AgentFunction(name=name, desc="Preview fixture", args=[], system_prompt="",
                         user_prompt_template="", uses=[]),
        provider=provider, inputs={}, state=NodeState.Success, outputs=None,
        exception=None, children=children, usage=None,
        transcript=(ModelTextPart(text=text),), started_at=100.0,
        ended_at=104.0 + number, update_seqnum=1,
    )


def rendered(palette):
    children = tuple(node(i, p, "review", "Independent check complete.")
                     for i, p in enumerate(list(Provider)[1:], 2))
    view = node(1, Provider.OpenAI, "coordinator", "Comparing provider responses.", children)
    renderer = tui.ConsoleRender(follow=False)
    with patch.dict(tui._PROVIDER_FG, palette, clear=True):
        renderer.render_body(width=60, height=30, view=view, tick=0)
        for i in range(1, 5):
            renderer._collapse_overrides[f"tp:{i}:0:model"] = False
        renderer.render_body(width=60, height=30, tick=0)
    return renderer._lines


def draw_ansi(draw, rows, x, y):
    regular, bold = font(21, mono=True), font(21, mono=True, bold=True)
    symbols = ImageFont.truetype(str(FONT_DIR / "seguisym.ttf"), 21) if FONT_DIR.exists() else regular
    emoji = ImageFont.truetype(str(FONT_DIR / "seguiemj.ttf"), 21) if FONT_DIR.exists() else symbols
    cell = regular.getlength("M")
    console = Console(width=100, color_system="truecolor")
    for row in rows:
        column = 0
        for segment in console.render(Text.from_ansi(row)):
            style = segment.style
            colour = tuple(style.color.get_truecolor()) if style and style.color else (216, 224, 236)
            if style and style.dim:
                colour = tuple(round(c * .63 + b * .37) for c, b in zip(colour, (17, 25, 35)))
            for char in segment.text.rstrip("\n"):
                width = max(0, wcwidth(char))
                face = bold if style and style.bold else regular
                if ord(char) > 0xffff or char == "✨":
                    face = emoji
                elif char in "✔▾▸":
                    face = symbols
                draw.text((x + column * cell, y), char, font=face, fill=colour)
                column += width
        y += 34


def main():
    before = original_palette()
    old_rgb = {p: tuple(map(int, re.findall(r"\d+", code)[2:])) for p, code in before.items()}
    old_indices = {p: tmux_index(colour) for p, colour in old_rgb.items()}
    before_256 = {p: f"\x1b[38;5;{index}m" for p, index in old_indices.items()}
    after = dict(tui._PROVIDER_FG)
    new_indices = {p: int(re.findall(r"\d+", code)[2]) for p, code in after.items()}
    before_rows, after_rows = rendered(before_256), rendered(after)
    strip = lambda row: re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", row)
    assert list(map(strip, before_rows)) == list(map(strip, after_rows))
    assert all("38;2;" not in row for row in after_rows)

    image = Image.new("RGB", (1712, 1150), BG)
    draw = ImageDraw.Draw(image)
    draw.text((48, 28), "Provider colors on a 256-color terminal", font=font(36, bold=True), fill=FG)
    draw.text((48, 82), "Same rendered tree. Explicit palette entries preserve teal, rose, blue and lavender.",
              font=font(22), fill=MUTED)
    for x, title, subtitle, rows, footer in (
        (48, "BEFORE  /  PR 73", "tmux converts the original RGB colors", before_rows,
         "Rose and lavender become neutral grey."),
        (880, "AFTER  /  256-COLOR PALETTE", "The app emits indexed colors directly", after_rows,
         "All four provider hues remain visible."),
    ):
        draw.rounded_rectangle((x, 140, x + 784, 732), radius=16, fill=PANEL, outline="#263446", width=2)
        draw.text((x + 24, 158), title, font=font(21, bold=True), fill=FG)
        draw.text((x + 24, 194), subtitle, font=font(20), fill=MUTED)
        draw.line((x + 24, 236, x + 760, 236), fill="#263446", width=1)
        draw_ansi(draw, rows, x + 25, 258)
        draw.text((x + 24, 690), footer, font=font(20), fill=MUTED)

    draw.text((48, 768), "Palette reference", font=font(25, bold=True), fill=FG)
    columns = [48, 360, 804, 1248]
    for x, title in zip(columns, ("PROVIDER", "ORIGINAL RGB", "BEFORE · TMUX 256", "AFTER · INDEXED 256")):
        draw.text((x, 814), title, font=font(17, bold=True), fill=MUTED)
    for y, provider in zip(range(858, 1050, 48), Provider):
        draw.text((48, y), provider.value, font=font(22, bold=True), fill=FG)
        entries = [(old_rgb[provider], None), (rgb(old_indices[provider]), old_indices[provider]),
                   (rgb(new_indices[provider]), new_indices[provider])]
        for x, (colour, index) in zip(columns[1:], entries):
            draw.rounded_rectangle((x, y + 2, x + 58, y + 30), radius=4, fill=colour)
            label = "#" + "".join(f"{c:02x}" for c in colour)
            if index is not None:
                label = f"{index:<3}  {label}"
            draw.text((x + 76, y), label, font=font(21, mono=True), fill=FG)
    draw.text((48, 1086), "Simulated default xterm palette · Tree text and ANSI styles come from ConsoleRender.",
              font=font(19), fill=MUTED)
    draw.text((48, 1115), "Terminal themes can vary. Direct 256-only terminals may ignore the old RGB sequences instead.",
              font=font(18), fill=MUTED)
    path = OUT / "before-after.png"
    image.save(path)
    print(path)
    print("Old palette:", {p.value: v for p, v in old_indices.items()})
    print("New palette:", {p.value: v for p, v in new_indices.items()})


if __name__ == "__main__":
    main()
