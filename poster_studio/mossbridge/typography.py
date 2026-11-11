"""Tipografía real sobre el arte generado (modo híbrido).

MOSS dibuja el texto *dentro* del modelo: no elegís fuente, ni kerning, ni interlineado, y el
texto no se puede editar después. Para piezas de estudio conviene lo contrario: el modelo hace
el arte y acá se compone tipografía real con Pillow, con las fuentes del sistema.

Solo necesita Pillow.
"""

from __future__ import annotations

import os
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT = "C:/Windows/Fonts/segoeui.ttf"
FALLBACK_FONTS = [
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


@lru_cache(maxsize=64)
def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    for candidate in (path, DEFAULT_FONT, *FALLBACK_FONTS):
        try:
            if candidate and os.path.exists(candidate):
                return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _advance(font: ImageFont.FreeTypeFont, ch: str, tracking: float) -> float:
    return font.getlength(ch) + tracking


def line_width(font: ImageFont.FreeTypeFont, text: str, tracking: float) -> float:
    if not text:
        return 0.0
    return sum(_advance(font, ch, tracking) for ch in text) - tracking


def wrap(text: str, font: ImageFont.FreeTypeFont, max_width: float, tracking: float) -> list[str]:
    """Respeta los saltos de línea escritos y después corta por palabra."""
    out: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split(" ")
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip() if current else word
            if line_width(font, candidate, tracking) <= max_width or not current:
                current = candidate
            else:
                out.append(current)
                current = word
        out.append(current)
    return out


def _draw_tracked(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str,
                  font: ImageFont.FreeTypeFont, fill, tracking: float) -> None:
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += _advance(font, ch, tracking)


def render_box(img: Image.Image, box: dict, width: int, height: int) -> None:
    """Dibuja una caja de texto. Coords normalizadas 0..1 sobre el canvas."""
    text = str(box.get("text", ""))
    if box.get("uppercase"):
        text = text.upper()
    if not text.strip():
        return

    x0 = int(round(float(box.get("x", 0.1)) * width))
    y0 = int(round(float(box.get("y", 0.1)) * height))
    x1 = int(round((float(box.get("x", 0.1)) + float(box.get("w", 0.8))) * width))
    y1 = int(round((float(box.get("y", 0.1)) + float(box.get("h", 0.1))) * height))
    bw, bh = max(4, x1 - x0), max(4, y1 - y0)

    pad = int(box.get("pad", 0) or 0)
    avail_w = max(4, bw - 2 * pad)
    avail_h = max(4, bh - 2 * pad)

    size = int(float(box.get("size", 64)))
    size = max(6, size)
    tracking = float(box.get("tracking", 0) or 0)
    line_height = float(box.get("line_height", 1.12) or 1.12)
    font_path = box.get("font") or DEFAULT_FONT

    # Ajuste automático: baja el cuerpo hasta que entre en la caja.
    lines: list[str] = []
    for _ in range(60):
        font = load_font(font_path, size)
        lines = wrap(text, font, avail_w, tracking)
        total_h = len(lines) * size * line_height
        widest = max((line_width(font, ln, tracking) for ln in lines), default=0)
        fits = total_h <= avail_h and widest <= avail_w
        if fits or size <= 6 or not box.get("fit", True):
            break
        size = max(6, int(size * 0.94))

    font = load_font(font_path, size)
    total_h = len(lines) * size * line_height
    valign = box.get("valign", "top")
    if valign == "middle":
        cursor_y = y0 + pad + max(0, (avail_h - total_h) / 2)
    elif valign == "bottom":
        cursor_y = y1 - pad - total_h
    else:
        cursor_y = y0 + pad

    align = box.get("align", "left")
    color = box.get("color") or "#ffffff"
    shadow = box.get("shadow", True)
    draw = ImageDraw.Draw(img, "RGBA")

    # Placa de contraste opcional (#rrggbb o #rrggbbaa) detrás del bloque de texto.
    bg = box.get("bg")
    if bg:
        from PIL import ImageColor
        try:
            fill = ImageColor.getrgb(bg) if len(str(bg)) <= 7 else ImageColor.getrgb(str(bg)[:7]) + (
                int(str(bg)[7:9], 16),)
        except Exception:
            fill = None
        if fill:
            pad_bg = int(box.get("bg_pad", 10))
            top = cursor_y - pad_bg
            bottom = cursor_y + total_h + pad_bg
            draw.rounded_rectangle([x0, top, x1, bottom], radius=int(box.get("bg_radius", 8)),
                                   fill=fill)

    for line in lines:
        lw = line_width(font, line, tracking)
        if align == "center":
            cursor_x = x0 + pad + (avail_w - lw) / 2
        elif align == "right":
            cursor_x = x1 - pad - lw
        else:
            cursor_x = x0 + pad
        cursor_x = max(0.0, cursor_x)
        if shadow:
            offset = max(1.0, size * 0.02)
            _draw_tracked(draw, (cursor_x + offset, cursor_y + offset),
                          line, font, (0, 0, 0, 110), tracking)
        _draw_tracked(draw, (cursor_x, cursor_y), line, font, color, tracking)
        cursor_y += size * line_height


def render_boxes(base: Image.Image, boxes: list[dict]) -> Image.Image:
    """Devuelve una copia del arte con todas las cajas de texto compuestas."""
    out = base.convert("RGB").copy()
    for box in boxes or []:
        if box.get("text"):
            render_box(out, box, out.width, out.height)
    return out
