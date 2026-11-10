"""Layout de MOSS-Image: cajas en coordenadas de canvas → JSON que el modelo lee como prompt.

Adaptado de src/moss_studio/domain/layout.py del repo MOSS-Image-Colab (Apache-2.0).

Diferencia importante con el original (hallazgo propio, 2026-11-10):
en cajas `text`, si `caption` != `text`, el modelo PINTA la caption como texto fantasma.
Acá se fuerza `caption == text` para las cajas de texto. La descripción tipográfica va en
`summary.caption`, que es lo que el modelo usa como brief general.
"""

from __future__ import annotations

import json
import math

LABELS = frozenset({"text", "object", "character"})
DEFAULT_META = ("poster",)


class LayoutError(ValueError):
    pass


def _finite(value, what="coordenada") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise LayoutError(f"{what} tiene que ser un número finito")
    return float(value)


def check_canvas(width: int, height: int) -> None:
    for name, value in (("width", width), ("height", height)):
        if type(value) is not int or value <= 0 or value % 16:
            raise LayoutError(f"{name} debe ser entero positivo múltiplo de 16")


def normalized_bbox(values) -> list[float]:
    if not isinstance(values, (tuple, list)) or len(values) != 4:
        raise LayoutError("bbox normalizado debe tener 4 valores (x0, y0, x1, y1)")
    x0, y0, x1, y1 = (_finite(v) for v in values)
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise LayoutError("bbox fuera de 0..1, invertido o de área cero")
    return [x0, y0, x1, y1]


def to_centralized_yxyx(bbox_norm, width: int, height: int) -> list[int]:
    """0..1 normalizado → píxeles relativos al centro del lienzo, en orden yxyx.

    El modelo usa `abs_centralized`: (0,0) es el centro de la imagen, así que cada borde
    se corre medio lienzo. Es exactamente lo que hace el nodo MOSSImageLayoutPrompt.
    """
    x0, y0, x1, y1 = normalized_bbox(bbox_norm)
    px = [round(v * size) for v, size in ((x0, width), (y0, height), (x1, width), (y1, height))]
    if px[0] == px[2] or px[1] == px[3]:
        raise LayoutError("la caja mide menos de un píxel")
    return [px[1] - height // 2, px[0] - width // 2, px[3] - height // 2, px[2] - width // 2]


def _palette(value) -> list[str]:
    if (not isinstance(value, list) or not 1 <= len(value) <= 8
            or any(not isinstance(c, str) or not c.strip() or len(c) > 32 for c in value)):
        raise LayoutError("palette: 1 a 8 colores, texto de hasta 32 caracteres")
    return list(value)


def _meta(value) -> list[str]:
    if (not isinstance(value, list) or len(value) > 8
            or any(not isinstance(m, str) or not m.strip() or len(m) > 40 for m in value)):
        raise LayoutError("meta: hasta 8 etiquetas de hasta 40 caracteres")
    return list(value)


def compile_layout(caption: str, items: list[dict], width: int, height: int,
                   meta=None, force_caption_equal_text: bool = True) -> dict:
    """Arma el dict de layout listo para json.dumps.

    items: [{"label": "text"|"object"|"character",
             "text": "..."        (solo text),
             "caption": "..."     (opcional; en text se ignora si force_caption_equal_text),
             "bbox": (x0,y0,x1,y1) normalizado,
             "palette": [...]     (opcional)}]
    """
    check_canvas(width, height)
    if not isinstance(caption, str) or not caption.strip():
        raise LayoutError("hace falta el caption general de la imagen")
    if not isinstance(items, list) or not 1 <= len(items) <= 32:
        raise LayoutError("el layout necesita de 1 a 32 cajas")

    out = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise LayoutError(f"item {index}: tiene que ser un objeto")
        unknown = set(item) - {"label", "caption", "text", "palette", "bbox", "z_order"}
        if unknown:
            raise LayoutError(f"item {index}: campos desconocidos {sorted(unknown)}")
        label = item.get("label")
        if label not in LABELS:
            raise LayoutError(f"item {index}: label tiene que ser uno de {sorted(LABELS)}")

        entry = {"label": label}
        if label == "text":
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                raise LayoutError(f"item {index}: una caja de texto necesita `text`")
            desc = item.get("caption")
            if not force_caption_equal_text:
                if not isinstance(desc, str) or not desc.strip():
                    raise LayoutError(f"item {index}: falta `caption`")
                entry["caption"] = desc
            else:
                entry["caption"] = text
            entry["text"] = text
        else:
            desc = item.get("caption")
            if not isinstance(desc, str) or not desc.strip():
                raise LayoutError(f"item {index}: una caja de {label} necesita `caption`")
            if "text" in item:
                raise LayoutError(f"item {index}: solo las cajas de texto llevan `text`")
            entry["caption"] = desc

        if "palette" in item:
            entry["palette"] = _palette(item["palette"])
        entry["bbox"] = to_centralized_yxyx(item.get("bbox"), width, height)
        out.append(entry)

    return {
        "summary": {"meta": list(DEFAULT_META) if meta is None else _meta(list(meta)),
                    "canvas": "single", "caption": caption.strip()},
        "items": out,
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
    }


def dumps(layout: dict) -> str:
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":"))
