# -*- coding: utf-8 -*-
"""Genera el logo/icono de Bass & Drums Extractor.

Diseño: tambor de batería estilizado (parche circular con doble anillo de
"tensión" y soportes), ondas de audio saliendo a izquierda/derecha y la
etiqueta del instrumento. Salidas:
  - assets/logo.svg       (vectorial, fuente editable)
  - assets/logo.png       (PNG 512x512)
  - assets/icon.ico       (ICO multi-resolución para Windows)
  - assets/icon_256.png   (PNG 256x256 para el botón de ventana)
"""
import os
import sys
from PySide6.QtCore import QPointF, QRectF, Qt, QFile
from PySide6.QtGui import (
    QColor,
    QConicalGradient,
    QFont,
    QImage,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PySide6.QtSvg import QSvgGenerator

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")


# ---------------------------------------------------------------- colores
BG_TOP = QColor("#101626")
BG_BOTTOM = QColor("#1d2b4d")
DRUM_EDGE = QColor("#2c3e6b")
DRUM_LIGHT = QColor("#4a65a8")
PATCH = QColor("#e8edf7")
PATCH_DIM = QColor("#c6d2e8")
RIM = QColor("#7f96c7")
ACCENT = QColor("#ffb454")
WAVE = QColor("#6fd3e0")
STEM = QColor("#3f5388")
TEXT = QColor("#eef2fb")


def _drum_path(w, h):
    """Camino del cuerpo del tambor (cilindro con vista frontal)."""
    cx, cy = w * 0.5, h * 0.52
    r = w * 0.30
    p = QPainterPath()
    p.addEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))
    return p, cx, cy, r


def draw_logo(painter, w, h):
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

    # --- fondo degradado ---
    bg = QLinearGradient(0, 0, 0, h)
    bg.setColorAt(0.0, BG_TOP)
    bg.setColorAt(1.0, BG_BOTTOM)
    painter.fillRect(QRectF(0, 0, w, h), bg)

    # --- halo radial detrás del tambor ---
    halo = QRadialGradient(w * 0.5, h * 0.52, w * 0.42)
    halo.setColorAt(0.0, QColor(111, 211, 224, 55))
    halo.setColorAt(1.0, QColor(111, 211, 224, 0))
    painter.fillRect(QRectF(0, 0, w, h), halo)

    # --- ondas de audio (izquierda y derecha) ---
    wave_pen = QPen(WAVE, max(w * 0.012, 2.0))
    wave_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(wave_pen)
    for side in (-1, 1):
        x0 = w * (0.5 + side * 0.34)
        x1 = w * (0.5 + side * 0.46)
        for i, amp in enumerate((0.16, 0.24, 0.18, 0.26, 0.14)):
            y = h * (0.30 + i * 0.075)
            path = QPainterPath(QPointF(x0, y))
            midx = (x0 + x1) / 2
            path.cubicTo(midx, y - h * amp * 0.5, midx, y + h * amp * 0.5, x1, y)
            painter.drawPath(path)

    # --- cuerpo del tambor ---
    body, cx, cy, r = _drum_path(w, h)
    body_grad = QConicalGradient(QPointF(cx, cy), 30)
    body_grad.setColorAt(0.0, DRUM_LIGHT)
    body_grad.setColorAt(0.35, DRUM_EDGE)
    body_grad.setColorAt(0.65, DRUM_LIGHT)
    body_grad.setColorAt(1.0, DRUM_EDGE)
    painter.fillPath(body, body_grad)
    painter.setPen(QPen(QColor("#0c1220"), max(w * 0.008, 1.0)))
    painter.drawPath(body)

    # --- tornillos de tensión (tuning lugs) ---
    lug_pen = QPen(QColor("#9db4dd"), max(w * 0.014, 2.0))
    lug_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(lug_pen)
    n_lugs = 8
    for i in range(n_lugs):
        ang = 2.0 * 3.14159265 * i / n_lugs
        lx = cx + (r - w * 0.012) * __import__("math").cos(ang)
        ly = cy + (r - w * 0.012) * __import__("math").sin(ang)
        painter.drawPoint(QPointF(lx, ly))

    # --- parche (membrana) ---
    pr = r * 0.80
    patch_path = QPainterPath()
    patch_path.addEllipse(QRectF(cx - pr, cy - pr, pr * 2, pr * 2))
    pat = QLinearGradient(cx - pr, cy - pr, cx + pr, cy + pr)
    pat.setColorAt(0.0, PATCH)
    pat.setColorAt(0.55, PATCH_DIM)
    pat.setColorAt(1.0, PATCH)
    painter.fillPath(patch_path, pat)
    painter.setPen(QPen(RIM, max(w * 0.012, 2.0)))
    painter.drawPath(patch_path)

    # --- forma de onda sobre el parche (audio analizado) ---
    wave2 = QPen(ACCENT, max(w * 0.016, 2.2))
    wave2.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(wave2)
    seg = 2.0 * pr * 0.62
    start_x = cx - seg / 2
    # "golpe" de bombo: pico central con decaimiento a ambos lados
    ws = QPainterPath(QPointF(start_x, cy))
    N = 24
    for k in range(1, N + 1):
        fx = start_x + seg * k / N
        off = k / N - 0.5
        env = 1.0 - abs(off) * 1.6
        env = max(env, 0.05)
        amp = pr * 0.62 * env
        # oscilación grave tipo 60 Hz
        wob = __import__("math").sin(off * 3.14159265 * 6.0)
        fy = cy + amp * wob * 0.5
        ws.lineTo(QPointF(fx, fy))
    painter.drawPath(ws)

    # --- baquetas cruzadas detrás (opcional sutil) ---
    stick = QPen(STEM, max(w * 0.014, 2.0))
    stick.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(stick)
    s = w * 0.075
    painter.drawLine(QPointF(cx - r - s, cy - r - s), QPointF(cx - r * 0.2, cy + r * 0.35))
    painter.drawLine(QPointF(cx + r + s, cy - r - s), QPointF(cx + r * 0.2, cy + r * 0.35))

    # --- etiqueta inferior ---
    font = QFont("Segoe UI")
    font.setBold(True)
    font.setPixelSize(int(h * 0.052))
    painter.setFont(font)
    painter.setPen(TEXT)
    label = "BASS & DRUMS"
    fm = painter.fontMetrics()
    tw = fm.horizontalAdvance(label)
    painter.drawText(QRectF((w - tw) / 2, h * 0.82, tw, h * 0.12), Qt.AlignmentFlag.AlignCenter, label)

    sub_font = QFont("Segoe UI")
    sub_font.setPixelSize(int(h * 0.036))
    sub_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, int(w * 0.006))
    painter.setFont(sub_font)
    painter.setPen(QColor(238, 242, 251, 200))
    sub = "EXTRACTOR"
    sw = painter.fontMetrics().horizontalAdvance(sub)
    painter.drawText(QRectF((w - sw) / 2, h * 0.90, sw, h * 0.09), Qt.AlignmentFlag.AlignCenter, sub)


def render_svg(w, h, path):
    gen = QSvgGenerator()
    gen.setFileName(path)
    gen.setSize(__import__("PySide6.QtCore", fromlist=["QSize"]).QSize(w, h))
    gen.setViewBox(QRectF(0, 0, w, h))
    painter = QPainter(gen)
    draw_logo(painter, w, h)
    painter.end()


def render_png(w, h, path):
    img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    painter = QPainter(img)
    draw_logo(painter, w, h)
    painter.end()
    ok = img.save(path, "PNG")
    return img if ok else None


def _ico_bytes(images):
    """Construye un .ico multi-resolución a partir de QImages RGBA."""
    import struct
    from PySide6.QtCore import QBuffer, QIODevice

    blobs = []
    for img in images:
        buf = QBuffer()
        buf.open(QIODevice.OpenModeFlag.WriteOnly)
        img.save(buf, "PNG")
        png = bytes(buf.data())
        buf.close()
        blobs.append((img.width(), img.height(), png))

    header = struct.pack("<HHH", 0, 1, len(blobs))
    entries = b""
    offset = 6 + 16 * len(blobs)
    for (w, h, png) in blobs:
        entries += struct.pack(
            "<BBBBHHII",
            w & 0xFF if w < 256 else 0,
            h & 0xFF if h < 256 else 0,
            0,
            0,
            1,
            32,
            len(png),
            offset,
        )
        offset += len(png)
    payload = b"".join(b for (_, _, b) in blobs)
    return header + entries + payload


def make_ico(sizes=(16, 24, 32, 48, 64, 128, 256)):
    blobs = []
    for s in sizes:
        img = render_png(s, s, os.path.join(ASSETS, f"icon_{s}.png"))
        blobs.append(img)
    icon_path = os.path.join(ASSETS, "icon.ico")
    data = _ico_bytes(blobs)
    with open(icon_path, "wb") as f:
        f.write(data)
    # guardar también el PNG 256 como icon principal
    render_png(256, 256, os.path.join(ASSETS, "icon.png"))
    return icon_path, os.path.getsize(icon_path) > 0


def _icns_bytes(sizes_to_entries):
    """Construye un archivo .icns multi-resolución (Apple).

    sizes_to_entries: lista de (size, OSType, img)
      e.g. (16, b'ic04', img16)  (32, b'ic05', img32) ...
    """
    from PySide6.QtCore import QBuffer, QIODevice
    import struct
    header = b"\x00\x00\x01\x00"  # ICNS magic
    body = b""
    for (size, ostype, img) in sizes_to_entries:
        buf = QBuffer()
        buf.open(QIODevice.OpenModeFlag.WriteOnly)
        img.save(buf, "PNG")
        png = bytes(buf.data())
        buf.close()
        entry = ostype + struct.pack(">I", len(png) + 8) + png
        body += entry
    total = 8 + len(body)
    return header + struct.pack(">I", total) + body


def make_icns():
    """Genera assets/icon.icns para macOS desde los PNG ya renderizados."""
    icns_specs = [
        (16,  b"ic04"),
        (32,  b"ic05"),
        (32,  b"ic06"),   # 32  (fondo negro para retina) — uso genérico
        (48,  b"ic07"),
        (128, b"ic08"),
        (256, b"ic09"),
        (512, b"ic10"),
        (512, b"ic11"),   # 512  (retina 256)
        (1024,b"ic12"),   # 1024 (retina 512)
    ]
    seen = set()
    entries = []
    for (size, ostype) in icns_specs:
        if size in seen:
            # Renderizar a una resolución válida para este OSType
            pass
        if not os.path.exists(os.path.join(ASSETS, f"icon_{size}.png")):
            render_png(size, size, os.path.join(ASSETS, f"icon_{size}.png"))
        from PySide6.QtGui import QImage
        img = QImage(os.path.join(ASSETS, f"icon_{size}.png"))
        entries.append((size, ostype, img))
        seen.add(size)

    # Asegurar 512x512 y 1024x1024 para retina
    for size in (512, 1024):
        f = os.path.join(ASSETS, f"icon_{size}.png")
        if not os.path.exists(f):
            render_png(size, size, f)

    icns_path = os.path.join(ASSETS, "icon.icns")
    data = _icns_bytes(entries)
    with open(icns_path, "wb") as f:
        f.write(data)
    print(f"ICNS:  {icns_path} ({len(data)} bytes)")
    return icns_path, len(data) > 0


def main():
    from PySide6.QtWidgets import QApplication
    _app = QApplication([])
    os.makedirs(ASSETS, exist_ok=True)
    render_svg(512, 512, os.path.join(ASSETS, "logo.svg"))
    render_png(512, 512, os.path.join(ASSETS, "logo.png"))
    icon_path, ok = make_ico()
    render_png(512, 512, os.path.join(ASSETS, "icon_512.png"))
    render_png(1024, 1024, os.path.join(ASSETS, "icon_1024.png"))
    print(f"SVG:   {os.path.join(ASSETS, 'logo.svg')}")
    print(f"PNG:   {os.path.join(ASSETS, 'logo.png')}")
    print(f"ICO:   {icon_path} -> {ok}")
    icns_path, icns_ok = make_icns()
    print(f"ICNS:  {icns_path} -> {icns_ok}")
    print("OK")


if __name__ == "__main__":
    main()