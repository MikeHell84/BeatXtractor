# -*- coding: utf-8 -*-
"""Genera screenshots auténticas usando el programa REAL capturado con la
plataforma nativa de Windows (tema real, no offscreen) y archivos de audio
reales (separación y división previamente realizadas por el propio programa).

Flujo:
  1. Carga _original.wav -> muestra forma de onda + análisis
  2. Simula el "separate" cargando los stems sep_*.wav en el path real del
     programa (_on_worker_done "separate") -> auto-ajuste de gain
  3. Simula el "split" cargando split_*.wav -> auto-ajuste
  4. Detecta beats (real) sobre un pieza
"""
import os
import sys
import time

import numpy as np
import soundfile as sf
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

import app as A
import audio_engine as engine

HERE = os.path.dirname(os.path.abspath(__file__))
SHOTS = os.path.join(HERE, "Screenshots")
SESSION = os.path.join(HERE, "sesion_redTormenthor_session")
TRUNC = 12.0  # segundos a capturar de cada archivo (screenshot legible y rápida)


def _load_short(path):
    """Carga un recorte corto del WAV real (primeros TRUNC segundos)."""
    info = sf.info(path)
    frames = min(info.frames, int(info.samplerate * TRUNC))
    with sf.SoundFile(path) as f:
        data = f.read(frames)  # (frames, channels)
    sr = info.samplerate
    if data.ndim == 1:
        data = np.vstack([data, data])
    else:
        data = data.T  # sf devuelve (samples, channels) -> (channels, samples)
    if data.shape[0] == 1:
        data = np.vstack([data, data])
    return data.astype(np.float32), sr


def grab(win, path, delay_ms=120):
    """Forced a pintura real antes de capturar."""
    QApplication.processEvents()
    time.sleep(delay_ms / 1000.0 + 0.05)
    QApplication.processEvents()
    win.grab().save(path, "PNG")
    print("  ->", os.path.relpath(path, HERE))


def load_real(win):
    audio, sr = _load_short(os.path.join(SESSION, "_original.wav"))
    bpm = engine.estimate_bpm(audio, sr)
    if not np.isfinite(bpm) or bpm <= 0:
        bpm = 120.0
    win.audio = audio
    win.sr = sr
    win.file_path = "redTormenthor.wav"
    win.current_bpm = float(bpm)
    win.separation = {}
    win.split_result = {}
    win._reset_playback()
    win.tracks.clear()
    win.track_list.clear()
    win._engine_dirty = True
    win.add_track("Original", audio)
    win.track_list.setCurrentRow(0)
    win._show_waveform(audio, sr)
    win._show_analysis({
        "bpm": float(bpm),
        "bands": engine.analyze_bands(audio, sr),
        "freqs": engine.dominant_frequencies(audio, sr),
    })
    return sr


def simulate_separate(win, sr):
    """Carga los stems ya separados por el programa (sep_*.wav) en el flujo real."""
    stem_files = {
        "drums":  os.path.join(SESSION, "sep_drums.wav"),
        "bass":   os.path.join(SESSION, "sep_bass.wav"),
        "vocals": os.path.join(SESSION, "sep_vocals.wav"),
        "other":  os.path.join(SESSION, "sep_other.wav"),
    }
    stems = {}
    for name, path in stem_files.items():
        a, _ = _load_short(path)
        stems[name] = a
    win.separation = stems
    win._busy_target = "separate"
    win._on_worker_done("separate", {
        "separation": stems,
        "stems": sorted(stems.keys()),
        "sr": sr,
    })
    win._busy_target = None
    # seleccionar la pista de batería
    idx = next(i for i, t in enumerate(win.tracks) if t.name.startswith("Drums"))
    win.track_list.setCurrentRow(idx)
    win.on_track_selected(idx)

    win._ensure_engine()
    win._show_waveform(engine.analysis_audio(win.tracks[idx]), sr)


def simulate_split(win, sr):
    """Carga las piezas ya divididas por el programa (split_*.wav)."""
    split_map = {
        "Bombo (Kick)":             "split_Bombo (Kick).wav",
        "Redoblante (Snare)":       "split_Redoblante (Snare).wav",
        "Hi-Hat":                   "split_Hi-Hat.wav",
        "Platillos (Crash/Ride)":   "split_Platillos (Crash_Ride).wav",
        "Tom Agudo":                "split_Tom Agudo.wav",
        "Tom Bajo":                 "split_Tom Bajo.wav",
        "Tom Medio":                "split_Tom Medio.wav",
    }
    parts = {}
    for name, fname in split_map.items():
        path = os.path.join(SESSION, fname)
        if os.path.exists(path):
            a, _ = _load_short(path)
            parts[name] = a
    win.split_result = parts
    win._busy_target = "split"
    win._on_worker_done("split", {"split": parts, "parts": sorted(parts.keys())})
    win._busy_target = None
    idx = next((i for i, t in enumerate(win.tracks)
                if t.name.startswith("Bombo")) if any(t.name.startswith("Bombo") for t in win.tracks)
               else 0, 0)
    win.track_list.setCurrentRow(idx)
    win.on_track_selected(idx)
    win._ensure_engine()
    win._show_waveform(engine.analysis_audio(win.tracks[idx]), sr)


def detect(win, sr, name):
    matches = [t for t in win.tracks if t.name == name]
    if not matches:
        matches = [t for t in win.tracks if name.split()[0] in t.name]
    if not matches:
        return None
    t = matches[0]
    idx = win.tracks.index(t)
    win.track_list.setCurrentRow(idx)
    win.on_track_selected(idx)
    hits = engine.detect_beats(
        engine.analysis_audio(t), sr,
        band=engine.band_for_name(t.name),
        threshold=0.10, wait_ms=25, fast=True,
    )
    win.track_hits[t.name] = (hits, engine.note_for_name(t.name))
    win.lbl_hits.setText(f"{len(hits)} hits detectados")
    win._draw_hits(hits)
    win._ensure_engine()
    return hits


def main():
    os.makedirs(SHOTS, exist_ok=True)
    app = A.QApplication([])

    win = A.MainWindow()
    win.resize(1280, 840)
    win.setWindowTitle("Bass & Drums Extractor")
    win.show()

    print("Captura 1: carga inicial (audio real)")
    sr = load_real(win)
    grab(win, os.path.join(SHOTS, "01-inicio.png"))

    print("Captura 2: separación Demucs (stems reales)")
    simulate_separate(win, sr)
    grab(win, os.path.join(SHOTS, "02-separacion.png"))

    print("Captura 3: división de batería (piezas reales)")
    simulate_split(win, sr)
    grab(win, os.path.join(SHOTS, "03-division.png"))

    print("Captura 4: detección de beats sobre el Bombo")
    detect(win, sr, "Bombo (Kick)")
    grab(win, os.path.join(SHOTS, "04-beats.png"))

    print("Captura 5: bajo (melódico)")
    detect(win, sr, "Bass (Demucs)")
    grab(win, os.path.join(SHOTS, "05-bajo.png"))

    print("OK")


if __name__ == "__main__":
    main()
