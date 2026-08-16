# -*- coding: utf-8 -*-
"""Genera screenshots realistas de la app cargando audio sintético,
simulando separación Demucs, división de batería y detección de beats."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["TQDM_DISABLE"] = "1"

import numpy as np

import app as A
import audio_engine as engine

HERE = os.path.dirname(os.path.abspath(__file__))
SHOTS = os.path.join(HERE, "Screenshots")


def synth_mix(sr=22050, bpm=120, bars=8):
    beat = 60.0 / bpm
    total = beat * 4 * bars
    n = int(sr * total)
    t = np.arange(n) / sr
    drums = np.zeros(n)
    bass = np.zeros(n)
    other = np.zeros(n)

    rng = np.random.default_rng(42)

    def add_hit(buf, start_s, dur_s, freq, amp, decay=20.0):
        i0 = int(start_s * sr)
        i1 = min(n, int((start_s + dur_s) * sr))
        if i0 >= n:
            return
        seg = np.arange(i1 - i0) / sr
        env = np.exp(-seg * decay)
        tone = np.sin(2 * np.pi * freq * seg)
        buf[i0:i1] += amp * env * tone

    # --- batería: kick en cada beat, snare en 2 y 4, hats en corcheas ---
    for b in range(4 * bars):
        st = b * beat
        add_hit(drums, st, 0.25, 55, 0.9, 14.0)          # kick
        add_hit(drums, st, 0.35, 190, 0.5, 9.0)          # snare body
        for h in range(4):
            ht = st + h * (beat / 4)
            add_hit(drums, ht, 0.06, 6000 + rng.integers(500), 0.18, 55.0)
        if b % 2 == 0:
            add_hit(drums, st + beat * 0.5, 0.4, 165, 0.45, 8.0)  # snare offbeat
    for b in range(4 * bars):
        st = b * beat
        add_hit(drums, st + beat * 1.5, 0.3, 260, 0.4, 10.0)  # tom
        add_hit(drums, st + beat * 0.75, 0.5, 5000, 0.25, 30.0)  # crash

    # --- bajo siguiendo armonía ---
    chord = [55.0, 57.0, 49.0, 43.0]
    for b in range(4 * bars):
        f = chord[b % 4]
        st = b * beat
        seg = np.arange(int(beat * sr)) / sr
        idx = slice(int(st * sr), min(n, int((st + beat) * sr)))
        ln = idx.stop - idx.start
        seg = np.arange(ln) / sr
        bass[idx] += 0.5 * np.sin(2 * np.pi * f * seg) * np.exp(-seg * 1.5)
        bass[idx] += 0.22 * np.sin(2 * np.pi * f * 2 * seg) * np.exp(-seg * 1.5)

    # --- relleno: cuerdas/teclado ---
    pad = 0.06 * np.sin(2 * np.pi * 220 * t)
    pad += 0.05 * np.sin(2 * np.pi * 330 * t)
    pad += 0.04 * np.sin(2 * np.pi * 440 * t)
    other += pad

    def lim(x):
        m = np.max(np.abs(x))
        return x / m * 0.9 if m > 0 else x

    mix = lim(drums + bass + other * 1.0)
    return (
        np.vstack([mix, mix]).astype(np.float32),
        np.vstack([lim(drums), lim(drums)]).astype(np.float32),
        np.vstack([lim(bass), lim(bass)]).astype(np.float32),
        np.vstack([lim(other), lim(other)]).astype(np.float32),
        sr,
    )


def grab(win, path, wait=0):
    win.grab().save(path, "PNG")
    print("  ->", os.path.relpath(path, HERE))


def load_stage(win):
    mix, drums, bass, other, sr = synth_mix()
    win.audio = mix
    win.sr = sr
    win.file_path = "Grabación demo.wav"
    win.current_bpm = 120.0
    win.separation = {}
    win.split_result = {}
    win._reset_playback()
    win.tracks.clear()
    win.track_list.clear()
    win._engine_dirty = True
    win.add_track("Original", mix)
    win.track_list.setCurrentRow(0)
    win._show_waveform(mix, sr)
    win._show_analysis({"bpm": 120.0, "bands": [
        {"band": "Sub-bass", "pct": 34.1},
        {"band": "Bajo", "pct": 22.7},
        {"band": "Medios bajos", "pct": 17.3},
        {"band": "Medios", "pct": 13.9},
        {"band": "Agudos", "pct": 12.0},
    ], "freqs": [(55.0, 1.0), (110.0, 0.8), (165.0, 0.6), (220.0, 0.4), (440.0, 0.3)]})
    return mix, drums, bass, other, sr


def separate_stage(win, drums, bass, other, sr):
    win.separation = {"drums": drums, "bass": bass, "other": other}
    win.split_result = {}
    win.add_track("Drums (Demucs)", drums)
    win.add_track("Bass (Demucs)", bass)
    win.add_track("Other (Demucs)", other)
    win._mute_track_named("Original")
    engine.auto_adjust_tracks([t for t in win.tracks if not t.mute])
    # seleccionar drums
    idx = [t.name for t in win.tracks].index("Drums (Demucs)")
    win.track_list.setCurrentRow(idx)
    win.on_track_selected(idx)
    win._ensure_engine()
    win._show_waveform(engine.analysis_audio(win.tracks[idx]), sr)
    return idx


def split_stage(win, sr):
    src = [t for t in win.tracks if t.name == "Drums (Demucs)"][0]
    parts = engine.split_drums_advanced(src.audio, sr, {"sensitivity": 1.0, "gamma": 2.0, "smooth_ms": 60.0})
    names = sorted(parts.keys())
    for name in names:
        win.add_track(name, parts[name])
    win._mute_track_named("Drums")
    engine.auto_adjust_tracks([t for t in win.tracks if not t.mute])
    kick_name = "Bombo (Kick)" if "Bombo (Kick)" in [t.name for t in win.tracks] else "Kick"
    idx = [t.name for t in win.tracks].index(kick_name)
    win.track_list.setCurrentRow(idx)
    win.on_track_selected(idx)
    win._ensure_engine()
    win._show_waveform(engine.analysis_audio(win.tracks[idx]), sr)
    return idx


def detect_stage(win, sr, name):
    t = [t for t in win.tracks if t.name == name][0]
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
    win.show()

    print("Captura 1: carga inicial")
    mix, drums, bass, other, sr = load_stage(win)
    grab(win, os.path.join(SHOTS, "01-inicio.png"))

    print("Captura 2: separación Demucs")
    separate_stage(win, drums, bass, other, sr)
    grab(win, os.path.join(SHOTS, "02-separacion.png"))

    print("Captura 3: división de batería")
    split_stage(win, sr)
    grab(win, os.path.join(SHOTS, "03-division.png"))

    print("Captura 4: detección de beats")
    kick_name = "Bombo (Kick)" if "Bombo (Kick)" in [t.name for t in win.tracks] else "Kick"
    detect_stage(win, sr, kick_name)
    grab(win, os.path.join(SHOTS, "04-beats.png"))

    print("Captura 5: análisis de bajo (melódico)")
    if any("bass" in t.name.lower() for t in win.tracks):
        name = [t.name for t in win.tracks if "bass" in t.name.lower()][0]
        detect_stage(win, sr, name)
        grab(win, os.path.join(SHOTS, "05-bajo.png"))

    print("OK")


if __name__ == "__main__":
    main()