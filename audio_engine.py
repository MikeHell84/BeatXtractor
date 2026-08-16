import io
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
import librosa

from dataclasses import dataclass
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import uniform_filter1d


# -------------------------------------------------------------
# Bandas para análisis espectral
# -------------------------------------------------------------

BANDS = [
    ("Sub", 20, 60),
    ("Kick / Bass Drum", 40, 150),
    ("Bass fundamental", 40, 250),
    ("Low Mid", 250, 500),
    ("Snare body", 150, 350),
    ("Snare crack", 1200, 6000),
    ("Toms", 250, 1000),
    ("Cymbals", 6000, 16000),
    ("Air", 12000, 20000),
]


# -------------------------------------------------------------
# Utilidades básicas
# -------------------------------------------------------------

def ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg no está instalado o no está en el PATH. "
            "Es necesario para MP3/M4A/OPUS y otros formatos."
        )


def safe_filename(name, max_len=80):
    """Convierte cualquier nombre en uno válido para archivos."""
    name = str(name)
    # Caracteres prohibidos en Windows: \ / : * ? " < > |
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    # Espacios múltiples -> uno solo
    name = re.sub(r"\s+", " ", name).strip()
    # Sin puntos/espacios al final (problema clásico de Windows)
    name = name.strip(". ")
    if not name:
        name = "track"
    return name[:max_len]


def to_mono(audio):
    audio = np.asarray(audio)
    if audio.ndim == 1:
        return audio
    if audio.shape[0] <= audio.shape[1]:
        return audio.mean(axis=0)
    return audio.mean(axis=1)


def db_to_lin(db):
    return 10.0 ** (float(db) / 20.0)


def lin_to_db(x):
    return 20.0 * np.log10(max(float(x), 1e-12))


# -------------------------------------------------------------
# Carga / guardado
# -------------------------------------------------------------

def load_audio(path, sr=None):
    """
    Carga audio intentando soundfile primero.
    Si falla, usa FFmpeg como decodificador universal.
    Devuelve: audio shape (channels, samples), samplerate
    """
    try:
        y, samplerate = sf.read(path, always_2d=True, dtype="float32")
    except Exception:
        ensure_ffmpeg()
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", path,
            "-vn",
            "-acodec", "pcm_f32le",
            "-f", "wav",
            tmp.name,
        ]
        subprocess.run(cmd, check=True)

        y, samplerate = sf.read(tmp.name, always_2d=True, dtype="float32")
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

    y = y.T  # (samples, channels) -> (channels, samples)

    if sr is not None and samplerate != sr:
        y = librosa.resample(y, orig_sr=samplerate, target_sr=sr)
        samplerate = sr

    return y.astype(np.float32), samplerate


def resample_audio(audio, sr_in, sr_out):
    """Re-muestrea audio (1D mono o (channels, samples)) entre tasas."""
    a = np.asarray(audio, dtype=np.float32)
    if sr_in == sr_out:
        return a
    if a.ndim == 1:
        return librosa.resample(a, orig_sr=sr_in, target_sr=sr_out).astype(np.float32)
    out = np.stack(
        [librosa.resample(ch, orig_sr=sr_in, target_sr=sr_out).astype(np.float32) for ch in a]
    )
    return out


def save_audio(path, audio, sr, subtype=None):
    """
    Guarda audio en formato de alta calidad.
    WAV/FLAC/AIFF usan soundfile.
    MP3 usa FFmpeg a 320 kbps.
    """
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[None, :]

    ext = os.path.splitext(path)[-1].lower()

    if ext in {".wav", ".flac", ".aiff", ".aif"}:
        sub = subtype or "PCM_24"
        sf.write(path, audio.T, sr, subtype=sub)

    elif ext == ".ogg":
        sf.write(path, audio.T, sr, subtype="VORBIS")

    elif ext == ".mp3":
        ensure_ffmpeg()
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-f", "f32le",
            "-ar", str(sr),
            "-ac", str(audio.shape[0]),
            "-i", "-",
            "-b:a", "320k",
            path,
        ]
        subprocess.run(
            cmd,
            input=audio.T.astype("<f4").tobytes(),
            check=True,
        )

    else:
        sf.write(path, audio.T, sr, subtype=subtype or "PCM_24")


# -------------------------------------------------------------
# Análisis espectral
# -------------------------------------------------------------

def analyze_bands(audio, sr, n_fft=8192, max_seconds=30):
    mono = to_mono(audio)
    if len(mono) == 0:
        return []

    if max_seconds and len(mono) > sr * max_seconds:
        mono = mono[: sr * max_seconds]

    if len(mono) < n_fft:
        n_fft = max(256, 2 ** int(np.ceil(np.log2(len(mono)))))

    hop = n_fft // 4
    S = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    avg = S.mean(axis=1)
    total = avg.sum() + 1e-12

    result = []
    for name, lo, hi in BANDS:
        idx = (freqs >= lo) & (freqs < hi)
        energy = avg[idx].sum()
        result.append(
            {
                "band": name,
                "lo": float(lo),
                "hi": float(hi),
                "pct": float(100.0 * energy / total),
            }
        )

    return result


def dominant_frequencies(audio, sr, top=5, max_seconds=30):
    mono = to_mono(audio)
    if len(mono) == 0:
        return []

    if max_seconds and len(mono) > sr * max_seconds:
        mono = mono[: sr * max_seconds]

    n_fft = 8192
    if len(mono) < n_fft:
        n_fft = max(256, 2 ** int(np.ceil(np.log2(len(mono)))))

    S = np.abs(librosa.stft(mono, n_fft=n_fft))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    avg = S.mean(axis=1)

    mask = freqs >= 25.0
    freqs = freqs[mask]
    avg = avg[mask]

    idx = np.argsort(avg)[::-1][:top]
    return [(float(freqs[i]), float(avg[i])) for i in idx]


def estimate_bpm(audio, sr):
    mono = to_mono(audio)
    if len(mono) == 0:
        return 0.0

    if len(mono) > sr * 30:
        mono = mono[: sr * 30]

    tempo, _ = librosa.beat.beat_track(y=mono, sr=sr)
    return float(np.atleast_1d(tempo)[0])


# -------------------------------------------------------------
# Pista editable
# -------------------------------------------------------------

@dataclass
class Track:
    name: str
    audio: np.ndarray
    gain_db: float = 0.0
    pan: float = 0.0
    mute: bool = False
    solo: bool = False
    highpass: float = 0.0
    lowpass: float = 0.0
    gate_db: float = -90.0
    smooth_ms: float = 30.0
    fade_in_ms: float = 0.0
    fade_out_ms: float = 0.0
    processed: object = None          # bounce de los ajustes (para el detector)
    processed_stale: bool = False     # True si cambiaste perillas tras el bounce


# -------------------------------------------------------------
# Procesado manual de pista
# -------------------------------------------------------------

def _make_sos(hp, lp, sr):
    sections = []
    nyq = sr / 2.0

    if hp and hp > 10 and hp < nyq * 0.95:
        sections.append(butter(4, hp / nyq, btype="high", output="sos"))

    if lp and lp > 100 and lp < nyq * 0.95:
        sections.append(butter(4, lp / nyq, btype="low", output="sos"))

    if not sections:
        return None

    return np.vstack(sections)


def apply_filters(audio, sr, hp, lp):
    sos = _make_sos(hp, lp, sr)
    if sos is None:
        return audio

    try:
        return sosfiltfilt(sos, audio, axis=-1).astype(np.float32)
    except Exception:
        return audio


def apply_gate(audio, sr, threshold_db, smooth_ms=30):
    if threshold_db <= -89:
        return audio

    mono = to_mono(audio)
    if len(mono) == 0:
        return audio

    if np.max(np.abs(mono)) < 1e-8:
        return audio * 0.0

    hop = 512
    frame = 2048

    rms = librosa.feature.rms(
        y=mono,
        frame_length=frame,
        hop_length=hop
    )[0]

    if rms.max() < 1e-8:
        return audio * 0.0

    env_db = librosa.amplitude_to_db(rms, ref=float(rms.max()))
    gain = np.where(env_db > threshold_db, 1.0, 0.05)

    positions = np.arange(len(rms)) * hop
    gain_full = np.interp(np.arange(len(mono)), positions, gain)

    smooth = max(1, int((max(1.0, float(smooth_ms)) / 1000.0) * sr))
    if smooth > 1:
        gain_full = uniform_filter1d(gain_full, size=smooth, mode="nearest")

    return audio * gain_full


def apply_fades(audio, sr, fade_in_ms, fade_out_ms):
    n = audio.shape[-1]
    env = np.ones(n, dtype=np.float32)

    fi = min(int((fade_in_ms / 1000.0) * sr), n)
    fo = min(int((fade_out_ms / 1000.0) * sr), n)

    if fi > 1:
        env[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)

    if fo > 1:
        env[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)

    return audio * env


def render_for_analysis(track, sr):
    """Render de la pista CON sus ajustes (filtros/gate/ganancia/fades),
    sin pan ni mute: es lo que el detector debe analizar."""
    x = np.asarray(track.audio, dtype=np.float32).copy()
    x = apply_filters(x, sr, track.highpass, track.lowpass)
    x = apply_gate(x, sr, track.gate_db, track.smooth_ms)
    x = x * db_to_lin(track.gain_db)
    x = apply_fades(x, sr, track.fade_in_ms, track.fade_out_ms)
    return x.astype(np.float32)


def analysis_audio(track):
    """Audio que usan detector/MIDI/one-shots: el bounce si existe, si no el original."""
    return track.processed if track.processed is not None else track.audio


def apply_pan(audio, pan):
    pan = float(np.clip(pan, -1.0, 1.0))

    if audio.ndim == 1:
        audio = np.vstack([audio, audio])

    if audio.shape[0] == 1:
        audio = np.vstack([audio, audio])

    angle = (pan + 1.0) * np.pi / 4.0
    left = np.cos(angle)
    right = np.sin(angle)

    out = audio.copy()
    out[0] *= left
    out[1] *= right
    return out


def render_track(track, sr):
    """Render de la pista con sus ajustes (filtros/gate/ganancia/pan/fades).

    Importante: aquí NO se tiene en cuenta `mute`. El mute solo afecta a la
    mezcla (`mix_tracks`), que ya salta pistas silenciadas. Si `render_track`
    silenciase cada pista muteada, exportar un stem individual (p.ej. el
    "Drums (Demucs)" que se mutéa tras dividir para no duplicar en la mezcla)
    produciría un WAV en silencio. Para exportar individual siempre debe
    sonar el audio real de la pista.
    """
    x = np.asarray(track.audio, dtype=np.float32).copy()

    x = apply_filters(x, sr, track.highpass, track.lowpass)
    x = apply_gate(x, sr, track.gate_db, track.smooth_ms)
    x *= db_to_lin(track.gain_db)
    x = apply_pan(x, track.pan)
    x = apply_fades(x, sr, track.fade_in_ms, track.fade_out_ms)

    return x.astype(np.float32)


# -------------------------------------------------------------
# Mezcla final
# -------------------------------------------------------------

def soft_clip(x, threshold=0.95):
    y = x.copy()
    mask = np.abs(x) > threshold
    if np.any(mask):
        den = 1.0 - threshold
        y[mask] = np.sign(x[mask]) * (
            threshold + den * np.tanh((np.abs(x[mask]) - threshold) / den)
        )
    return y


def mix_tracks(tracks, sr, normalize_db=-0.3, limiter=True):
    if not tracks:
        return np.zeros((2, 1), dtype=np.float32)

    solo_any = any(t.solo for t in tracks)

    max_len = max((t.audio.shape[-1] for t in tracks), default=0)
    if max_len == 0:
        return np.zeros((2, 1), dtype=np.float32)

    mix = np.zeros((2, max_len), dtype=np.float64)

    for t in tracks:
        if t.mute:
            continue
        if solo_any and not t.solo:
            continue

        x = render_track(t, sr)

        if x.ndim == 1:
            x = np.vstack([x, x])

        if x.shape[0] == 1:
            x = np.vstack([x, x])

        if x.shape[0] > 2:
            x = x[:2]

        mix[:, : x.shape[-1]] += x

    if limiter:
        mix = soft_clip(mix)

    peak = np.max(np.abs(mix))
    if peak > 1e-8:
        mix *= db_to_lin(normalize_db) / peak

    return mix.astype(np.float32)


# -------------------------------------------------------------
# Separación IA con Demucs
# -------------------------------------------------------------

def separate_sources(
    path,
    model_name="htdemucs_ft",
    stems=("drums", "bass"),
    device="cuda",
):
    """
    Separa stems con Demucs.
    Devuelve dict: {'drums': audio, 'bass': audio}, sr
    """
    # Parche de consola (tqdm / descarga de modelos)
    if sys.stdout is None:
        sys.stdout = io.StringIO()
    if sys.stderr is None:
        sys.stderr = io.StringIO()
    os.environ["TQDM_DISABLE"] = "1"

    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = get_model(model_name)
    model.eval()
    model.to(device)

    audio, sr = librosa.load(path, sr=model.samplerate, mono=False)

    if audio.ndim == 1:
        audio = np.vstack([audio, audio])
    elif audio.shape[0] > 2:
        audio = np.mean(audio, axis=0, keepdims=True)
        audio = np.vstack([audio, audio])

    tensor = torch.from_numpy(audio).float()
    mean = tensor.mean()
    std = tensor.std() + 1e-8

    normalize = getattr(model, "normalize", True)
    inp = (tensor - mean) / std if normalize else tensor

    with torch.no_grad():
        mix = inp[None].to(device)

        try:
            sources = apply_model(
                model,
                mix,
                device=device,
                split=True,
                overlap=0.25,
            )[0]
        except TypeError:
            try:
                sources = apply_model(
                    model,
                    mix,
                    device=device,
                    splits=True,
                    overlap=0.25,
                )[0]
            except TypeError:
                sources = apply_model(model, mix, device=device)[0]

    if normalize:
        sources = sources * std + mean

    out = {}
    for name, src in zip(model.sources, sources):
        if name in stems:
            out[name] = src.detach().cpu().numpy().astype(np.float32)

    return out, model.samplerate


# -------------------------------------------------------------
# División de batería en Kick / Snare / Cymbals
# -------------------------------------------------------------

def split_drums(drums_audio, sr, params=None):
    """
    Divide un stem de batería en:
      - Kick / Bass Drum
      - Snare
      - Cymbals

    Usa máscaras tiempo-frecuencia basadas en energía por banda,
    onsets y suavizado temporal.
    """
    p = {
        "kick_lo": 20,
        "kick_hi": 150,
        "snare_lo": 150,
        "snare_hi": 350,
        "crack_lo": 1200,
        "crack_hi": 6000,
        "cym_lo": 6000,
        "cym_hi": 16000,
        "sharpness": 2.0,
        "smooth_ms": 80.0,
        "gate_db": -45.0,
        "transient_weight": 1.0,
    }

    if params:
        p.update(params)

    drums_audio = np.asarray(drums_audio, dtype=np.float32)
    if drums_audio.ndim == 1:
        drums_audio = np.vstack([drums_audio, drums_audio])

    n_samples = drums_audio.shape[-1]
    mono = to_mono(drums_audio)

    n_fft = 4096
    hop = 1024

    S_mono = librosa.stft(mono, n_fft=n_fft, hop_length=hop)
    mag = np.abs(S_mono) + 1e-10
    frames = mag.shape[1]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    onset = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop)

    if len(onset) < frames:
        onset = np.pad(onset, (0, frames - len(onset)), mode="constant")
    else:
        onset = onset[:frames]

    onset = onset / (onset.max() + 1e-8)

    smooth_frames = max(3, int((float(p["smooth_ms"]) / 1000.0) * sr / hop))

    def smooth_time(x):
        return uniform_filter1d(x, size=smooth_frames, mode="nearest")

    def band_energy(lo, hi):
        idx = (freqs >= lo) & (freqs <= hi)
        if not idx.any():
            return np.zeros(frames, dtype=np.float32)
        return mag[idx, :].sum(axis=0)

    total = mag.sum(axis=0) + 1e-10

    e_kick = band_energy(p["kick_lo"], p["kick_hi"]) / total
    e_snare = (
        band_energy(p["snare_lo"], p["snare_hi"])
        + 0.7 * band_energy(p["crack_lo"], p["crack_hi"])
    ) / total
    e_cym = band_energy(p["cym_lo"], p["cym_hi"]) / total

    e_kick = smooth_time(e_kick)
    e_snare = smooth_time(e_snare)
    e_cym = smooth_time(e_cym)
    onset_s = smooth_time(onset)

    sharp = float(p.get("sharpness", 2.0))
    tw = float(p.get("transient_weight", 1.0))

    m_kick_t = (e_kick ** sharp) * (0.45 + 0.55 * onset_s) * tw
    m_snare_t = (e_snare ** sharp) * (0.35 + 0.65 * onset_s) * tw
    m_cym_t = (e_cym ** sharp) * (0.85 + 0.15 * onset_s)

    tsum = m_kick_t + m_snare_t + m_cym_t + 1e-8
    limit = np.minimum(1.0, 1.0 / tsum)

    m_kick_t *= limit
    m_snare_t *= limit
    m_cym_t *= limit

    gate_thresh = float(np.clip(db_to_lin(float(p.get("gate_db", -45.0))), 0.0, 0.2))

    def gate_time(m):
        return np.where(m > gate_thresh, m, m * 0.12)

    m_kick_t = gate_time(m_kick_t)
    m_snare_t = gate_time(m_snare_t)
    m_cym_t = gate_time(m_cym_t)

    prof_kick = (
        (freqs >= p["kick_lo"]) & (freqs <= (p["kick_hi"] + 120))
    ).astype(np.float32)

    prof_snare = (
        ((freqs >= p["snare_lo"]) & (freqs <= p["snare_hi"]))
        | ((freqs >= p["crack_lo"]) & (freqs <= p["crack_hi"]))
    ).astype(np.float32)

    prof_cym = (freqs >= p["cym_lo"]).astype(np.float32)

    m_kick = m_kick_t[None, :] * prof_kick[:, None]
    m_snare = m_snare_t[None, :] * prof_snare[:, None]
    m_cym = m_cym_t[None, :] * prof_cym[:, None]

    m_all = m_kick + m_snare + m_cym + 1e-8

    m_kick /= m_all
    m_snare /= m_all
    m_cym /= m_all

    result = {}

    for name, mask in [
        ("Kick", m_kick),
        ("Snare", m_snare),
        ("Cymbals", m_cym),
    ]:
        chans = []
        for ch in range(drums_audio.shape[0]):
            S = librosa.stft(drums_audio[ch], n_fft=n_fft, hop_length=hop)

            if S.shape[1] != mask.shape[1]:
                min_t = min(S.shape[1], mask.shape[1])
                S_use = S[:, :min_t]
                mask_use = mask[:, :min_t]
            else:
                S_use = S
                mask_use = mask

            y = librosa.istft(
                S_use * mask_use,
                hop_length=hop,
                length=n_samples,
            )
            chans.append(y)

        result[name] = np.vstack(chans).astype(np.float32)

    return result


# -------------------------------------------------------------
# Piezas de batería soportadas por el divisor automático
# -------------------------------------------------------------

DRUM_PIECES = [
    {"name": "Bombo (Kick)", "bands": [(20, 150, 1.0), (150, 300, 0.25), (2000, 6000, 0.10)],
     "ct": 0.85, "cs": 0.15, "min_share": 0.02, "min_presence": 0.005,
     "decay_s": 0.18, "thr": 0.08},
    {"name": "Redoblante (Snare)", "bands": [(150, 350, 1.0), (1200, 6000, 0.8)],
     "ct": 0.75, "cs": 0.25, "min_share": 0.02, "min_presence": 0.005,
     "decay_s": 0.15, "thr": 0.08},
    {"name": "Hi-Hat", "bands": [(6000, 16000, 1.0)],
     "ct": 0.90, "cs": 0.10, "min_share": 0.01, "min_presence": 0.005,
     "decay_s": 0.06, "thr": 0.07},
    {"name": "Platillos (Crash/Ride)", "bands": [(5000, 16000, 1.0)],
     "ct": 0.15, "cs": 0.85, "min_share": 0.01, "min_presence": 0.004,
     "decay_s": 0.60, "thr": 0.12},
    {"name": "Tom Agudo", "bands": [(180, 420, 1.0)],
     "den_bands": [(20, 100, 1.0), (1000, 6000, 0.7)], "ratio_pow": 1.5,
     "ct": 0.7, "cs": 0.3, "min_share": 0.01, "min_presence": 0.004,
     "decay_s": 0.30, "thr": 0.10},
    {"name": "Tom Medio", "bands": [(120, 260, 1.0)],
     "den_bands": [(20, 100, 1.0), (1000, 6000, 0.7)], "ratio_pow": 1.5,
     "ct": 0.7, "cs": 0.3, "min_share": 0.01, "min_presence": 0.004,
     "decay_s": 0.35, "thr": 0.10},
    {"name": "Tom Bajo", "bands": [(70, 160, 1.0)],
     "den_bands": [(20, 60, 1.0), (1000, 6000, 0.7)], "ratio_pow": 1.5,
     "ct": 0.7, "cs": 0.3, "min_share": 0.01, "min_presence": 0.004,
     "decay_s": 0.40, "thr": 0.10},
]


ORCHESTRA_PIECES = [
    {"name": "Contrabajos", "bands": [(28, 120, 1.0), (120, 250, 0.4)],
     "ct": 0.2, "cs": 0.8, "min_share": 0.02, "min_presence": 0.01},
    {"name": "Tuba", "bands": [(30, 220, 1.0)], "den_bands": [(120, 500, 0.4)],
     "ratio_pow": 1.2, "ct": 0.3, "cs": 0.7, "min_share": 0.015, "min_presence": 0.008},
    {"name": "Timbales", "bands": [(40, 200, 1.0)], "den_bands": [(200, 1200, 0.5)],
     "ratio_pow": 1.3, "ct": 0.8, "cs": 0.2, "min_share": 0.01, "min_presence": 0.004},
    {"name": "Cellos", "bands": [(60, 500, 1.0), (500, 1200, 0.4)],
     "den_bands": [(28, 60, 0.6)], "ratio_pow": 1.2,
     "ct": 0.2, "cs": 0.8, "min_share": 0.02, "min_presence": 0.01},
    {"name": "Fagotes", "bands": [(60, 900, 1.0), (900, 2600, 0.5)],
     "den_bands": [(28, 60, 0.5)], "ratio_pow": 1.2,
     "ct": 0.3, "cs": 0.7, "min_share": 0.012, "min_presence": 0.006},
    {"name": "Trombones", "bands": [(80, 800, 1.0)], "den_bands": [(28, 80, 0.5)],
     "ratio_pow": 1.2, "ct": 0.4, "cs": 0.6, "min_share": 0.012, "min_presence": 0.006},
    {"name": "Trompas", "bands": [(100, 1000, 1.0), (1000, 3000, 0.4)],
     "den_bands": [(28, 100, 0.5)], "ratio_pow": 1.2,
     "ct": 0.3, "cs": 0.7, "min_share": 0.012, "min_presence": 0.006},
    {"name": "Clarinete", "bands": [(130, 1500, 1.0), (1500, 4000, 0.4)],
     "den_bands": [(28, 130, 0.5)], "ratio_pow": 1.2,
     "ct": 0.3, "cs": 0.7, "min_share": 0.012, "min_presence": 0.006},
    {"name": "Trompetas", "bands": [(160, 1200, 1.0), (1200, 6000, 0.6)],
     "ct": 0.5, "cs": 0.5, "min_share": 0.012, "min_presence": 0.006},
    {"name": "Violas", "bands": [(200, 1200, 1.0)],
     "den_bands": [(28, 200, 0.5), (1200, 4000, 0.4)], "ratio_pow": 1.2,
     "ct": 0.2, "cs": 0.8, "min_share": 0.015, "min_presence": 0.008},
    {"name": "Oboes", "bands": [(250, 2000, 1.0), (2000, 4000, 0.5)],
     "den_bands": [(28, 250, 0.5)], "ratio_pow": 1.2,
     "ct": 0.3, "cs": 0.7, "min_share": 0.012, "min_presence": 0.006},
    {"name": "Flautas", "bands": [(250, 2500, 1.0), (2500, 8000, 0.6)],
     "den_bands": [(28, 250, 0.5)], "ratio_pow": 1.2,
     "ct": 0.2, "cs": 0.8, "min_share": 0.012, "min_presence": 0.006},
    {"name": "Violines", "bands": [(196, 1000, 0.7), (1000, 6000, 1.0)],
     "den_bands": [(28, 196, 0.5)], "ratio_pow": 1.2,
     "ct": 0.2, "cs": 0.8, "min_share": 0.02, "min_presence": 0.01},
    {"name": "Percusión / Platillos", "bands": [(3000, 16000, 1.0)],
     "ct": 0.3, "cs": 0.7, "min_share": 0.01, "min_presence": 0.004},
]


def split_by_configs(audio_in, sr, configs, params=None):
    """
    Divide audio según perfiles espectrales competitivos,
    detectando automáticamente qué piezas/instrumentos están presentes.
    """
    p = {
        "sharpness": 2.0,
        "smooth_ms": 60.0,
        "gate_db": -48.0,
        "fast_ms": 30.0,
        "slow_ms": 400.0,
        "sensitivity": 1.0,
    }
    if params:
        p.update(params)

    audio = np.asarray(audio_in, dtype=np.float32)
    if audio.ndim == 1:
        audio = np.vstack([audio, audio])

    n_samples = audio.shape[-1]
    mono = to_mono(audio)

    n_fft = 4096
    hop = 1024

    S_mono = librosa.stft(mono, n_fft=n_fft, hop_length=hop)
    mag = np.abs(S_mono) + 1e-10
    frames = mag.shape[1]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    total = mag.sum(axis=0) + 1e-10

    fast_frames = max(2, int((p["fast_ms"] / 1000.0) * sr / hop))
    slow_frames = max(4, int((p["slow_ms"] / 1000.0) * sr / hop))
    smooth_frames = max(2, int((p["smooth_ms"] / 1000.0) * sr / hop))

    def smooth(x, size):
        return uniform_filter1d(x, size=size, mode="nearest")

    def band_energy(lo, hi):
        idx = (freqs >= lo) & (freqs <= hi)
        if not idx.any():
            return np.zeros(frames, dtype=np.float64)
        return mag[idx, :].sum(axis=0)

    sharp = float(p["sharpness"])
    sens = max(0.25, float(p["sensitivity"]))
    gate_lin = float(np.clip(db_to_lin(float(p["gate_db"])), 0.0, 0.2))

    detected = []

    for cfg in configs:
        e = np.zeros(frames, dtype=np.float64)
        prof = np.zeros(len(freqs), dtype=np.float32)

        for lo, hi, w in cfg["bands"]:
            e += w * band_energy(lo, hi)
            idx = (freqs >= lo) & (freqs <= hi)
            prof[idx] = np.maximum(prof[idx], w)

        share = float(e.sum() / (total.sum() + 1e-10))

        fast = smooth(e, fast_frames)
        slow = smooth(e, slow_frames)
        trans = np.clip(fast - slow, 0.0, None)
        trans_n = trans / (trans.max() + 1e-10)
        sus_n = slow / (slow.max() + 1e-10)

        if "den_bands" in cfg:
            e_den = np.zeros(frames, dtype=np.float64)
            for lo, hi, w in cfg["den_bands"]:
                e_den += w * band_energy(lo, hi)
            base = (e / (e_den + 1e-6 * total + 1e-10)) ** cfg.get("ratio_pow", 1.5)
        else:
            base = smooth(e / total, smooth_frames) ** sharp

        base = base / (base.max() + 1e-10)

        t = base * (cfg["ct"] * trans_n + cfg["cs"] * sus_n + 1e-3)
        t = t / (t.max() + 1e-10)

        presence = float(np.mean(t > 0.15))

        if share >= cfg["min_share"] / sens and presence >= cfg["min_presence"] / sens:
            t = np.where(t > gate_lin, t, t * 0.10)
            detected.append(
                {
                    "name": cfg["name"],
                    "t": t,
                    "prof": prof,
                    "share": share,
                    "presence": presence,
                }
            )

    if not detected:
        return {"Batería (completa)": audio}

    # Máscaras competitivas: la suma nunca supera 1
    sum2d = np.zeros((len(freqs), frames), dtype=np.float64)
    for d in detected:
        sum2d += d["t"][None, :] * d["prof"][:, None]
    sum2d += 1e-8

    result = {}
    for d in detected:
        mask = (d["t"][None, :] * d["prof"][:, None]) / sum2d
        chans = []
        for ch in range(audio.shape[0]):
            S = librosa.stft(audio[ch], n_fft=n_fft, hop_length=hop)
            if S.shape[1] != mask.shape[1]:
                mt = min(S.shape[1], mask.shape[1])
                S_use, mask_use = S[:, :mt], mask[:, :mt]
            else:
                S_use, mask_use = S, mask
            y = librosa.istft(S_use * mask_use, hop_length=hop, length=n_samples)
            chans.append(y)
        result[d["name"]] = np.vstack(chans).astype(np.float32)

    return result


def split_drums_full(drums_audio, sr, params=None):
    return split_drums_advanced(drums_audio, sr, params)


def split_drums_advanced(drums_audio, sr, params=None):
    """
    Separación de batería por eventos + plantillas espectrales aprendidas
    + enmascarado Wiener con residual. Mucho más limpio que el reparto
    continuo de energía: cada pieza suena solo cuando suena.
    """
    p = {"sensitivity": 1.0, "gamma": 2.0, "residual": 0.2}
    if params:
        p.update(params)

    audio = np.asarray(drums_audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = np.vstack([audio, audio])
    n_samples = audio.shape[-1]
    mono = to_mono(audio)

    n_fft = 2048
    hop = 512
    S = librosa.stft(mono, n_fft=n_fft, hop_length=hop)
    mag = np.abs(S) + 1e-10
    n_f, frames = mag.shape
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    total = mag.sum(axis=0)

    sens = max(0.25, float(p["sensitivity"]))
    gamma = float(p["gamma"])

    comps = []
    for cfg in DRUM_PIECES:
        e = np.zeros(frames, dtype=np.float64)
        for lo, hi, w in cfg["bands"]:
            idx = (freqs >= lo) & (freqs <= hi)
            if idx.any():
                e += w * mag[idx, :].sum(axis=0)
        share = float(e.sum() / (total.sum() + 1e-10))

        flux = np.clip(np.diff(e, prepend=e[0]), 0.0, None)
        flux_n = flux / (flux.max() + 1e-10)
        avg_frames = max(16, int(0.4 * sr / hop))
        local = uniform_filter1d(flux_n, size=avg_frames, mode="nearest")
        delta = (cfg.get("thr", 0.10) / sens) * (0.25 + local)
        wait = max(1, int((wait_for_name(cfg["name"]) / 1000.0) * sr / hop))

        peaks = _pick_peaks(flux_n, 1, 2, 4, 8, delta, wait)
        if len(peaks) < 3 or share < cfg.get("min_share", 0.01) / sens:
            continue

        vel = np.clip(flux_n[peaks], 0.05, 1.0)

        # Plantilla espectral = incremento medio de espectro en cada golpe
        acc = np.zeros(n_f, dtype=np.float64)
        for pk in peaks:
            b0, b1 = pk, min(frames, pk + 2)
            a0, a1 = max(0, pk - 3), pk
            if a1 > a0:
                inc = mag[:, b0:b1].mean(axis=1) - mag[:, a0:a1].mean(axis=1)
            else:
                inc = mag[:, b0:b1].mean(axis=1)
            acc += np.clip(inc, 0.0, None)
        T = uniform_filter1d(acc / len(peaks), size=9, mode="nearest")
        T = T / (T.max() + 1e-10)

        # Activación temporal con decaimiento propio de la pieza
        K = max(3, int(cfg.get("decay_s", 0.15) * sr / hop))
        kern = np.exp(-np.linspace(0.0, 3.0, K))
        act = np.zeros(frames, dtype=np.float64)
        for pk, v in zip(peaks, vel):
            seg = min(K, frames - pk)
            if seg > 0:
                act[pk:pk + seg] += v * kern[:seg]

        comps.append({"name": cfg["name"], "T": T, "act": act})

    if not comps:
        return {"Batería (completa)": audio}

    # Máscaras Wiener + residual (la energía no explicada no se asigna)
    pows = []
    for c in comps:
        M = c["act"][None, :] * c["T"][:, None]     # (freqs, frames)
        pows.append(M ** gamma)
    res = (float(p["residual"]) * total / (float(total.mean()) + 1e-10))[None, :] * np.ones((n_f, 1))
    res **= gamma
    denom = sum(pows) + res + 1e-10

    out = {}
    for c, pw in zip(comps, pows):
        mask = pw / denom
        chans = []
        for ch in range(audio.shape[0]):
            Sc = librosa.stft(audio[ch], n_fft=n_fft, hop_length=hop)
            if Sc.shape[1] != mask.shape[1]:
                mt = min(Sc.shape[1], mask.shape[1])
                Sc, mask_use = Sc[:, :mt], mask[:, :mt]
            else:
                mask_use = mask
            y = librosa.istft(Sc * mask_use, hop_length=hop, length=n_samples)
            chans.append(y)
        out[c["name"]] = np.vstack(chans).astype(np.float32)

    return out


def split_orchestra(audio, sr, params=None):
    return split_by_configs(audio, sr, ORCHESTRA_PIECES, params)


# -------------------------------------------------------------
# Auto-ajuste de pistas
# -------------------------------------------------------------

def auto_adjust_tracks(tracks):
    for t in tracks:
        peak = float(np.max(np.abs(t.audio))) if t.audio.size else 0.0
        if peak > 1e-6:
            target = db_to_lin(-1.0)
            t.gain_db = float(np.clip(20.0 * np.log10(target / peak), -18.0, 12.0))
        else:
            t.gain_db = 0.0

        name = t.name.lower()

        if "kick" in name or "bombo" in name:
            t.highpass, t.lowpass = 25.0, 300.0
            t.gate_db, t.smooth_ms = -45.0, 80.0
            t.fade_out_ms = 80.0

        elif "snare" in name or "redoblante" in name:
            t.highpass, t.lowpass = 110.0, 10000.0
            t.gate_db, t.smooth_ms = -45.0, 60.0
            t.fade_out_ms = 80.0

        elif "hi-hat" in name or "hihat" in name:
            t.highpass, t.lowpass = 4500.0, 16000.0
            t.gate_db, t.smooth_ms = -50.0, 50.0
            t.fade_out_ms = 120.0

        elif "platillo" in name or "crash" in name or "ride" in name or "cymbal" in name:
            t.highpass, t.lowpass = 3500.0, 18000.0
            t.gate_db, t.smooth_ms = -52.0, 140.0
            t.fade_out_ms = 250.0

        elif "tom" in name:
            t.highpass = 60.0
            if "agudo" in name:
                t.lowpass = 700.0
            elif "medio" in name:
                t.lowpass = 500.0
            else:
                t.lowpass = 320.0
            t.gate_db, t.smooth_ms = -48.0, 90.0
            t.fade_out_ms = 150.0

        elif "contrabajo" in name:
            t.highpass, t.lowpass = 25.0, 1500.0
            t.gate_db, t.smooth_ms = -60.0, 60.0

        elif "tuba" in name:
            t.highpass, t.lowpass = 25.0, 1200.0
            t.gate_db, t.smooth_ms = -60.0, 60.0

        elif "timbales" in name:
            t.highpass, t.lowpass = 30.0, 500.0
            t.gate_db, t.smooth_ms = -50.0, 120.0

        elif "cello" in name or "violonchelo" in name:
            t.highpass, t.lowpass = 50.0, 5000.0
            t.gate_db, t.smooth_ms = -60.0, 50.0

        elif "viola" in name:
            t.highpass, t.lowpass = 150.0, 7000.0
            t.gate_db, t.smooth_ms = -60.0, 50.0

        elif "violin" in name:
            t.highpass, t.lowpass = 180.0, 14000.0
            t.gate_db, t.smooth_ms = -60.0, 50.0

        elif "flauta" in name:
            t.highpass, t.lowpass = 250.0, 10000.0
            t.gate_db, t.smooth_ms = -60.0, 50.0

        elif "oboe" in name:
            t.highpass, t.lowpass = 250.0, 8000.0
            t.gate_db, t.smooth_ms = -60.0, 50.0

        elif "clarinete" in name:
            t.highpass, t.lowpass = 130.0, 8000.0
            t.gate_db, t.smooth_ms = -60.0, 50.0

        elif "fagot" in name:
            t.highpass, t.lowpass = 50.0, 4000.0
            t.gate_db, t.smooth_ms = -60.0, 50.0

        elif "trompeta" in name:
            t.highpass, t.lowpass = 150.0, 10000.0
            t.gate_db, t.smooth_ms = -55.0, 50.0

        elif "trombon" in name:
            t.highpass, t.lowpass = 70.0, 4000.0
            t.gate_db, t.smooth_ms = -55.0, 50.0

        elif "trompa" in name:
            t.highpass, t.lowpass = 90.0, 6000.0
            t.gate_db, t.smooth_ms = -55.0, 50.0

        elif "percusi" in name or "platillo" in name:
            t.highpass, t.lowpass = 2500.0, 18000.0
            t.gate_db, t.smooth_ms = -55.0, 100.0

        elif "bass" in name or "bajo" in name:
            t.highpass, t.lowpass = 25.0, 9000.0
            t.gate_db, t.smooth_ms = -60.0, 40.0

        elif "drum" in name or "bater" in name:
            t.highpass, t.lowpass = 20.0, 18000.0
            t.gate_db, t.smooth_ms = -55.0, 50.0

        else:
            t.highpass, t.lowpass = 20.0, 19000.0
            t.gate_db, t.smooth_ms = -70.0, 25.0


# -------------------------------------------------------------
# Guardado / carga de sesión
# -------------------------------------------------------------

def save_session(path, audio, sr, tracks, separation=None, split_result=None):
    """
    Guarda la sesión completa en un archivo .json.
    Los audios se escriben como .wav junto al .json.
    """
    import json

    base = os.path.splitext(path)[0]
    folder = base + "_session"
    os.makedirs(folder, exist_ok=True)

    def write_wav(tag, data):
        audio = np.asarray(data, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[None, :]
        fname = os.path.join(folder, f"{tag}.wav")
        sf.write(fname, audio.T, sr, subtype="PCM_24")
        return fname

    payload = {
        "sr": int(sr),
        "tracks": [],
        "separation": {},
        "split_result": {},
        "audio_file": None,
    }

    if audio is not None and audio.size:
        payload["audio_file"] = write_wav("_original", audio)

    for i, t in enumerate(tracks):
        fname = write_wav(f"track_{i}", t.audio)
        payload["tracks"].append(
            {
                "name": t.name,
                "gain_db": t.gain_db,
                "pan": t.pan,
                "mute": t.mute,
                "solo": t.solo,
                "highpass": t.highpass,
                "lowpass": t.lowpass,
                "gate_db": t.gate_db,
                "smooth_ms": t.smooth_ms,
                "fade_in_ms": t.fade_in_ms,
                "fade_out_ms": t.fade_out_ms,
                "file": fname,
            }
        )

    for key, data in (separation or {}).items():
        if data is not None:
            payload["separation"][key] = write_wav(f"sep_{safe_filename(key)}", data)

    for key, data in (split_result or {}).items():
        if data is not None:
            payload["split_result"][key] = write_wav(f"split_{safe_filename(key)}", data)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_session(path):
    """
    Carga una sesión guardada con save_session.
    Devuelve dict con audio, sr, tracks, separation y split_result.
    """
    import json

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    sr = int(payload.get("sr", 44100))
    folder = os.path.splitext(path)[0] + "_session"

    def read_wav(fname):
        if not fname or not os.path.exists(fname):
            return None
        y, _ = sf.read(fname, always_2d=True, dtype="float32")
        return y.T.astype(np.float32)

    audio = read_wav(payload.get("audio_file"))

    tracks = []
    for td in payload.get("tracks", []):
        t = Track(
            name=td["name"],
            audio=read_wav(td["file"]),
            gain_db=td.get("gain_db", 0.0),
            pan=td.get("pan", 0.0),
            mute=td.get("mute", False),
            solo=td.get("solo", False),
            highpass=td.get("highpass", 0.0),
            lowpass=td.get("lowpass", 0.0),
            gate_db=td.get("gate_db", -90.0),
            smooth_ms=td.get("smooth_ms", 30.0),
            fade_in_ms=td.get("fade_in_ms", 0.0),
            fade_out_ms=td.get("fade_out_ms", 0.0),
        )
        if t.audio is not None:
            tracks.append(t)

    separation = {}
    for key, fname in payload.get("separation", {}).items():
        data = read_wav(fname)
        if data is not None:
            separation[key] = data

    split_result = {}
    for key, fname in payload.get("split_result", {}).items():
        data = read_wav(fname)
        if data is not None:
            split_result[key] = data

    return {
        "audio": audio,
        "sr": sr,
        "tracks": tracks,
        "separation": separation,
        "split_result": split_result,
    }


# -------------------------------------------------------------
# Detección de beats y exportación MIDI (compatible con
# Addictive Drums, Superior Drummer, EZdrummer, etc.)
# -------------------------------------------------------------

DETECT_BANDS = {
    "bombo": (20, 150), "kick": (20, 150),
    "redoblante": (150, 350), "snare": (150, 350),
    "hi-hat": (6000, 16000), "hihat": (6000, 16000),
    "platillo": (5000, 16000), "crash": (5000, 16000),
    "ride": (5000, 16000), "cymbal": (5000, 16000),
    "tom agudo": (180, 420), "tom medio": (120, 260), "tom bajo": (70, 160),
    "bass": (30, 250), "bajo": (30, 250),
}

DEFAULT_NOTES = {
    "bombo": 36, "kick": 36,
    "redoblante": 38, "snare": 38,
    "hi-hat": 42, "hihat": 42,
    "platillo": 49, "crash": 49, "ride": 51, "cymbal": 49,
    "tom agudo": 50, "tom medio": 47, "tom bajo": 45,
}

DEFAULT_WAIT = {
    "bombo": 25, "kick": 25,
    "redoblante": 25, "snare": 25,
    "hi-hat": 20, "hihat": 20,
    "platillo": 150, "crash": 150, "ride": 150, "cymbal": 150,
    "tom agudo": 40, "tom medio": 40, "tom bajo": 40,
    "bass": 60, "bajo": 60,
}


def _match_key(name):
    n = str(name).lower()
    for k in [
        "tom agudo", "tom medio", "tom bajo",
        "bombo", "kick", "redoblante", "snare",
        "hi-hat", "hihat", "platillo", "crash", "ride", "cymbal",
        "bass", "bajo",
    ]:
        if k in n:
            return k
    return None


def band_for_name(name):
    k = _match_key(name)
    return DETECT_BANDS.get(k, (20, 16000))


def note_for_name(name):
    k = _match_key(name)
    if k is None:
        return None
    return DEFAULT_NOTES.get(k)


def wait_for_name(name):
    return style_wait_for(name)


# ------------------------------------------------------------------
# Estilos de batería (presets de detección)
# ------------------------------------------------------------------

DRUM_STYLES = {
    "slow": {
        "label": "Lento (pop / rock / funk)",
        "fast": False,
        "threshold": 0.10,
        "wait": {
            "bombo": 90, "kick": 90,
            "redoblante": 70, "snare": 70,
            "hi-hat": 40, "hihat": 40,
            "platillo": 150, "crash": 150, "ride": 150, "cymbal": 150,
            "tom agudo": 90, "tom medio": 90, "tom bajo": 90,
            "bass": 90, "bajo": 90,
        },
    },
    "metal": {
        "label": "Rápido (metal / blast beats)",
        "fast": True,
        "threshold": 0.08,
        "wait": {
            "bombo": 25, "kick": 25,
            "redoblante": 25, "snare": 25,
            "hi-hat": 20, "hihat": 20,
            "platillo": 150, "crash": 150, "ride": 150, "cymbal": 150,
            "tom agudo": 40, "tom medio": 40, "tom bajo": 40,
            "bass": 60, "bajo": 60,
        },
    },
}

current_style = "metal"


def set_drum_style(style):
    global current_style
    if style in DRUM_STYLES:
        current_style = style


def style_fast():
    return DRUM_STYLES[current_style]["fast"]


def style_threshold():
    return DRUM_STYLES[current_style]["threshold"]


def style_wait_for(name):
    k = _match_key(name)
    return DRUM_STYLES[current_style]["wait"].get(k, 80)


def _pick_peaks(x, pre_max, post_max, pre_avg, post_avg, delta, wait):
    """Peak picking con umbral adaptativo (delta puede ser array)."""
    n = len(x)
    delta_arr = np.asarray(delta, dtype=np.float64)
    if delta_arr.ndim == 0:
        delta_arr = np.full(n, float(delta_arr))

    peaks = []
    last = -10**9

    for i in range(n):
        v = x[i]

        lo = max(0, i - pre_max)
        hi = min(n, i + post_max + 1)
        if v < np.max(x[lo:hi]):
            continue

        lo = max(0, i - pre_avg)
        hi = min(n, i + post_avg + 1)
        if v <= np.mean(x[lo:hi]) + delta_arr[i]:
            continue

        if i - last < wait:
            continue

        peaks.append(i)
        last = i

    return peaks


def detect_beats(audio, sr, band=(20, 150), threshold=0.10,
                 wait_ms=30, smooth_ms=5, fast=True):
    """
    Detector tipo trigger de alta precisión:
    - hop pequeño (256) para golpes muy rápidos (doble bombo / blast),
    - flux de energía en banda (ataque),
    - umbral adaptativo según dinámica de la zona,
    - retrigger configurable.
    Devuelve [(tiempo_seg, velocity), ...]
    """
    mono = to_mono(audio)
    if len(mono) == 0:
        return []

    hop = 256 if fast else 512
    n_fft = 1024 if fast else 2048

    S = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    m = (freqs >= band[0]) & (freqs <= band[1])
    if not m.any():
        return []

    e = S[m, :].sum(axis=0)

    # Detector de ataque: diferencia positiva de energía
    flux = np.clip(np.diff(e, prepend=e[0]), 0.0, None)

    sm = max(1, int((smooth_ms / 1000.0) * sr / hop))
    if sm > 1:
        flux = uniform_filter1d(flux, size=sm, mode="nearest")

    flux_n = flux / (flux.max() + 1e-10)

    # Umbral adaptativo: se ajusta a partes suaves y fuertes
    avg_frames = max(16, int(0.4 * sr / hop))
    local = uniform_filter1d(flux_n, size=avg_frames, mode="nearest")
    delta_arr = threshold * (0.25 + local)

    wait_frames = max(1, int((wait_ms / 1000.0) * sr / hop))

    peaks = _pick_peaks(
        flux_n,
        pre_max=1,
        post_max=2,
        pre_avg=4,
        post_avg=8,
        delta=delta_arr,
        wait=wait_frames,
    )

    hits = []
    for i in peaks:
        t = float(i) * hop / sr
        vel = int(20 + float(np.clip(flux_n[i], 0, 1)) * 107)
        hits.append((t, max(1, min(127, vel))))

    return hits


def detect_bass_notes(audio, sr, min_dur_ms=70, gap_ms=40, prob=0.2):
    """
    Detecta notas reales del bajo (altura, inicio, duración, dinámica)
    con pyin. Devuelve [(start_seg, dur_seg, midi_note, velocity), ...]
    """
    mono = to_mono(audio)
    if len(mono) == 0:
        return []

    hop = 256
    f0, voiced, vprob = librosa.pyin(
        mono, fmin=25, fmax=600, sr=sr,
        hop_length=hop, frame_length=4096,
    )

    n = len(f0)
    midi_f = np.full(n, np.nan, dtype=np.float64)
    valid = np.isfinite(f0) & (f0 > 0)
    midi_f[valid] = 69.0 + 12.0 * np.log2(f0[valid] / 440.0)

    active = voiced & (vprob >= prob) & np.isfinite(midi_f)

    gap_frames = max(1, int((gap_ms / 1000.0) * sr / hop))
    min_frames = max(2, int((min_dur_ms / 1000.0) * sr / hop))

    raw = []
    i = 0
    while i < n:
        if not active[i]:
            i += 1
            continue

        k = i
        gap = 0
        last_v = i
        while k < n:
            if active[k]:
                last_v = k
                gap = 0
            else:
                gap += 1
                if gap > gap_frames:
                    break
            k += 1

        j = last_v + 1
        if j - i >= min_frames:
            seg = midi_f[i:j][active[i:j]]
            if len(seg) > 0:
                pitch = int(round(float(np.median(seg))))
                if 24 <= pitch <= 72:
                    seg_a = mono[i * hop: j * hop]
                    rms = float(np.sqrt(np.mean(seg_a ** 2) + 1e-12))
                    raw.append([i, j, pitch, rms])
        i = k + 1

    if not raw:
        return []

    vmax = max(r[3] for r in raw) + 1e-12
    notes = []
    for i0, j0, pitch, rms in raw:
        vel = int(min(127, 20 + (rms / vmax) * 107))
        notes.append((i0 * hop / sr, (j0 - i0) * hop / sr, pitch, vel))

    return notes


def _vlq(value):
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(out))


def write_midi(path, events_by_channel, bpm=120.0, tpq=480, gate_s=0.12):
    """
    events_by_channel: { canal(1-16): { nota: [ (t, vel) o (t, dur, vel) ] } }
    Canal 10 = batería GM, canal 1 = bajo, etc.
    """
    events = []
    for ch, notes in events_by_channel.items():
        on = 0x90 | (int(ch) - 1)
        off = 0x80 | (int(ch) - 1)
        for note, evs in notes.items():
            for ev in evs:
                if len(ev) == 2:
                    t, vel = ev
                    dur = gate_s
                else:
                    t, dur, vel = ev
                events.append((t, on, int(note), int(vel)))
                events.append((t + dur, off, int(note), 0))

    events.sort(key=lambda e: e[0])

    secs_per_tick = 60.0 / (float(bpm) * tpq)

    body = bytearray()
    uspq = int(round(60_000_000 / float(bpm)))
    body += _vlq(0) + bytes([0xFF, 0x51, 0x03]) + uspq.to_bytes(3, "big")

    last_tick = 0
    for t, status, d1, d2 in events:
        tick = int(round(t / secs_per_tick))
        if tick < last_tick:
            tick = last_tick
        body += _vlq(tick - last_tick) + bytes([status, d1, d2])
        last_tick = tick

    body += _vlq(0) + bytes([0xFF, 0x2F, 0x00])

    with open(path, "wb") as f:
        f.write(b"MThd" + (6).to_bytes(4, "big"))
        f.write((0).to_bytes(2, "big"))
        f.write((1).to_bytes(2, "big"))
        f.write(tpq.to_bytes(2, "big"))
        f.write(b"MTrk" + len(body).to_bytes(4, "big"))
        f.write(bytes(body))


# ------------------------------------------------------------------
# One-shots (kit de samples WAV)
# ------------------------------------------------------------------

ONE_SHOT_DUR = {
    "bombo": 300, "kick": 300,
    "redoblante": 350, "snare": 350,
    "hi-hat": 180, "hihat": 180,
    "platillo": 1200, "crash": 1200, "ride": 1200, "cymbal": 1200,
    "tom agudo": 450, "tom medio": 500, "tom bajo": 600,
}


def one_shot_dur_for(name):
    k = _match_key(name)
    return ONE_SHOT_DUR.get(k, 400)


def slice_one_shots(audio, sr, hits, pre_ms=5, dur_ms=400, fade_ms=15):
    """
    Corta cada golpe detectado como un sample independiente (one-shot),
    con un poco de pre-roll y fade de salida para evitar clicks.
    Devuelve [(segmento_audio, velocity), ...]
    """
    n = audio.shape[-1]
    pre = int((pre_ms / 1000.0) * sr)
    fade = int((fade_ms / 1000.0) * sr)
    min_len = int(0.02 * sr)

    outs = []
    for t, vel in hits:
        start = int(t * sr) - pre
        end = int(t * sr) + int((dur_ms / 1000.0) * sr)
        start = max(0, start)
        end = min(n, end)
        if end - start < min_len:
            continue

        if audio.ndim == 1:
            seg = audio[start:end].astype(np.float32).copy()[None, :]
        else:
            seg = audio[:, start:end].astype(np.float32).copy()
        if fade > 1 and seg.shape[-1] > fade:
            env = np.ones(seg.shape[-1], dtype=np.float32)
            env[-fade:] = np.linspace(1.0, 0.0, fade, dtype=np.float32)
            seg *= env
        outs.append((seg, int(vel)))

    return outs
