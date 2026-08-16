import io
import math
import os
import sys
import tempfile
import threading

import numpy as np
import sounddevice as sd
from scipy.signal import sosfilt

# ---------------------------------------------------------------------
# Parche anti-error de consola:
# Evita "AttributeError: 'NoneType' object has no attribute 'write'"
# que produce tqdm al descargar modelos sin terminal visible.
# ---------------------------------------------------------------------
if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

os.environ["TQDM_DISABLE"] = "1"

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QPointF, QRectF, QSize
from PySide6.QtGui import (
    QAction,
    QFont,
    QIcon,
    QPainter,
    QPen,
    QColor,
    QBrush,
    QLinearGradient,
    QRadialGradient,
    QPainterPath,
    QPixmap,
    QShortcut,
    QKeySequence,
)
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QSlider,
    QSplitter,
    QMessageBox,
    QProgressBar,
    QStatusBar,
    QInputDialog,
)

import pyqtgraph as pg

import audio_engine as engine


# ------------------------------------------------------------------
# Icono de la aplicación
# ------------------------------------------------------------------

def load_app_icon():
    """Carga el icono de la app desde assets/ con fallback a un icono
    dibujado en memoria si el archivo no existe."""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    ico = os.path.join(base, "icon.ico")
    if os.path.exists(ico):
        icon = QIcon(ico)
        if not icon.isNull():
            return icon
    png = os.path.join(base, "icon.png")
    if os.path.exists(png):
        icon = QIcon(png)
        if not icon.isNull():
            return icon
    return _fallback_icon()


def _fallback_icon():
    """Icono simple dibujado en memoria (tambor + onda de audio)."""
    pm = QPixmap(64, 64)
    pm.fill(QColor(16, 22, 38))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(232, 237, 247))
    p.setPen(QPen(QColor(127, 150, 199), 2))
    p.drawEllipse(QRectF(14, 14, 36, 36))
    p.setPen(QPen(QColor(255, 180, 84), 3))
    for i, amp in enumerate((0.5, 0.8, 0.6)):
        y = 32 + (i - 1) * 10
        p.drawLine(24, int(y), 40, int(y))
    p.end()
    return QIcon(pm)


# ------------------------------------------------------------------
# Candado global de reproducción: solo puede existir UN stream vivo.
# ------------------------------------------------------------------

_ACTIVE_ENGINES = []


def stop_all_audio():
    """Detiene cualquier reproducción activa (motores y sounddevice)."""
    try:
        sd.stop()
    except Exception:
        pass
    for e in list(_ACTIVE_ENGINES):
        try:
            e.stop()
        except Exception:
            pass


# ------------------------------------------------------------------
# Hilo de procesamiento (evita bloquear la UI)
# ------------------------------------------------------------------

class Worker(QThread):
    done = Signal(str, object)
    error = Signal(str)
    progress = Signal(int, str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            result = self._fn(self.progress)
            self.done.emit("ok", result)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")


class BassWorker(QThread):
    done = Signal(list)
    error = Signal(str)

    def __init__(self, fn, *args):
        super().__init__()
        self.fn = fn
        self.args = args

    def run(self):
        try:
            self.done.emit(self.fn(*self.args))
        except Exception as e:
            self.error.emit(str(e))


# ------------------------------------------------------------------
# Perilla rotativa (knob)
# ------------------------------------------------------------------

class Knob(QWidget):
    valueChanged = Signal(float)

    def __init__(self, vmin, vmax, value=0.0, title="", fmt="{:.0f}", parent=None):
        super().__init__(parent)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self._value = float(value)
        self._default = float(value)
        self.title = title
        self.fmt = fmt
        self.setFixedSize(86, 110)     # tamaño fijo: nada se solapa nunca
        self._drag = None

    def value(self):
        return self._value

    def setValue(self, v):
        v = max(self.vmin, min(self.vmax, float(v)))
        if v != self._value:
            self._value = v
            self.update()

    def sizeHint(self):
        return QSize(86, 110)

    def mousePressEvent(self, ev):
        self._drag = (ev.position().y(), self._value)

    def mouseMoveEvent(self, ev):
        if self._drag is None:
            return
        dy = self._drag[0] - ev.position().y()
        rng = self.vmax - self.vmin
        self._value = max(self.vmin, min(self.vmax, self._drag[1] + dy * rng / 200.0))
        self.update()
        self.valueChanged.emit(self._value)

    def mouseReleaseEvent(self, ev):
        self._drag = None

    def mouseDoubleClickEvent(self, ev):
        self._value = self._default
        self.update()
        self.valueChanged.emit(self._value)

    def wheelEvent(self, ev):
        step = (self.vmax - self.vmin) / 100.0
        d = step if ev.angleDelta().y() > 0 else -step
        self._value = max(self.vmin, min(self.vmax, self._value + d))
        self.update()
        self.valueChanged.emit(self._value)

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        cx = w / 2.0
        cy = 36.0          # centro del círculo, fijo
        r = 26.0           # radio fijo
        rect = QRectF(cx - r, cy - r, 2 * r, 2 * r)

        # --- arco de fondo ---
        p.setPen(QPen(QColor("#252B3B"), 6, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(rect, 225 * 16, -270 * 16)

        # --- arco de valor ---
        frac = (self._value - self.vmin) / (self.vmax - self.vmin)
        grad = QLinearGradient(cx - r, 0, cx + r, 0)
        grad.setColorAt(0.0, QColor("#7C4DFF"))
        grad.setColorAt(1.0, QColor("#00C2FF"))
        p.setPen(QPen(QBrush(grad), 6, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(rect, 225 * 16, int(-270 * 16 * frac))

        # --- aguja ---
        ang = math.radians(225 - 270 * frac)
        p.setPen(QPen(QColor("#E8ECF1"), 2))
        p.drawLine(
            QPointF(cx, cy),
            QPointF(cx + math.cos(ang) * (r - 6), cy - math.sin(ang) * (r - 6)),
        )

        # --- VALOR (zona fija 68-84 px) ---
        p.setPen(QColor("#E8ECF1"))
        p.setFont(QFont("Segoe UI", 8, QFont.Bold))
        p.drawText(QRectF(0, 68, w, 16), Qt.AlignCenter, self.fmt.format(self._value))

        # --- TÍTULO (zona fija 88-104 px, recortado si es largo) ---
        p.setPen(QColor("#9FB2FF"))
        p.setFont(QFont("Segoe UI", 7))
        title = p.fontMetrics().elidedText(self.title, Qt.ElideRight, w - 6)
        p.drawText(QRectF(0, 88, w, 16), Qt.AlignCenter, title)

        p.end()


# ------------------------------------------------------------------
# Motor de reproducción en tiempo real
# ------------------------------------------------------------------

class RealtimeEngine:
    def __init__(self, sr):
        self.sr = sr
        self.stream = None
        self.lock = threading.Lock()
        self.ets = []
        self.pos = 0
        self.maxlen = 0

    def _static_buf(self, t):
        x = np.asarray(t.audio, dtype=np.float32)
        x = engine.apply_gate(x, self.sr, t.gate_db, t.smooth_ms)
        x = engine.apply_fades(x, self.sr, t.fade_in_ms, t.fade_out_ms)
        if x.ndim == 1:
            x = np.vstack([x, x])
        if x.shape[0] == 1:
            x = np.vstack([x, x])
        return x[:2].astype(np.float32)

    def set_tracks(self, tracks):
        with self.lock:
            self.ets = []
            for t in tracks:
                sos = engine._make_sos(t.highpass, t.lowpass, self.sr)
                self.ets.append({
                    "track": t,
                    "buf": self._static_buf(t),
                    "sos": sos,
                    "zi": [np.zeros((sos.shape[0], 2)) for _ in range(2)] if sos is not None else None,
                })
            self.maxlen = max((et["buf"].shape[1] for et in self.ets), default=0)
        self.pos = 0

    def refresh_static(self, t):
        try:
            buf = self._static_buf(t)
        except Exception:
            return
        with self.lock:
            for et in self.ets:
                if et["track"] is t:
                    et["buf"] = buf
            self.maxlen = max((et["buf"].shape[1] for et in self.ets), default=0)

    def refresh_filters(self, t):
        with self.lock:
            for et in self.ets:
                if et["track"] is t:
                    sos = engine._make_sos(t.highpass, t.lowpass, self.sr)
                    et["sos"] = sos
                    et["zi"] = [np.zeros((sos.shape[0], 2)) for _ in range(2)] if sos is not None else None

    def play(self, start=None):
        stop_all_audio()
        if not self.ets or self.maxlen == 0:
            return
        if start is None:
            self.pos = 0
        else:
            self.pos = max(0, min(int(start), self.maxlen - 1))
        with self.lock:
            for et in self.ets:
                if et["zi"] is not None:
                    et["zi"] = [np.zeros(z.shape) for z in et["zi"]]
        self.stream = sd.OutputStream(
            samplerate=self.sr, channels=2, blocksize=1024, callback=self._cb
        )
        self.stream.start()
        _ACTIVE_ENGINES.append(self)

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        if self in _ACTIVE_ENGINES:
            _ACTIVE_ENGINES.remove(self)

    def _cb(self, outdata, frames, time_info, status):
        start = self.pos
        end = start + frames
        acc = np.zeros((frames, 2), dtype=np.float32)

        with self.lock:
            solo_any = any(et["track"].solo for et in self.ets)
            single = len(self.ets) <= 1
            for et in self.ets:
                t = et["track"]
                if not single and (t.mute or (solo_any and not t.solo)):
                    continue
                buf = et["buf"]
                n = buf.shape[1]
                if start >= n:
                    continue
                m = min(frames, n - start)
                seg = np.ascontiguousarray(buf[:, start:start + m]).T.copy()

                sos = et["sos"]
                if sos is not None:
                    for ch in range(2):
                        y, zf = sosfilt(sos, seg[:, ch], zi=et["zi"][ch])
                        seg[:, ch] = y
                        et["zi"][ch] = zf

                g = 10.0 ** (t.gain_db / 20.0)
                ang = (max(-1.0, min(1.0, t.pan)) + 1) * np.pi / 4
                gl = g * math.cos(ang)
                gr = g * math.sin(ang)
                acc[:m, 0] += seg[:, 0] * gl
                acc[:m, 1] += seg[:, 1] * gr

        outdata[:] = acc
        self.pos = end
        if end >= self.maxlen:
            raise sd.CallbackStop


# ------------------------------------------------------------------
# Workers auxiliares
# ------------------------------------------------------------------

class FnWorker(QThread):
    done = Signal(object)
    error = Signal(str)

    def __init__(self, fn, *args):
        super().__init__()
        self.fn = fn
        self.args = args

    def run(self):
        try:
            self.done.emit(self.fn(*self.args))
        except Exception as e:
            self.error.emit(str(e))


def _build_midi_events(items):
    """items: (name, audio, sr, kind, note, thr, wait, fast, stored_hits)"""
    events = {}
    for name, audio, sr, kind, note, thr, wait, fast, stored in items:
        if kind == "bass":
            ch = events.setdefault(1, {})
            for s, d, p, v in engine.detect_bass_notes(audio, sr):
                ch.setdefault(p, []).append((s, d, v))
        else:
            ch = events.setdefault(10, {})
            hits = stored if stored else engine.detect_beats(
                audio, sr, band=engine.band_for_name(name),
                threshold=thr, wait_ms=wait, fast=fast,
            )
            ch.setdefault(note, []).extend(hits)
    return events


# ------------------------------------------------------------------
# Ventana principal
# ------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bass & Drums Extractor")
        self.setWindowIcon(load_app_icon())
        self.resize(1280, 840)

        self.audio = None
        self.sr = None
        self.file_path = None
        self.tracks = []
        self.split_result = {}
        self.separation = {}
        self.track_hits = {}
        self.current_bpm = 120.0
        self.hit_scatter = None
        self.bass_worker = None

        self._play_index = None
        self._playing = False
        self.engine = None
        self._engine_dirty = True
        self._editing = None

        self._play_timer = QTimer(self)
        self._play_timer.setInterval(40)
        self._play_timer.timeout.connect(self._tick_playhead)

        self._static_timer = QTimer(self)
        self._static_timer.setSingleShot(True)
        self._static_timer.setInterval(120)
        self._static_timer.timeout.connect(self._flush_static)
        self._static_track = None
        self._static_pending = False
        self._static_worker = None

        self._build_ui()
        self._make_menu()
        self._space_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self._space_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self._space_shortcut.activated.connect(self.toggle_playback)
        self.statusBar().showMessage("Listo")

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)

        top = QHBoxLayout()
        self.btn_open = QPushButton("Abrir audio…")
        self.btn_open.clicked.connect(self.open_file)

        self.btn_separate = QPushButton("Separar con Demucs")
        self.btn_separate.setEnabled(False)
        self.btn_separate.clicked.connect(self.run_separation)

        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Pop/Rock (Demucs)", "Sinfónico (orquesta)"])
        self.combo_mode.currentIndexChanged.connect(self._on_mode_change)

        self.btn_split = QPushButton("Dividir batería")
        self.btn_split.setEnabled(False)
        self.btn_split.clicked.connect(self.run_drum_split)

        self.btn_export = QPushButton("Exportar mezcla…")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_mix)

        self.btn_stems = QPushButton("Exportar stems…")
        self.btn_stems.setEnabled(False)
        self.btn_stems.clicked.connect(self.export_stems)

        self.btn_play = QPushButton("▶ Escuchar")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self.toggle_playback)

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setVisible(False)

        top.addWidget(self.btn_open)
        top.addWidget(self.btn_separate)
        top.addWidget(self.combo_mode)
        top.addWidget(self.btn_split)
        top.addWidget(self.btn_export)
        top.addWidget(self.btn_stems)
        top.addWidget(self.btn_play)
        top.addStretch(1)
        top.addWidget(self.progress)
        root.addLayout(top)

        splitter = QSplitter(Qt.Horizontal)

        # --- Panel izquierdo: pistas ---
        left = QWidget()
        lv = QVBoxLayout(left)

        self.track_list = QListWidget()
        self.track_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.track_list.currentRowChanged.connect(self.on_track_selected)
        self.track_list.itemSelectionChanged.connect(self._on_selection_changed)
        self.track_list.itemDoubleClicked.connect(self.rename_track)
        lv.addWidget(QLabel("Pistas:"))
        lv.addWidget(self.track_list, 1)

        props = QGroupBox("Propiedades de la pista")
        grid = QGridLayout(props)
        grid.setHorizontalSpacing(10)

        self.knob_gain = Knob(-40.0, 40.0, 0.0, "Gain", "{:.1f} dB")
        self.knob_gain.valueChanged.connect(self.on_gain)

        self.knob_pan = Knob(-1.0, 1.0, 0.0, "Pan", "{:.2f}")
        self.knob_pan.valueChanged.connect(self.on_pan)

        self.knob_hp = Knob(0.0, 20000.0, 0.0, "High-pass", "{:.0f} Hz")
        self.knob_hp.valueChanged.connect(self.on_hp)

        self.knob_lp = Knob(0.0, 24000.0, 0.0, "Low-pass", "{:.0f} Hz")
        self.knob_lp.valueChanged.connect(self.on_lp)

        self.knob_gate = Knob(-90.0, 0.0, -90.0, "Gate", "{:.1f} dB")
        self.knob_gate.valueChanged.connect(self.on_gate)

        self.knob_smooth = Knob(1.0, 1000.0, 50.0, "Suavizado", "{:.0f} ms")
        self.knob_smooth.valueChanged.connect(self.on_smooth)

        self.knob_fade_in = Knob(0.0, 10000.0, 0.0, "Fade in", "{:.0f} ms")
        self.knob_fade_in.valueChanged.connect(self.on_fade_in)

        self.knob_fade_out = Knob(0.0, 10000.0, 0.0, "Fade out", "{:.0f} ms")
        self.knob_fade_out.valueChanged.connect(self.on_fade_out)

        self.ch_mute = QCheckBox("Mute")
        self.ch_mute.toggled.connect(self.on_mute)
        self.ch_solo = QCheckBox("Solo")
        self.ch_solo.toggled.connect(self.on_solo)

        knobs = [
            self.knob_gain, self.knob_pan, self.knob_hp, self.knob_lp,
            self.knob_gate, self.knob_smooth, self.knob_fade_in, self.knob_fade_out,
        ]
        for i, k in enumerate(knobs[:4]):
            grid.addWidget(k, 1, i, Qt.AlignmentFlag.AlignCenter)
        for i, k in enumerate(knobs[4:]):
            grid.addWidget(k, 2, i, Qt.AlignmentFlag.AlignCenter)

        mut_solo = QHBoxLayout()
        mut_solo.addWidget(self.ch_mute)
        mut_solo.addWidget(self.ch_solo)
        mut_solo.addStretch(1)
        grid.addLayout(mut_solo, 3, 0, 1, 4)

        self.btn_auto = QPushButton("Auto-ajustar")
        self.btn_auto.clicked.connect(self.auto_adjust)
        self.btn_del = QPushButton("Eliminar pista")
        self.btn_del.clicked.connect(self.delete_track)
        grid.addWidget(self.btn_auto, 4, 0)
        grid.addWidget(self.btn_del, 4, 1)

        self.btn_commit = QPushButton("Renderizar ajustes → detector")
        self.btn_commit.setObjectName("primary")
        self.btn_commit.clicked.connect(self.commit_processed)
        self.btn_revert = QPushButton("Usar audio original")
        self.btn_revert.clicked.connect(self.revert_processed)
        grid.addWidget(self.btn_commit, 5, 0, 1, 2)
        grid.addWidget(self.btn_revert, 5, 2, 1, 2)

        lv.addWidget(props)

        # --- Panel: división de batería ---
        dbox = QGroupBox("División de batería")
        dgrid = QGridLayout(dbox)

        self.knob_sens = Knob(0.5, 3.0, 1.0, "Sensibilidad", "{:.1f}")
        self.knob_sharp = Knob(1.0, 6.0, 2.0, "Agresividad", "{:.1f}")
        self.knob_split_smooth = Knob(10.0, 300.0, 60.0, "Suavizado", "{:.0f} ms")

        dgrid.setHorizontalSpacing(10)
        dgrid.addWidget(self.knob_sens, 0, 0, Qt.AlignmentFlag.AlignCenter)
        dgrid.addWidget(self.knob_sharp, 0, 1, Qt.AlignmentFlag.AlignCenter)
        dgrid.addWidget(self.knob_split_smooth, 0, 2, Qt.AlignmentFlag.AlignCenter)

        lv.addWidget(dbox)

        # --- Panel derecho: gráfica + análisis ---
        right = QWidget()
        rv = QVBoxLayout(right)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Tiempo", units="s")
        self.plot.setLabel("left", "Amplitud")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.plot.plot(pen=pg.mkPen("c", width=1))
        self.playhead = pg.InfiniteLine(
            pos=0.0, angle=90, pen=pg.mkPen("r", width=2), movable=False
        )
        self.playhead.setVisible(False)
        self.plot.addItem(self.playhead)
        self.plot.scene().sigMouseClicked.connect(self._on_plot_click)
        rv.addWidget(self.plot, 1)

        info = QGroupBox("Análisis")
        ih = QHBoxLayout(info)
        self.lbl_info = QLabel("Abre un archivo de audio para analizarlo.")
        self.lbl_info.setWordWrap(True)
        ih.addWidget(self.lbl_info)
        rv.addWidget(info)

        beats_group = QGroupBox("Beats / MIDI  (Addictive, Superior, EZdrummer…)")
        bgrid = QGridLayout(beats_group)
        bgrid.setColumnStretch(0, 1)

        self.spin_note = QSpinBox()
        self.spin_note.setRange(24, 84)
        self.spin_note.setValue(36)

        self.knob_thresh = Knob(0.01, 0.50, 0.10, "Umbral", "{:.2f}")
        self.knob_thresh.valueChanged.connect(self.on_thresh)

        self.knob_wait = Knob(10.0, 500.0, 25.0, "Retrigger", "{:.0f} ms")
        self.knob_wait.valueChanged.connect(self.on_wait)

        self.chk_fast = QCheckBox("Alta resolución (doble bombo / blast beats)")
        self.chk_fast.setChecked(True)

        self.btn_detect = QPushButton("Detectar beats")
        self.btn_detect.clicked.connect(self.detect_beats_current)

        self.lbl_hits = QLabel("0 hits")

        self.btn_midi_track = QPushButton("MIDI de esta pista")
        self.btn_midi_track.clicked.connect(self.export_midi_track)

        self.btn_midi_all = QPushButton("MIDI batería completa")
        self.btn_midi_all.clicked.connect(self.export_midi_all)

        self.btn_midi_sel = QPushButton("MIDI por selección")
        self.btn_midi_sel.clicked.connect(self.export_midi_selection)

        self.btn_bass_midi = QPushButton("MIDI melódico del bajo")
        self.btn_bass_midi.clicked.connect(self.export_bass_midi)

        self.btn_oneshots = QPushButton("Exportar one-shots (kit WAV)")
        self.btn_oneshots.clicked.connect(self.export_one_shots)

        self.cmb_style = QComboBox()
        for key, pres in engine.DRUM_STYLES.items():
            self.cmb_style.addItem(pres["label"], key)
        self.cmb_style.currentIndexChanged.connect(self.on_style_change)

        row_style = QHBoxLayout()
        row_style.addWidget(QLabel("Estilo de batería"))
        row_style.addWidget(self.cmb_style)
        row_style.addSpacing(14)
        row_style.addWidget(QLabel("Nota GM"))
        row_style.addWidget(self.spin_note)
        row_style.addStretch(1)
        bgrid.addLayout(row_style, 0, 0)

        row_knobs = QHBoxLayout()
        row_knobs.setSpacing(4)
        row_knobs.addWidget(QLabel("Umbral"))
        row_knobs.addWidget(self.knob_thresh)
        row_knobs.addSpacing(10)
        row_knobs.addWidget(QLabel("Retrigger"))
        row_knobs.addWidget(self.knob_wait)
        row_knobs.addStretch(1)
        bgrid.addLayout(row_knobs, 1, 0)

        bgrid.addWidget(self.chk_fast, 2, 0)
        bgrid.addWidget(self.btn_detect, 3, 0)
        bgrid.addWidget(self.lbl_hits, 4, 0)

        row_midi = QHBoxLayout()
        row_midi.addWidget(self.btn_midi_track)
        row_midi.addWidget(self.btn_midi_all)
        row_midi.addWidget(self.btn_midi_sel)
        row_midi.addWidget(self.btn_bass_midi)
        bgrid.addLayout(row_midi, 5, 0)

        bgrid.addWidget(self.btn_oneshots, 6, 0)

        rv.addWidget(beats_group)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([440, 760])
        root.addWidget(splitter, 1)

    def _make_menu(self):
        bar = self.menuBar()
        menu = bar.addMenu("&Archivo")
        act_open = QAction("&Abrir audio…", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self.open_file)
        act_save = QAction("&Guardar sesión…", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self.save_session)
        act_load = QAction("&Cargar sesión…", self)
        act_load.setShortcut("Ctrl+L")
        act_load.triggered.connect(self.load_session)
        act_quit = QAction("&Salir", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        menu.addAction(act_open)
        menu.addAction(act_save)
        menu.addAction(act_load)
        menu.addSeparator()
        menu.addAction(act_quit)

    # ------------------------------------------------------------------
    # Estado / utilidades
    # ------------------------------------------------------------------

    def _selected_track(self):
        return self.current_track()

    def current_track(self):
        item = self.track_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _set_busy(self, busy, msg=""):
        self.progress.setVisible(busy)
        if busy:
            self.progress.setRange(0, 0)
            self.progress.setFormat(msg)
        for w in (
            self.btn_open,
            self.btn_separate,
            self.btn_split,
            self.btn_export,
            self.btn_stems,
            self.btn_play,
        ):
            w.setEnabled(not busy)
        if not busy:
            self._refresh_buttons()

    def _refresh_buttons(self):
        self.btn_export.setEnabled(bool(self.tracks))
        self.btn_stems.setEnabled(bool(self.tracks))
        self.btn_play.setEnabled(bool(self.tracks))
        self.btn_split.setEnabled(bool(self.separation.get("drums") is not None))
        self.btn_separate.setEnabled(bool(self.audio is not None))

    def add_track(self, name, audio):
        t = engine.Track(name=name, audio=np.asarray(audio, dtype=np.float32))
        self.tracks.append(t)
        self.track_list.addItem(self._track_item(t))
        self._engine_dirty = True
        self._refresh_buttons()

    def _track_label(self, t):
        dur = t.audio.shape[-1] / max(self.sr or 44100, 1)
        return f"{t.name}  |  {t.audio.shape[0]}ch · {dur:.1f}s"

    def _track_item(self, t):
        item = QListWidgetItem(self._track_label(t))
        item.setData(Qt.ItemDataRole.UserRole, t)
        return item

    def refresh_track_list(self):
        self.track_list.clear()
        for t in self.tracks:
            self.track_list.addItem(self._track_item(t))
        if self.tracks:
            self.track_list.setCurrentRow(0)

    def _mute_track_named(self, prefix):
        """Silencia la pista fuente redundante (Original / Drums) al generar
        derivados, evitando que la mezcla duplique el mismo contenido."""
        for i, t in enumerate(self.tracks):
            if t.name.lower().startswith(prefix.lower()):
                t.mute = True
                cur = self.current_track()
                if cur is t:
                    self._loading = True
                    self.ch_mute.setChecked(True)
                    self._loading = False
                return True
        return False

    # ------------------------------------------------------------------
    # Carga / separación / división
    # ------------------------------------------------------------------

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Abrir audio",
            "",
            "Audio (*.wav *.flac *.aiff *.aif *.mp3 *.m4a *.aac *.ogg *.opus);;Todos (*.*)",
        )
        if not path:
            return
        stop_all_audio()
        self._busy_file = path
        self._set_busy(True, "Cargando…")
        self._busy_target = "load"
        worker = Worker(self._load_worker)
        worker.done.connect(self._on_worker_done)
        worker.error.connect(self._on_worker_error)
        worker.start()
        self._worker = worker

    def _load_worker(self, progress):
        audio, sr = engine.load_audio(self._busy_file)
        bands = engine.analyze_bands(audio, sr)
        freqs = engine.dominant_frequencies(audio, sr)
        bpm = engine.estimate_bpm(audio, sr)
        return {"audio": audio, "sr": sr, "bands": bands, "freqs": freqs, "bpm": bpm}

    def run_separation(self):
        if self.audio is None:
            return
        self._mode_sinfonico = self.combo_mode.currentText().startswith("Sinfónico")
        self._set_busy(
            True,
            "Separando orquesta por instrumentos…" if self._mode_sinfonico
            else "Separando con Demucs…",
        )
        self._busy_target = "separate"
        worker = Worker(self._separate_worker)
        worker.done.connect(self._on_worker_done)
        worker.error.connect(self._on_worker_error)
        worker.start()
        self._worker = worker

    def _on_mode_change(self, index):
        sinfonico = self.combo_mode.currentText().startswith("Sinfónico")
        self._mode_sinfonico = sinfonico
        self.btn_separate.setText(
            "Separar orquesta" if sinfonico else "Separar con Demucs"
        )

    def _separate_worker(self, progress):
        if getattr(self, "_mode_sinfonico", False):
            audio, sr = engine.load_audio(self.file_path, sr=self.sr or 44100)
            params = {
                "sensitivity": self.knob_sens.value(),
                "sharpness": self.knob_sharp.value(),
                "smooth_ms": self.knob_split_smooth.value(),
            }
            out = engine.split_orchestra(audio, sr, params)
            return {"sr": sr, "stems": sorted(out.keys()), "separation": out}
        out, sr = engine.separate_sources(
            self.file_path, stems=("drums", "bass", "vocals", "other")
        )
        return {"sr": sr, "stems": sorted(out.keys()), "separation": out}

    def run_drum_split(self):
        drums = self.separation.get("drums")
        if drums is None:
            return
        self._set_busy(True, "Dividiendo batería…")
        self._busy_target = "split"
        self._split_params = {
            "sensitivity": self.knob_sens.value(),
            "sharpness": self.knob_sharp.value(),
            "smooth_ms": self.knob_split_smooth.value(),
        }
        worker = Worker(self._drum_split_worker)
        worker.done.connect(self._on_worker_done)
        worker.error.connect(self._on_worker_error)
        worker.start()
        self._worker = worker

    def _drum_split_worker(self, progress):
        drum_params = dict(self._split_params)
        drum_params["gamma"] = drum_params.get("sharpness", 2.0)
        result = engine.split_drums_advanced(self.separation["drums"], self.sr, drum_params)
        return {"sr": self.sr, "parts": sorted(result.keys()), "split": result}

    # ------------------------------------------------------------------
    # Manejo de workers
    # ------------------------------------------------------------------

    def _on_worker_done(self, status, result):
        if getattr(self, "_busy_target", None) == "load":
            self.audio = result["audio"]
            self.sr = result["sr"]
            self.file_path = self._busy_file
            self.current_bpm = float(result["bpm"]) or 120.0
            self.separation = {}
            self.split_result = {}
            self._reset_playback()
            self.tracks.clear()
            self.track_list.clear()
            self._engine_dirty = True
            self.add_track("Original", self.audio)
            self.track_list.setCurrentRow(0)
            self._show_waveform(self.audio, self.sr)
            self._show_analysis(result)
        elif getattr(self, "_busy_target", None) == "separate":
            stop_all_audio()
            self.separation = result["separation"]
            sinfonico = getattr(self, "_mode_sinfonico", False)
            stem_sr = result.get("sr") or self.sr
            for name in result["stems"]:
                audio = self.separation[name]
                if stem_sr != self.sr:
                    audio = engine.resample_audio(audio, stem_sr, self.sr)
                    self.separation[name] = audio
                label = name if sinfonico else f"{name.capitalize()} (Demucs)"
                self.add_track(label, audio)
            self._mute_track_named("Original")
            # Normaliza el nivel de los stems recién creados para que suenen
            # a volumen audible sin subir el gain a mano (Demucs devuelve picos bajos).
            engine.auto_adjust_tracks([t for t in self.tracks if not t.mute])
            self.on_track_selected(self.track_list.currentRow())
            if sinfonico:
                self.statusBar().showMessage(
                    "Instrumentos detectados: " + ", ".join(result["stems"]), 8000
                )
                self.auto_adjust()
        elif getattr(self, "_busy_target", None) == "split":
            stop_all_audio()
            self.split_result = result["split"]
            for name in result["parts"]:
                self.add_track(name, self.split_result[name])
            self._mute_track_named("Drums")
            engine.auto_adjust_tracks([t for t in self.tracks if not t.mute])
            self.on_track_selected(self.track_list.currentRow())
            self.statusBar().showMessage(
                "Pistas generadas: " + ", ".join(result["parts"]), 8000
            )
        elif getattr(self, "_busy_target", None) == "export":
            self.statusBar().showMessage(
                f"Exportado: {result.get('path', '')}", 8000
            )
        elif getattr(self, "_busy_target", None) == "export_stems":
            self.statusBar().showMessage(
                f"{len(result.get('files', []))} stems en {result.get('dir', '')}",
                8000,
            )
        elif getattr(self, "_busy_target", None) == "save_session":
            self.statusBar().showMessage(
                f"Sesión guardada: {result.get('path', '')}", 8000
            )
        elif getattr(self, "_busy_target", None) == "load_session":
            data = result["session"]
            self.audio = data["audio"]
            self.sr = data["sr"]
            self.separation = data["separation"]
            self.split_result = data["split_result"]
            self._reset_playback()
            self.tracks = data["tracks"]
            self._engine_dirty = True
            self.refresh_track_list()
            self._refresh_buttons()
            self.statusBar().showMessage(
                f"Sesión cargada: {self._load_session_path}", 8000
            )
        elif getattr(self, "_busy_target", None) == "detect":
            hits = result["hits"]
            name = self._detect_track.name
            self.track_hits[name] = (hits, self.spin_note.value())
            self.lbl_hits.setText(f"{len(hits)} hits detectados")
            self.statusBar().showMessage(
                f"{name}: {len(hits)} beats detectados.", 5000
            )
            self._draw_hits(hits)
        elif getattr(self, "_busy_target", None) == "export_midi_all":
            self.statusBar().showMessage(
                f"MIDI de batería exportado: {result.get('path', '')}", 8000
            )
            QMessageBox.information(self, "OK", "MIDI de batería exportado.")
        elif getattr(self, "_busy_target", None) == "export_midi_sel":
            self.statusBar().showMessage(
                f"MIDI por selección exportado: {result.get('path', '')}", 8000
            )
            QMessageBox.information(self, "OK", "MIDI por selección exportado.")
        elif getattr(self, "_busy_target", None) == "export_wav_sel":
            self.statusBar().showMessage(
                f"{len(result.get('files', []))} WAV por selección en {result.get('dir', '')}",
                8000,
            )
            QMessageBox.information(
                self, "OK", f"WAV exportados: {len(result.get('files', []))} archivos."
            )

        self._busy_target = None
        self._set_busy(False)
        self.statusBar().showMessage("Listo", 5000)

    def _on_worker_error(self, msg):
        self._busy_target = None
        self._set_busy(False)
        QMessageBox.critical(self, "Error", msg)

    # ------------------------------------------------------------------
    # Pistas
    # ------------------------------------------------------------------

    def on_track_selected(self, row):
        if self._editing is not None:
            self._flush_controls(self._editing)

        t = self.current_track()
        if not t:
            self._editing = None
            return
        self._editing = t
        self._update_controls(t)
        self._engine_dirty = True
        if self._playing and self.engine is not None:
            pos = self.engine.pos
            self._ensure_engine()
            self.engine.play(start=pos)
        self._show_waveform(engine.analysis_audio(t), self.sr or 44100)
        if not self._playing:
            self._play_index = None
            self.playhead.setVisible(False)

        note = engine.note_for_name(t.name)
        if note is not None:
            self.spin_note.setValue(note)
        self.knob_wait.setValue(engine.wait_for_name(t.name))

        stored = self.track_hits.get(t.name)
        if stored and stored[0]:
            self.lbl_hits.setText(f"{len(stored[0])} hits")
            self._draw_hits(stored[0])
        else:
            self.lbl_hits.setText("0 hits")
            self._draw_hits([])

        self._refresh_commit_btn(t)

    def _flush_controls(self, t):
        """Vuelca el estado actual de los controles a la pista t."""
        t.gain_db = self.knob_gain.value()
        t.pan = self.knob_pan.value()
        t.highpass = self.knob_hp.value()
        t.lowpass = self.knob_lp.value()
        t.gate_db = self.knob_gate.value()
        t.smooth_ms = self.knob_smooth.value()
        t.fade_in_ms = self.knob_fade_in.value()
        t.fade_out_ms = self.knob_fade_out.value()
        t.mute = self.ch_mute.isChecked()
        t.solo = self.ch_solo.isChecked()

    def _update_controls(self, t):
        ws = [
            self.knob_gain, self.knob_pan, self.knob_hp, self.knob_lp,
            self.knob_gate, self.knob_smooth, self.knob_fade_in, self.knob_fade_out,
            self.ch_mute, self.ch_solo,
        ]
        for w in ws:
            w.blockSignals(True)

        self.knob_gain.setValue(t.gain_db)
        self.knob_pan.setValue(t.pan)
        self.knob_hp.setValue(t.highpass)
        self.knob_lp.setValue(t.lowpass)
        self.knob_gate.setValue(t.gate_db)
        self.knob_smooth.setValue(t.smooth_ms)
        self.knob_fade_in.setValue(t.fade_in_ms)
        self.knob_fade_out.setValue(t.fade_out_ms)
        self.ch_mute.setChecked(t.mute)
        self.ch_solo.setChecked(t.solo)

        for w in ws:
            w.blockSignals(False)

    def on_style_change(self, index):
        style = self.cmb_style.itemData(index)
        engine.set_drum_style(style)
        pres = engine.DRUM_STYLES[style]
        self.chk_fast.setChecked(pres["fast"])
        self.knob_thresh.setValue(pres["threshold"])
        t = self._selected_track()
        if t is not None:
            self.knob_wait.setValue(engine.style_wait_for(t.name))
        self.statusBar().showMessage(f"Estilo: {pres['label']}", 5000)

    def on_gain(self, v):
        if self._editing:
            self._editing.gain_db = v
            self._editing.processed_stale = True
            self._refresh_commit_btn(self._editing)

    def on_pan(self, v):
        if self._editing:
            self._editing.pan = v

    def on_hp(self, v):
        if self._editing:
            self._editing.highpass = v
            self._editing.processed_stale = True
            self._refresh_commit_btn(self._editing)
            if self.engine: self.engine.refresh_filters(self._editing)

    def on_lp(self, v):
        if self._editing:
            self._editing.lowpass = v
            self._editing.processed_stale = True
            self._refresh_commit_btn(self._editing)
            if self.engine: self.engine.refresh_filters(self._editing)

    def on_gate(self, v):
        if self._editing:
            self._editing.gate_db = v
            self._editing.processed_stale = True
            self._refresh_commit_btn(self._editing)
            if self.engine: self.engine._schedule_static(self._editing)

    def on_smooth(self, v):
        if self._editing:
            self._editing.smooth_ms = v
            self._editing.processed_stale = True
            self._refresh_commit_btn(self._editing)
            if self.engine: self.engine._schedule_static(self._editing)

    def on_fade_in(self, v):
        if self._editing:
            self._editing.fade_in_ms = v
            self._editing.processed_stale = True
            self._refresh_commit_btn(self._editing)
            if self.engine: self.engine._schedule_static(self._editing)

    def on_fade_out(self, v):
        if self._editing:
            self._editing.fade_out_ms = v
            self._editing.processed_stale = True
            self._refresh_commit_btn(self._editing)
            if self.engine: self.engine._schedule_static(self._editing)

    def _refresh_commit_btn(self, t):
        if t is not None and t.processed is not None and t.processed_stale:
            self.btn_commit.setText("Re-renderizar ajustes → detector")
        else:
            self.btn_commit.setText("Renderizar ajustes → detector")

    def _analyze_track_audio(self, audio):
        bands = engine.analyze_bands(audio, self.sr or 44100)
        freqs = engine.dominant_frequencies(audio, self.sr or 44100)
        bpm = engine.estimate_bpm(audio, self.sr or 44100)
        return {"bpm": bpm, "bands": bands, "freqs": freqs}

    def commit_processed(self):
        t = self._selected_track()
        if not t:
            return
        self.statusBar().showMessage(f"Renderizando ajustes de {t.name}…", 3000)
        t.processed = engine.render_for_analysis(t, self.sr or 44100)
        t.processed_stale = False

        self._show_waveform(engine.analysis_audio(t), self.sr or 44100)
        self._show_analysis(self._analyze_track_audio(engine.analysis_audio(t)))

        if self.track_hits.get(t.name):
            self.detect_beats_current()

        self._refresh_commit_btn(t)
        self.statusBar().showMessage(
            f"{t.name}: el detector ahora usa el audio renderizado.", 5000
        )

    def revert_processed(self):
        t = self._selected_track()
        if not t:
            return
        t.processed = None
        t.processed_stale = False

        self._show_waveform(engine.analysis_audio(t), self.sr or 44100)
        self._show_analysis(self._analyze_track_audio(engine.analysis_audio(t)))

        if self.track_hits.get(t.name):
            self.detect_beats_current()

        self._refresh_commit_btn(t)
        self.statusBar().showMessage(
            f"{t.name}: detector de vuelta al audio original.", 5000
        )

    def _schedule_static(self, t):
        self._static_track = t
        self._static_pending = True
        self._static_timer.start()

    def _flush_static(self):
        t = getattr(self, "_static_track", None)
        if t is None or self.engine is None:
            self._static_pending = False
            return
        if getattr(self, "_static_worker", None) is not None:
            if self._static_worker.isRunning():
                self._static_timer.start()
                return
            self._static_worker = None
        self._static_pending = False
        w = FnWorker(self.engine.refresh_static, t)
        w.done.connect(self._on_static_done)
        w.error.connect(self._on_static_error)
        w.start()
        self._static_worker = w

    def _on_static_done(self, _result):
        self._static_worker = None
        if self._static_pending:
            self._static_timer.start()

    def _on_static_error(self, msg):
        self._static_worker = None
        self.statusBar().showMessage(f"Error al actualizar pista: {msg}", 5000)

    def on_mute(self, checked):
        if self._editing:
            self._editing.mute = checked

    def on_solo(self, checked):
        if self._editing:
            self._editing.solo = checked

    def auto_adjust(self):
        engine.auto_adjust_tracks(self.tracks)
        for t in self.tracks:
            t.processed_stale = True
        self.on_track_selected(self.track_list.currentRow())

    def delete_track(self):
        item = self.track_list.currentItem()
        if item is None:
            return
        t = item.data(Qt.ItemDataRole.UserRole)
        if t is not None:
            self.tracks.remove(t)
        row = self.track_list.row(item)
        self.track_list.takeItem(row)
        self._engine_dirty = True
        self._refresh_buttons()
        if self.tracks:
            self.track_list.blockSignals(True)
            self.track_list.setCurrentRow(min(row, len(self.tracks) - 1))
            self.track_list.blockSignals(False)
            self.on_track_selected(self.track_list.currentRow())
        else:
            self.plot.clear()

    def rename_track(self, item):
        t = item.data(Qt.ItemDataRole.UserRole)
        if t is None:
            return
        new, ok = QInputDialog.getText(
            self, "Renombrar pista", "Nuevo nombre:", text=t.name
        )
        if ok and new.strip():
            t.name = new.strip()
            item.setText(self._track_label(t))
            self.statusBar().showMessage(f"Pista renombrada a: {t.name}", 5000)
            self.on_track_selected(self.track_list.currentRow())

    def export_mix(self):
        selected = self._selected_tracks()
        if not selected:
            QMessageBox.information(
                self,
                "Exportar",
                "Selecciona al menos una pista para mezclar.",
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar mezcla",
            "mezcla.wav",
            "WAV (*.wav);;FLAC (*.flac);;MP3 (*.mp3);;AIFF (*.aiff)",
        )
        if not path:
            return
        self._set_busy(True, "Mezclando…")
        self._busy_target = "export"
        self._export_path = path
        self._export_tracks = selected
        worker = Worker(self._export_worker)
        worker.done.connect(self._on_worker_done)
        worker.error.connect(self._on_worker_error)
        worker.start()
        self._worker = worker

    def _export_worker(self, progress):
        mix = engine.mix_tracks(self._export_tracks, self.sr or 44100)
        engine.save_audio(self._export_path, mix, self.sr or 44100)
        return {"path": self._export_path}

    def export_stems(self):
        if not self.tracks:
            return
        directory = QFileDialog.getExistingDirectory(self, "Carpeta para los stems")
        if not directory:
            return
        self._set_busy(True, "Exportando stems…")
        self._busy_target = "export_stems"
        self._stems_dir = directory
        self._stems_tracks = list(self.tracks)
        worker = Worker(self._export_stems_worker)
        worker.done.connect(self._on_worker_done)
        worker.error.connect(self._on_worker_error)
        worker.start()
        self._worker = worker

    def _export_stems_worker(self, progress):
        sr = self.sr or 44100
        written = []
        for i, t in enumerate(self._stems_tracks, 1):
            fname = f"{i:02d}_{engine.safe_filename(t.name)}_stem.wav"
            path = os.path.join(self._stems_dir, fname)
            rendered = engine.render_track(t, sr)
            engine.save_audio(path, rendered, sr, subtype="PCM_24")
            written.append(fname)
        return {"files": written, "dir": self._stems_dir}

    # ------------------------------------------------------------------
    # Sesión
    # ------------------------------------------------------------------

    def save_session(self):
        if not self.tracks:
            QMessageBox.information(self, "Guardar sesión", "No hay pistas para guardar.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar sesión",
            "sesion.json",
            "Sesión (*.json)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        self._set_busy(True, "Guardando sesión…")
        self._busy_target = "save_session"
        self._save_session_path = path
        worker = Worker(self._save_session_worker)
        worker.done.connect(self._on_worker_done)
        worker.error.connect(self._on_worker_error)
        worker.start()
        self._worker = worker

    def _save_session_worker(self, progress):
        engine.save_session(
            self._save_session_path,
            self.audio,
            self.sr or 44100,
            self.tracks,
            self.separation,
            self.split_result,
        )
        return {"path": self._save_session_path}

    def load_session(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Cargar sesión",
            "",
            "Sesión (*.json)",
        )
        if not path:
            return
        self._set_busy(True, "Cargando sesión…")
        self._busy_target = "load_session"
        self._load_session_path = path
        worker = Worker(self._load_session_worker)
        worker.done.connect(self._on_worker_done)
        worker.error.connect(self._on_worker_error)
        worker.start()
        self._worker = worker

    def _load_session_worker(self, progress):
        data = engine.load_session(self._load_session_path)
        return {"session": data}

    # ------------------------------------------------------------------
    # Selección de pistas
    # ------------------------------------------------------------------

    def _selected_tracks(self):
        return self.selected_tracks()

    def selected_tracks(self):
        return [i.data(Qt.ItemDataRole.UserRole) for i in self.track_list.selectedItems()]

    def _on_selection_changed(self):
        self.btn_export.setText(
            "Exportar mezcla…"
            if not self._selected_tracks()
            else f"Exportar ({len(self._selected_tracks())})…"
        )

    # ------------------------------------------------------------------
    # Reproducción (mezcla en tiempo real)
    # ------------------------------------------------------------------

    def _ensure_engine(self):
        sr = self.sr or 44100
        if self.engine is None or self.engine.sr != sr:
            if self.engine is not None:
                self.engine.stop()
                self._playing = False
                self._play_timer.stop()
            self.engine = RealtimeEngine(sr)
            self._engine_dirty = True
        if self._engine_dirty:
            t = self._selected_track()
            self.engine.set_tracks([t] if t is not None else [])
            self._engine_dirty = False

    def toggle_playback(self):
        if self._playing:
            self._stop_playback()
            return
        if not self.tracks:
            return
        self._ensure_engine()
        start = self._play_index if self._play_index is not None else 0
        self.engine.play(start=start)
        self._playing = True
        self.btn_play.setText("■ Detener")
        self.playhead.setVisible(True)
        self._play_timer.start()
        t = self._selected_track()
        name = t.name if t is not None else "mezcla"
        self.statusBar().showMessage(f"Reproduciendo: {name}")

    def _tick_playhead(self):
        if not self._playing or self.engine is None:
            return
        sr = self.engine.sr or 44100
        self.playhead.setPos(self.engine.pos / sr)
        if self.engine.pos >= self.engine.maxlen:
            self._stop_playback()

    def _stop_playback(self):
        stop_all_audio()
        self._playing = False
        self._play_timer.stop()
        if self.engine is not None:
            self.engine.stop()
        self.btn_play.setText("▶ Escuchar")
        self.statusBar().showMessage("Listo", 5000)

    def _reset_playback(self):
        stop_all_audio()
        self._playing = False
        self._play_timer.stop()
        if self.engine is not None:
            self.engine.stop()
            self.engine = None
        self._play_index = None
        self.playhead.setVisible(False)
        self.btn_play.setText("▶ Escuchar")

    def _on_plot_click(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if not self.tracks:
            return
        vb = self.plot.getPlotItem().vb
        pos = vb.mapSceneToView(event.scenePos())
        self._on_plot_click_impl(float(pos.x()))

    def _on_plot_click_impl(self, seconds):
        if seconds < 0:
            return
        if not self.tracks:
            return
        self._ensure_engine()
        sr = self.engine.sr or 44100
        self._play_index = int(seconds * sr)
        self._play_index = min(self._play_index, max(0, self.engine.maxlen - 1))
        self.playhead.setPos(self._play_index / sr)
        if self._playing:
            self.engine.pos = self._play_index
        else:
            self.engine.play(start=self._play_index)
            self._playing = True
            self.btn_play.setText("■ Detener")
            self.playhead.setVisible(True)
            self._play_timer.start()
        self.statusBar().showMessage(
            f"Reproduciendo desde {self._play_index / sr:.1f} s…"
        )

    # ------------------------------------------------------------------
    # Visualización
    # ------------------------------------------------------------------

    def _show_waveform(self, audio, sr):
        mono = engine.to_mono(audio)
        n = len(mono)
        step = max(1, n // 200000)
        x = np.arange(0, n, step, dtype=np.float64) / sr
        y = mono[::step]
        self.curve.setData(x, y)

    def _show_analysis(self, result):
        bpm = result["bpm"]
        bands = result["bands"]
        freqs = result["freqs"]

        lines = [f"BPM estimado: {bpm:.1f}"]

        if bands:
            lines.append("\nEnergía por banda:")
            top = sorted(bands, key=lambda b: b["pct"], reverse=True)[:5]
            for b in top:
                lines.append(f"  {b['band']:<18} {b['pct']:5.1f}%")

        if freqs:
            lines.append("\nFrecuencias dominantes:")
            for f, mag in freqs[:5]:
                lines.append(f"  {f:7.0f} Hz")

        self.lbl_info.setText("\n".join(lines))

    # ------------------------------------------------------------------
    # Beats / MIDI
    # ------------------------------------------------------------------

    def on_thresh(self, v):
        pass

    def on_wait(self, v):
        pass

    def _draw_hits(self, hits):
        if self.hit_scatter is not None:
            try:
                self.plot.removeItem(self.hit_scatter)
            except Exception:
                pass
            self.hit_scatter = None

        if hits:
            xs = [h[0] for h in hits]
            self.hit_scatter = pg.ScatterPlotItem(
                xs,
                [0.0] * len(xs),
                size=10,
                pen=pg.mkPen("#FF3D6E", width=2),
                brush=pg.mkBrush(255, 61, 110, 120),
            )
            self.plot.addItem(self.hit_scatter)

    def detect_beats_current(self):
        t = self._selected_track() or (self.tracks[0] if self.tracks else None)
        if t is None:
            return
        self._set_busy(True, "Detectando beats…")
        self._busy_target = "detect"
        self._detect_track = t
        self._detect_audio = engine.analysis_audio(t)
        self._detect_threshold = self.knob_thresh.value()
        self._detect_wait = self.knob_wait.value()
        self._detect_fast = self.chk_fast.isChecked()
        worker = Worker(self._detect_worker)
        worker.done.connect(self._on_worker_done)
        worker.error.connect(self._on_worker_error)
        worker.start()
        self._worker = worker

    def _detect_worker(self, progress):
        hits = engine.detect_beats(
            self._detect_audio,
            self.sr or 44100,
            band=engine.band_for_name(self._detect_track.name),
            threshold=self._detect_threshold,
            wait_ms=self._detect_wait,
            fast=self._detect_fast,
        )
        return {"hits": hits}

    def export_midi_track(self):
        t = self._selected_track() or (self.tracks[0] if self.tracks else None)
        if t is None:
            return

        stored = self.track_hits.get(t.name)
        if not stored or not stored[0]:
            QMessageBox.warning(self, "Atención", "Pulsa antes 'Detectar beats'.")
            return

        hits, note = stored
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar MIDI",
            engine.safe_filename(t.name) + ".mid",
            "MIDI (*.mid)",
        )
        if not path:
            return

        engine.write_midi(path, {10: {note: hits}}, bpm=self.current_bpm)
        self.statusBar().showMessage(f"MIDI exportado: {path}", 8000)
        QMessageBox.information(self, "OK", "MIDI exportado correctamente.")

    def export_midi_all(self):
        drum_tracks = [
            t
            for t in self.tracks
            if engine.note_for_name(t.name) is not None
            and "bass" not in t.name.lower()
            and "bajo" not in t.name.lower()
        ]

        if not drum_tracks:
            QMessageBox.warning(self, "Atención", "No hay pistas de batería.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar MIDI batería", "bateria.mid", "MIDI (*.mid)"
        )
        if not path:
            return

        self._set_busy(True, "Detectando y exportando MIDI…")
        self._busy_target = "export_midi_all"
        self._midi_all_path = path
        self._midi_all_tracks = drum_tracks
        worker = Worker(self._export_midi_all_worker)
        worker.done.connect(self._on_worker_done)
        worker.error.connect(self._on_worker_error)
        worker.start()
        self._worker = worker

    def _export_midi_all_worker(self, progress):
        data = {}
        for t in self._midi_all_tracks:
            stored = self.track_hits.get(t.name)
            if stored and stored[0]:
                hits, note = stored
            else:
                hits = engine.detect_beats(
                    engine.analysis_audio(t),
                    self.sr or 44100,
                    band=engine.band_for_name(t.name),
                    threshold=0.10,
                    wait_ms=engine.wait_for_name(t.name),
                )
                note = engine.note_for_name(t.name)
                self.track_hits[t.name] = (hits, note)
            data.setdefault(note, []).extend(hits)

        engine.write_midi(self._midi_all_path, {10: data}, bpm=self.current_bpm)
        return {"path": self._midi_all_path}

    # ------------------------------------------------------------------
    # Exportación por selección
    # ------------------------------------------------------------------

    def export_midi_selection(self):
        tracks = self._selected_tracks()
        if not tracks:
            QMessageBox.warning(self, "Atención", "No hay pistas seleccionadas.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar MIDI por selección", "seleccion.mid", "MIDI (*.mid)"
        )
        if not path:
            return

        self._set_busy(True, "Generando MIDI por selección…")
        self._busy_target = "export_midi_sel"
        self._midi_sel_path = path
        self._midi_sel_tracks = list(tracks)
        worker = Worker(self._export_midi_selection_worker)
        worker.done.connect(self._on_worker_done)
        worker.error.connect(self._on_worker_error)
        worker.start()
        self._worker = worker

    def _export_midi_selection_worker(self, progress):
        items = []
        sr = self.sr or 44100
        for t in self._midi_sel_tracks:
            n = t.name.lower()
            note = engine.note_for_name(t.name)
            stored = self.track_hits.get(t.name)
            if "bass" in n or "bajo" in n:
                items.append(
                    (t.name, engine.analysis_audio(t), sr, "bass", 0, 0, 0, False, None)
                )
            elif note is not None:
                hits = stored[0] if stored and stored[0] else None
                items.append(
                    (
                        t.name,
                        engine.analysis_audio(t),
                        sr,
                        "drum",
                        note,
                        engine.style_threshold(),
                        engine.style_wait_for(t.name),
                        engine.style_fast(),
                        hits,
                    )
                )
        events = _build_midi_events(items)
        engine.write_midi(self._midi_sel_path, events, bpm=self.current_bpm)
        return {"path": self._midi_sel_path}

    def export_wav_selection(self):
        tracks = self._selected_tracks()
        if not tracks:
            QMessageBox.warning(self, "Atención", "No hay pistas seleccionadas.")
            return

        directory = QFileDialog.getExistingDirectory(
            self, "Carpeta para los WAV por selección"
        )
        if not directory:
            return

        self._set_busy(True, "Renderizando WAV por selección…")
        self._busy_target = "export_wav_sel"
        self._wav_sel_dir = directory
        self._wav_sel_tracks = list(tracks)
        worker = Worker(self._export_wav_selection_worker)
        worker.done.connect(self._on_worker_done)
        worker.error.connect(self._on_worker_error)
        worker.start()
        self._worker = worker

    def _export_wav_selection_worker(self, progress):
        sr = self.sr or 44100
        files = []
        for t in self._wav_sel_tracks:
            audio = engine.render_track(t, sr)
            path = os.path.join(
                self._wav_sel_dir, engine.safe_filename(t.name) + ".wav"
            )
            engine.save_audio(path, audio, sr, subtype="PCM_24")
            files.append(path)
        return {"dir": self._wav_sel_dir, "files": files}

    # ------------------------------------------------------------------
    # MIDI melódico del bajo
    # ------------------------------------------------------------------

    def export_bass_midi(self):
        bass_track = None
        for t in self.tracks:
            n = t.name.lower()
            if "bass" in n or "bajo" in n:
                bass_track = t
                break

        if bass_track is None:
            QMessageBox.warning(self, "Atención", "No hay pista de bajo.")
            return

        if self.bass_worker and self.bass_worker.isRunning():
            return

        self.btn_bass_midi.setEnabled(False)
        self.statusBar().showMessage("Analizando notas del bajo (puede tardar)…")

        self.bass_worker = BassWorker(
            engine.detect_bass_notes, engine.analysis_audio(bass_track), self.sr or 44100
        )
        self.bass_worker.done.connect(self.on_bass_notes)
        self.bass_worker.error.connect(self.on_bass_error)
        self.bass_worker.start()

    def on_bass_notes(self, notes):
        self.btn_bass_midi.setEnabled(True)

        if not notes:
            QMessageBox.warning(self, "Atención", "No se detectaron notas de bajo.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "MIDI del bajo", "bajo.mid", "MIDI (*.mid)"
        )
        if not path:
            return

        by_note = {}
        for start, dur, pitch, vel in notes:
            by_note.setdefault(pitch, []).append((start, dur, vel))

        engine.write_midi(path, {1: by_note}, bpm=self.current_bpm)
        self.statusBar().showMessage(
            f"MIDI del bajo exportado: {len(notes)} notas.", 8000
        )
        QMessageBox.information(self, "OK", "MIDI del bajo exportado.")

    def on_bass_error(self, msg):
        self.btn_bass_midi.setEnabled(True)
        self.statusBar().showMessage("Error en detección de bajo: " + msg, 8000)
        QMessageBox.critical(self, "Error", msg)

    # ------------------------------------------------------------------
    # One-shots (kit de samples WAV)
    # ------------------------------------------------------------------

    def export_one_shots(self):
        sel = self._selected_tracks()
        drum_tracks = [
            t
            for t in sel
            if engine.note_for_name(t.name) is not None
            and "bass" not in t.name.lower()
            and "bajo" not in t.name.lower()
        ]

        if not drum_tracks:
            QMessageBox.warning(self, "Atención", "No hay pistas de batería.")
            return

        directory = QFileDialog.getExistingDirectory(
            self, "Carpeta para el kit de one-shots"
        )
        if not directory:
            return

        sr = self.sr or 44100
        try:
            total = 0
            for t in drum_tracks:
                stored = self.track_hits.get(t.name)
                if stored and stored[0]:
                    hits = stored[0]
                else:
                    hits = engine.detect_beats(
                        engine.analysis_audio(t),
                        sr,
                        band=engine.band_for_name(t.name),
                        threshold=engine.style_threshold(),
                        wait_ms=engine.style_wait_for(t.name),
                        fast=engine.style_fast(),
                    )
                if not hits:
                    continue

                dur = engine.one_shot_dur_for(t.name)
                shots = engine.slice_one_shots(engine.analysis_audio(t), sr, hits, dur_ms=dur)

                folder = os.path.join(directory, engine.safe_filename(t.name))
                os.makedirs(folder, exist_ok=True)

                for i, (seg, vel) in enumerate(shots, 1):
                    fname = f"{engine.safe_filename(t.name)}_{i:03d}_vel{vel:03d}.wav"
                    engine.save_audio(
                        os.path.join(folder, fname),
                        seg,
                        sr,
                        subtype="PCM_24",
                    )
                    total += 1

            self.statusBar().showMessage(
                f"One-shots exportados: {total} samples.", 8000
            )
            QMessageBox.information(
                self, "OK", f"Kit exportado: {total} one-shots WAV."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def closeEvent(self, ev):
        stop_all_audio()
        super().closeEvent(ev)


def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setWindowIcon(load_app_icon())
    app.setFont(QFont("Segoe UI", 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
