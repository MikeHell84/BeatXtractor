# BeatXtractor — Bass & Drums Extractor

Aplicación de escritorio (Python / PySide6) para **separar, dividir y extraer la batería y el bajo** de cualquier mezcla, convertir los golpes en **MIDI**, generar **stems** limpios y **one-shots (kit de samples WAV)** listos para usar en superiores de batería como **Addictive Drums, Superior Drummer, EZdrummer, Toontrack**, etc.

![Inicio](Screenshots/01-inicio.png)

---

## Características

| Área | Qué hace |
| --- | --- |
| **Separación por instrumentos** | Aísla batería, bajo, voces y otros usando **Demucs** (HT-Demucs 4) o modo **Sinfónico** (separa instrumentos de orquesta). |
| **División de batería** | Descompone la pista de batería en piezas individuales (**Bombo, Redoblante, Hi-Hat, Platillos, Toms**) con un algoritmo avanzado de eventos + plantillas espectrales + enmascarado Wiener. |
| **Detección de beats** | Detecta golpes por pieza con umbral, retrigger y modo de alta resolución (doble bombo / blast beats). Presets para estilos **lento (pop/rock/funk)** y **rápido (metal)**. |
| **Exportar MIDI** | MIDI por pista, batería completa, por selección y **MIDI melódico del bajo** (notas con pitch detectado). |
| **One-shots (kit WAV)** | Recorta cada golpe a un sample individual con nombre, número y velocidad, listo para tu superior de batería. |
| **Stems y mezcla** | Exporta cada pista como WAV/FLAC/MP3/AIFF, o mezcla las pistas seleccionadas con pan, gain, filtros y limitador. |
| **Render → detector** | Aplica tus ajustes (gain, filtros, gate, suavizado, fades) al audio y **congela el resultado** para que el detector use exactamente lo que escuchas. |
| **Sesiones** | Guarda y carga proyectos completos (pistas, separación, división y ajustes) en JSON. |

![Separación Demucs](Screenshots/02-separacion.png)

---

## Requisitos

- **Python 3.10+** (probado con 3.11/3.12)
- Windows, Linux o macOS (interfaz Qt)
- 4 GB de RAM recomendados (Demucs carga un modelo de red neuronal)
- Las dependencias se listan en `requirements.txt`

## Instalación

```bash
git clone https://github.com/MikeHell84/BeatXtractor.git
cd BeatXtractor
python -m venv venv

# Windows
.\venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

> Demucs descargará su modelo automáticamente en la primera separación (unas decenas de MB).

## Ejecución

```bash
# Con el entorno activado:
python app.py

# O en Windows con el script incluido:
.\run.ps1
```

---

## Guía de uso

### 1. Cargar un audio

Pulsa **Abrir audio…** (o `Ctrl+O`) y selecciona un archivo (WAV, FLAC, AIFF, MP3, M4A, AAC, OGG, OPUS). Se muestra la forma de onda, el BPM estimado, la energía por bandas y las frecuencias dominantes.

### 2. Escuchar

Selecciona una pista en la lista y pulsa **▶ Escuchar** (o la barra espaciadora). En reproducción **solo suena la pista seleccionada** (ignora mute/solo al escucharla de forma individual). Haz clic en cualquier punto de la forma de onda para reproducir desde ahí.

### 3. Ajustar la pista

Cada pista tiene su propio panel de propiedades con potenciómetros:

- **Gain** — volumen (dB)
- **Pan** — balance estéreo
- **High-pass / Low-pass** — filtros por pista
- **Gate** — ruido de fondo (dB) con **Suavizado** (ms)
- **Fade in / Fade out**
- **Mute / Solo**
- **Auto-ajustar** — aplica filtros, gate y ganancia óptimos según el tipo de pista
- **Renderizar ajustes → detector** — congela el procesado como audio definitivo; **Usar audio original** lo deshace

### 4. Separar con Demucs

Pulsa **Separar con Demucs** (modo Pop/Rock) o elige **Sinfónico (orquesta)** para separación por instrumentos. El audio original se silencia para evitar duplicación; los stems generados se auto-ajustan de nivel automáticamente.

### 5. Dividir la batería

Con la pista de batería aislada, ajusta **Sensibilidad**, **Agresividad** y **Suavizado** y pulsa **Dividir batería**. Se crean las piezas individuales con sus propios filtros.

![División de batería](Screenshots/03-division.png)

### 6. Detectar beats

Selecciona una pieza (p. ej. Bombo), ajusta **Umbral** y **Retrigger**, elige el **estilo** (lento/rápido) y pulsa **Detectar beats**. Los golpes aparecen marcados en rojo sobre la forma de onda.

![Detección de beats](Screenshots/04-beats.png)

### 7. Exportar

- **MIDI de esta pista** — golpes de la pieza seleccionada
- **MIDI batería completa** — todos los golpes de batería asignados a notas estándar (bombo 36, redoblante 38, etc.)
- **MIDI por selección** — solo las pistas seleccionadas
- **MIDI melódico del bajo** — analiza la pista de bajo y genera notas con pitch
- **Exportar one-shots** — recorta cada golpe a un kit de samples WAV (una carpeta por pieza, con velocidad)
- **Exportar stems…** — todas las pistas a WAV 24-bit en una carpeta
- **Exportar mezcla…** — mezcla las pistas seleccionadas (WAV/FLAC/MP3/AIFF)

![Análisis del bajo](Screenshots/05-bajo.png)

### 8. Guardar / cargar sesión

**Archivo → Guardar sesión…** (`Ctrl+S`) y **Cargar sesión…** (`Ctrl+L`) guardan el proyecto completo en un único JSON.

---

## Mapa de teclas

| Tecla | Acción |
| --- | --- |
| `Ctrl+O` | Abrir audio |
| `Ctrl+S` | Guardar sesión |
| `Ctrl+L` | Cargar sesión |
| `Ctrl+Q` | Salir |
| `Espacio` | Play / Stop |
| Doble clic en una pista | Renombrarla |

---

## Mapeo MIDI por defecto

| Pieza | Nota MIDI |
| --- | --- |
| Bombo (Kick) | 36 (C1) |
| Redoblante (Snare) | 38 (D1) |
| Hi-Hat | 42 (F#1) |
| Platillos (Crash/Ride) | 49 (C#2) |
| Tom Agudo | 48 (C2) |
| Tom Medio | 45 (A1) |
| Tom Bajo | 43 (G1) |

---

## Estructura del proyecto

```
BeatXtractor/
├── app.py               # Interfaz gráfica (PySide6 + pyqtgraph)
├── audio_engine.py      # Procesado: separación, división, detección, MIDI, one-shots
├── make_icon.py         # Genera el logo/icono (SVG, PNG, ICO)
├── make_screenshots.py  # Genera las capturas de la interfaz
├── requirements.txt
├── run.ps1              # Lanzador Windows
└── assets/              # Logo e icono
```

## Licencia

Uso libre para fines personales y educativos. Las bibliotecas de terceros (PySide6, Demucs, librosa, scipy, numpy) conservan sus respectivas licencias.