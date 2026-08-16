# -*- coding: utf-8 -*-
"""Build offline installers for Bass & Drums Extractor.

Este script se encarga de:
  1. Descargar (solo en build-time, NO en runtime) el modelo Demucs y el
     binario de FFmpeg, dejandolos dentro de ``assets/`` para que el
     instalador final sea 100% offline.
  2. Ejecutar PyInstaller (modo ``onedir``) empaquetando Python + todas las
     dependencias + assets.

Soporta Windows, Linux y macOS. Para Linux se genera un AppDir/AppImage-ready
y para macOS un .app. Al correr en cada OS produce el instalador de ese OS.

Uso (en cada sistema operativo):
    python build_installer.py            # build del OS actual
    python build_installer.py --platform win32
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
MODELS_CACHE = ASSETS / "models" / "hf_cache"      # caché HuggingFace (modelo Demucs)
FFMPEG_DIR = ASSETS / "ffmpeg"                       # ffmpeg/<plat>/ffmpeg[.exe]
BUILD_OUT = HERE / "build"

# Binarios FFmpeg descargables (build-time). Se eligen según el SO objetivo.
FFMPEG_URLS = {
    "win32": "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-win64-gpl.zip",
    "linux": "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-amd64-lgpl-shared.tgz",
    "darwin": "https://evermeet.cx/ffmpeg/get/zip",
}

MODEL_NAME = "htdemucs_ft"


def platform_tag():
    return {"win32": "win64", "linux": "linux-amd64", "darwin": "macos"}[
        sys.platform if sys.platform in ("win32", "linux", "darwin") else "win32"
    ]


def ensure_ffmpeg(platform):
    plat = {"win32": "win64", "linux": "linux-amd64", "darwin": "macos"}.get(platform, "win64")
    ffbin = FFMPEG_DIR / plat
    exe = ffbin / ("ffmpeg.exe" if platform == "win32" else "ffmpeg")
    if ffbin.exists() and list(ffbin.iterdir()):
        return  # ya está bundleado
    ffbin.mkdir(parents=True, exist_ok=True)
    print(f"[build] descargando FFmpeg para {platform} ...")
    from urllib.request import urlretrieve
    url = FFMPEG_URLS[platform]
    tmp = HERE / f"_ffmpeg_dl"
    tmp_zip = tmp.with_suffix(".zip")
    if platform == "darwin":
        tmp_zip = tmp.with_suffix(".zip")
        urlretrieve(url, tmp_zip)
        import zipfile
        with zipfile.ZipFile(tmp_zip) as z:
            z.extractall(tmp)
        src_exe = next(tmp.rglob("ffmpeg"))
        shutil.copy2(src_exe, exe)
    else:
        urlretrieve(url, tmp_zip)
        if platform == "win32":
            import zipfile
            with zipfile.ZipFile(tmp_zip) as z:
                z.extractall(tmp)
            src_exe = next(tmp.rglob("ffmpeg.exe"))
            shutil.copy2(src_exe, exe)
        else:
            import tarfile
            with tarfile.open(tmp_zip) as t:
                t.extractall(tmp)
            src_exe = next(tmp.rglob("ffmpeg"))
            shutil.copy2(src_exe, exe)
    tmp_zip.unlink(missing_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)
    if not exe.exists():
        raise RuntimeError(f"ffmpeg no se descargó en {exe}")
    print(f"[build] ffmpeg bundleado en {exe}")


def ensure_model():
    """Descarga el modelo Demucs al assets/ bundle (build-time).

    Descarga tanto los archivos .safetensors (HF cache) como los .th
    legados (torch.hub) para que el bundle sea 100% offline tanto si
    Demucs usa la ruta HF como la ruta legacy."""
    torch_hub_cache = ASSETS / "models" / "torch_hub" / "hub" / "checkpoints"
    torch_hub_cache.mkdir(parents=True, exist_ok=True)

    # Descargar modelo HF (offline path 1)
    if not (MODELS_CACHE.exists() and any(MODELS_CACHE.rglob("*.safetensors"))):
        print("[build] descargando modelo Demucs HF (build-time) ...")
        import os as _os
        _os.environ.setdefault("HF_HOME", str(MODELS_CACHE))
        from demucs.pretrained import get_model
        get_model(MODEL_NAME)
        if not any(MODELS_CACHE.rglob("*.safetensors")):
            raise RuntimeError("Falló la descarga del modelo HF")
        print("[build] modelo HF bundleado")

    # Descargar modelo legacy torch.hub (offline path 2 — fallback)
    torch_hub_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
    if not list(torch_hub_cache.glob("*.th")):
        print("[build] descargando modelo Demucs legacy (build-time) ...")
        import os as _os
        # Deshabilitar HF para forzar la ruta legacy (AWS S3)
        _os.environ["HF_HUB_OFFLINE"] = "1"
        from demucs.repo import RemoteRepo, BagOnlyRepo, AnyModelRepo
        from demucs.pretrained import _parse_remote_files, REMOTE_ROOT
        models = _parse_remote_files(REMOTE_ROOT / "files.txt")
        model_repo = RemoteRepo(models)
        bag_repo = BagOnlyRepo(REMOTE_ROOT, model_repo)
        any_repo = AnyModelRepo(model_repo, bag_repo)
        any_repo.get_model(MODEL_NAME)  # descarga .th a ~/.cache/torch/hub/checkpoints/
        # Copiar .th desde la caché de torch.hub al bundle
        if os.path.isdir(torch_hub_dir):
            for th in Path(torch_hub_dir).glob("*.th"):
                shutil.copy2(th, torch_hub_cache / th.name)
        if not any(torch_hub_cache.glob("*.th")):
            raise RuntimeError("Falló la descarga del modelo legacy .th")
        print("[build] modelo legacy bundleado")
    else:
        print("[build] modelo legacy ya está en assets/models/torch_hub")


def _post_build_fixup(bundle_dir):
    """Parcha el bundle onedir para asegurar que assets críticos están presentes."""
    internal = bundle_dir / "_internal"
    assets_dest = internal / "assets"
    # Asegurar que los iconos están en assets/
    for icon in ("icon.ico", "icon.icns", "icon.png", "icon_256.png"):
        src = ASSETS / icon
        if src.exists():
            dst = assets_dest / icon
            if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
                shutil.copy2(src, dst)
    # Verificar ffmpeg
    for plat, subdir in [("win32", "win64"), ("linux", "linux-amd64"), ("darwin", "macos")]:
        ffbin = assets_dest / "ffmpeg" / subdir
        if not ffbin.exists():
            src_ffmpeg = ASSETS / "ffmpeg" / subdir
            if src_ffmpeg.exists():
                shutil.copytree(src_ffmpeg, ffbin)


def build_windows():
    ensure_ffmpeg("win32")
    ensure_model()
    BUILD_OUT.mkdir(parents=True, exist_ok=True)
    # Limpiar dist anterior
    dist = HERE / "dist"
    if dist.exists():
        shutil.rmtree(dist)
    build = HERE / "build_tmp"
    if build.exists():
        shutil.rmtree(build)

    # Hidden imports de deps pesadas importadas diferidamente (torch, demucs, scipy, …)
    hidden = [
        "torch", "torch._C",
        "demucs", "demucs.pretrained", "demucs.hf", "demucs.apply",
        "demucs.states", "demucs.utils", "demucs.demucs", "demucs.hdemucs",
        "demucs.spec", "demucs.wiener", "demucs.repo", "demucs.audio",
        "safetensors", "safetensors.torch_extension",
        "librosa", "librosa.core.beat", "librosa.feature",
        "soundfile", "audioread", "audioread.backends",
        "scipy", "scipy.ndimage", "scipy.signal", "scipy.signal._lsq",
        "sklearn", "sklearn.cluster",  # librosa lo usa de forma opcional
        "numba", "numba.core", "numba.np",
    ]

    sep = os.pathsep  # ';' en windows
    add_assets = f"{ASSETS}{sep}assets"

    cmd = [
        sys.executable, "-m", "PyInstaller",
        str(HERE / "app.py"),
        "--name", "BeatXtractor",
        "--onedir",
        "--noconfirm",
        "--noconsole",         # GUI app
        "--add-data", add_assets,
        "--icon", str(ASSETS / "icon.ico"),   # ícono del .exe
        "--collect-submodules", "torch",
        "--collect-submodules", "librosa",
        "--collect-submodules", "scipy",
        "--collect-data", "soundfile",
        "--distpath", str(dist),
        "--workpath", str(build),
    ]
    for h in hidden:
        cmd += ["--hidden-import", h]

    print("[build] corriendo PyInstaller ...")
    r = subprocess.run(cmd, cwd=str(HERE))
    if r.returncode != 0:
        raise SystemExit("PyInstaller falló")
    print(f"[build] onedir listo en {dist / 'BeatXtractor'}")
    _post_build_fixup(dist / "BeatXtractor")
    return dist / "BeatXtractor"


def build_linux():
    ensure_ffmpeg("linux")
    ensure_model()
    dist = HERE / "dist"
    if dist.exists():
        shutil.rmtree(dist)
    build = HERE / "build_tmp"
    if build.exists():
        shutil.rmtree(build)
    cmd = [
        sys.executable, "-m", "PyInstaller",
        str(HERE / "app.py"),
        "--name", "BeatXtractor",
        "--onedir",
        "--noconfirm",
        "--add-data", f"{ASSETS}:assets",
        "--icon", str(ASSETS / "icon.ico"),       # ícono del ejecutable
        "--collect-submodules", "torch",
        "--collect-submodules", "librosa",
        "--collect-submodules", "scipy",
        "--collect-data", "soundfile",
        "--distpath", str(dist),
        "--workpath", str(build),
        "--hidden-import", "torch",
        "--hidden-import", "demucs.pretrained",
        "--hidden-import", "demucs.hf",
        "--hidden-import", "librosa",
        "--hidden-import", "soundfile",
        "--hidden-import", "safetensors",
        "--hidden-import", "numba",
    ]
    r = subprocess.run(cmd, cwd=str(HERE))
    if r.returncode != 0:
        raise SystemExit("PyInstaller falló")
    print(f"[build] AppDir (AppImage-ready) listo en {dist / 'BeatXtractor'}")
    _post_build_fixup(dist / "BeatXtractor")
    return dist / "BeatXtractor"


def build_macos():
    ensure_ffmpeg("darwin")
    ensure_model()
    dist = HERE / "dist"
    if dist.exists():
        shutil.rmtree(dist)
    build = HERE / "build_tmp"
    if build.exists():
        shutil.rmtree(build)
    cmd = [
        sys.executable, "-m", "PyInstaller",
        str(HERE / "app.py"),
        "--name", "BeatXtractor",
        "--windowed",
        "--noconfirm",
        "--add-data", f"{ASSETS}:assets",
        "--icon", str(ASSETS / "icon.icns"),     # ícono del .app (macOS)
        "--collect-submodules", "torch",
        "--collect-submodules", "librosa",
        "--collect-submodules", "scipy",
        "--collect-data", "soundfile",
        "--distpath", str(dist),
        "--workpath", str(build),
        "--hidden-import", "torch",
        "--hidden-import", "demucs.pretrained",
        "--hidden-import", "demucs.hf",
        "--hidden-import", "librosa",
        "--hidden-import", "soundfile",
        "--hidden-import", "safetensors",
        "--hidden-import", "numba",
    ]
    r = subprocess.run(cmd, cwd=str(HERE))
    if r.returncode != 0:
        raise SystemExit("PyInstaller falló")
    print(f"[build] .app listo en {dist / 'BeatXtractor.app'}")
    _post_build_fixup(dist / "BeatXtractor.app" / "Contents" / "Resources")
    return dist / "BeatXtractor.app"


def main():
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == "--platform":
        platform = args[1]
    else:
        platform = sys.platform
    if platform.startswith("linux"):
        return build_linux() if platform == "linux" else None
    if platform == "darwin":
        return build_macos()
    if platform == "win32":
        return build_windows()
    return build_windows()


if __name__ == "__main__":
    main()
