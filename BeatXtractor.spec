# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules

datas = [('X:/Proyectos/AudioExtractor/BassDrumsExtractor/assets', 'assets')]
hiddenimports = ['torch', 'torch._C', 'demucs', 'demucs.pretrained', 'demucs.hf', 'demucs.apply', 'demucs.states', 'demucs.utils', 'demucs.demucs', 'demucs.hdemucs', 'demucs.spec', 'demucs.wiener', 'demucs.repo', 'demucs.audio', 'safetensors', 'safetensors.torch_extension', 'librosa', 'librosa.core.beat', 'librosa.feature', 'soundfile', 'audioread', 'audioread.backends', 'scipy', 'scipy.ndimage', 'scipy.signal', 'scipy.signal._lsq', 'sklearn', 'sklearn.cluster', 'numba', 'numba.core', 'numba.np']
datas += collect_data_files('soundfile')
hiddenimports += collect_submodules('torch')
hiddenimports += collect_submodules('librosa')
hiddenimports += collect_submodules('scipy')


a = Analysis(
    ['X:/Proyectos/AudioExtractor/BassDrumsExtractor/app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BeatXtractor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['X:/Proyectos/AudioExtractor/BassDrumsExtractor/assets/icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BeatXtractor',
)
