# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_submodules

torchvision_datas, torch_binaries, torch_hiddenimports = collect_all('torchvision')
torch_datas, torch_binaries2, torch_hiddenimports2 = collect_all('torch')
qfw_datas, qfw_binaries, qfw_hiddenimports = collect_all('qfluentwidgets')
qfw_extra = collect_submodules('qfluentwidgets')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=torch_binaries + torch_binaries2 + qfw_binaries,
    datas=[
        ('assets', 'assets'),
        ('models', 'models'),
    ] + torchvision_datas + torch_datas + qfw_datas,
    hiddenimports=[
        'torch',
        'torchvision',
        'numpy',
        'cv2',
        'PyQt5',
        'PyQt5.QtWidgets',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtSvg',
        'darkdetect',
    ] + torch_hiddenimports + torch_hiddenimports2 + qfw_hiddenimports + qfw_extra,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['rthook.py'],
    excludes=['tensorboard', 'torch.distributed', 'flet', 'flet_core'],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PenID',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PenID',
)
