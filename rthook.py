import os
import sys

if getattr(sys, 'frozen', False):
    base = sys._MEIPASS
    torch_lib = os.path.join(base, 'torch', 'lib')
    if os.path.exists(torch_lib):
        os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')

import torch
import torchvision
