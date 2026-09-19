import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch_lib = os.path.join(os.path.dirname(__file__), "venv", "Lib", "site-packages", "torch", "lib")
if os.path.exists(torch_lib):
    os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")

import torch
import torchvision

from src.ui.app import main

if __name__ == "__main__":
    main()
