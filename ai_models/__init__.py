# This file marks 'ai_models' as a Python package.
# All AI model code lives in this folder.
import os
import sys

if os.name == 'nt':
    scripts_dir = os.path.join(sys.prefix, 'Scripts')
    if os.path.isdir(scripts_dir):
        try:
            os.add_dll_directory(scripts_dir)
        except Exception:
            pass
