# utils.py
import os

def ensure_dirs():
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
