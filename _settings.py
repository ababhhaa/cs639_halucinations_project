import os

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.environ.get('EIGENSCORE_DATA_ROOT', os.path.join(_REPO_DIR, 'data'))

MODEL_PATH = os.environ.get('EIGENSCORE_MODEL_PATH', os.path.join(_DATA_ROOT, 'weights'))
DATA_FOLDER = os.environ.get('EIGENSCORE_DATA_FOLDER', os.path.join(_DATA_ROOT, 'datasets'))
GENERATION_FOLDER = os.environ.get('EIGENSCORE_GENERATION_FOLDER', os.path.join(_DATA_ROOT, 'output'))

for _path in (MODEL_PATH, DATA_FOLDER, GENERATION_FOLDER):
    os.makedirs(_path, exist_ok=True)
