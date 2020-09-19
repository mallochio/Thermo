import os
import pathlib

FEATURES = [
    'X1',
    'X2',
    'Y1',
    'Y2',
    "D1",
    "D2",
    "D1D2",
    "Perimeter1",
    "Perimeter2",
    "Area1",
    "Area2",
    "Identifier",
]
NFEATURES = len(FEATURES)
LABELS = ["Temperature"]
NLABELS = len(LABELS)

EPOCHS = 500
VALIDATION_SPLIT = 0.30

PACKAGE_ROOT = pathlib.Path.cwd()#.parent
DATASET_DIR = PACKAGE_ROOT / "data"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "mlruns" / "1"
DATA_FILE = "ML_Data.xlsx"
TF_LOG_DIR = PACKAGE_ROOT / "logs" / "fit"
PIPELINE_NAME = 'FullyConnectedNN'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output_v'

with open(os.path.join(PACKAGE_ROOT, 'VERSION')) as version_file:
    VERSION = version_file.read().strip()
