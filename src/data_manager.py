import typing as t

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src import config


def load_dataset(*, file_name: str) -> pd.DataFrame:
    _data = pd.read_excel(f"{config.DATASET_DIR}/{file_name}")
    return _data


# TODO - The save and load doesn't work currently. Need to fix.
def save_pipeline(*, pipeline_to_persist):
    """Persist the pipeline.

    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.PIPELINE_SAVE_FILE}{config.VERSION}.pkl"
    save_path = config.TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.

    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    However, we do also include the immediate previous
    pipeline version for differential testing purposes.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in config.TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""
    file_path = config.TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model
