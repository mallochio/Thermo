from sklearn.pipeline import Pipeline

import pipeline
from src import config
from src.data_manager import load_pipeline


def run_training(x_train, y_train) -> Pipeline:
    # run pipeline, train network
    pipe = pipeline.neuralPipe.fit(X=x_train, y=y_train)

    # save pipeline
    # save_pipeline(pipeline_to_persist=pipe)
    return pipe


def run_testing(x_test, model=None):
    if not model:
        pipeline_file_name = f"{config.PIPELINE_SAVE_FILE}{config.VERSION}.pkl"
        _regressor_pipe = load_pipeline(file_name=pipeline_file_name)
        predictions = _regressor_pipe.predict(x_test)
    else:
        predictions = model.predict(x_test)
    return predictions
