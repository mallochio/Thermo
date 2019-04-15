# -*- coding: utf-8 -*-
# Currently doesn't work, fix
import subprocess


def run_mlflow_ui() :
    bashCommand = "mlflow ui"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    process.wait()
