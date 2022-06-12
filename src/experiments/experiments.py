"""
Provides functions for running experiments, with wandb tracking.

Authors:
    Rahul Yedida <rahul@ryedida.me>
"""
from raise_utils.interpret import DODGEInterpreter
from tensorflow.keras.models import Sequential
import numpy as np
import wandb


def _run(data: Data, name: str):
    """
    This shows an alternate form of docstring for functions.

    Runs one experiment, given a Data instance.

    :param {Data} data - The dataset to run on, NOT preprocessed.
    :param {str} name - The name of the experiment.
    """
    input_shape = data.x_train.shape[1]
    model = Sequential([
        Dense(5, activation='relu', name='layer1'),
        Dense(5, activation='relu', name='layer2'),
        Dense(1, activation='sigmoid', name='output')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    wandb.init(project='dl4se-demo')

    history = model.fit(
        data.x_train,
        data.y_train,
        batch_size=128,
        epochs=30,
        validation_data=(data.x_test, data.y_test)
    )

    # Get the results.
    metrics_list = ['f1', 'd2h', 'pd', 'pf', 'prec']
    preds = np.argmax(model.predict(data.x_test), axis=1)
    m = ClassificationMetrics(data.y_test, preds)
    m.add_metrics(metrics_list)
    results = m.get_metrics()

    wandb.log(dict(zip(metrics_list, results)))

    return results


def run_experiment(name, dataset):
    """
    Runs an experiment with a given config, for a given dataset.

    :param name (str) - A name for the experiment.
    :param dataset (Data) - A Data object from src.data.get_data
    :return result - F1 scores for the experiment.
    """
    results = []
    for i in range(10):
        result = _run(dataset, name)
        results.append(result[0])

    return results
