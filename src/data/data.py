"""
Provides data fetching and preprocessing functionalities.

Authors:
    Rahul Yedida <rahul@ryedida.me>
"""
from raise_utils.data import DataLoader
from raise_utils.hooks import Hook


def _binarize(x, y):
    """
    Binarizes data.

    :param x (np.array | pd.DataFrame) - Ignored. Needed for Hooks.
    :param y (np.array | pd.DataFrame) - The output vector.
    """
    y[y > 1] = 1


def get_data(base_path, files):
    """
    Fetches data from a list of data files, and binarizes it.

    :param base_path (str) - The base path to the data files.
    :param files (list) - A list of files to load.
    :return dataset (Data) - A Data object with the dataset loaded and binarized.
    """
    dataset = DataLoader.from_files(
        base_path=base_path, files=files, hooks=[Hook('binarize', _binarize)])

    return dataset
