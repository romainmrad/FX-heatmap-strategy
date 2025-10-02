import os
import zipfile


def init_project() -> None:
    """
    Initialise project directories
    """
    dirs = [
        'config',
        'data',
        'data/backtest',
        'data/heatmaps',
        'data/heatmaps/negative',
        'data/heatmaps/positive',
        'data/matrices/negative',
        'data/matrices/positive',
        'data/processed/currency',
        'data/processed/index',
        'data/raw/currency',
        'data/raw/index',
        'documentation',
        'images',
        'images/backtest',
        'images/feature_selection',
        'images/feature_selection/currency',
        'images/feature_selection/index',
        'images/model_performance',
        'images/timeseries',
        'model'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    if not os.path.exists('./model/model.keras'):
        zip_path = "./model/model.keras.zip"
        extract_to = "./model"
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)


def clear_dir(path: str | os.PathLike[str]) -> None:
    """
    Clear directory contents if it exists
    :param path: directory path
    """
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


def remove_ds_store() -> None:
    """
    Recursively removes all .DS_Store files within the main directory,
    skipping ./.venv/ and ./.idea/
    """
    for root, dirs, files in os.walk('./'):
        dirs[:] = [d for d in dirs if d not in ('.venv', '.idea')]
        for file in files:
            if file == ".DS_Store":
                file_path = os.path.join(root, file)
                os.remove(file_path)
