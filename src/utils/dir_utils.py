import os


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
