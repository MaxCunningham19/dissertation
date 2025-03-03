import os
import constants


def create_file_structure(path: str):
    """Creates a file structure for the given path"""
    if not os.path.exists(path):
        os.makedirs(path)


def create_file_structures(paths: list[str]):
    """Creates a file structure for the given paths"""
    for path in paths:
        create_file_structure(path)


def images_dir(tag: str):
    """Returns the images directory"""
    return f"{constants.RESULTS_DIR}/{tag}/{constants.IMAGES_DIR}"


def models_dir(tag: str):
    """Returns the models directory"""
    return f"{constants.RESULTS_DIR}/{tag}/{constants.MODELS_DIR}"


def videos_dir(tag: str):
    """Returns the videos directory"""
    return f"{constants.RESULTS_DIR}/{tag}/{constants.VIDEOS_DIR}"


def generate_file_structure(tag: str):
    """Generates a file structure"""
    paths = [images_dir(tag), models_dir(tag), videos_dir(tag)]
    create_file_structures(paths)
