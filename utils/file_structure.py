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


def create_file_structure_for_run(run_id: str):
    """Creates a file structure for the given run id"""
    paths = [constants.RESULTS_DIR, constants.MODELS_DIR, constants.IMAGES_DIR]
    create_file_structures(paths)
