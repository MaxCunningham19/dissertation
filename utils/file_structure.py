import os
from typing import Iterable
from . import constants


def create_file_structure(path: str):
    """Creates a file structure for the given path"""
    if not os.path.exists(path):
        os.makedirs(path)


def create_file_structures(paths: Iterable[str]):
    """Creates a file structure for the given paths"""
    for path in paths:
        create_file_structure(path)


def images_dir(env_tag: str, model_tag: str):
    """Returns the images directory"""
    return f"{constants.RESULTS_DIR}/{env_tag}/{model_tag}/{constants.IMAGES_DIR}/"


def models_dir(env_tag: str, model_tag: str):
    """Returns the models directory"""
    return f"{constants.RESULTS_DIR}/{env_tag}/{model_tag}/{constants.MODELS_DIR}/"


def videos_dir(env_tag: str, model_tag: str):
    """Returns the videos directory"""
    return f"{constants.RESULTS_DIR}/{env_tag}/{model_tag}/{constants.VIDEOS_DIR}/"


def results_dir(env_tag: str, model_tag: str):
    """Returns the results directory"""
    return f"{constants.RESULTS_DIR}/{env_tag}/{model_tag}/"


def fnrp(string: str) -> str:
    """Makes a string file name safe"""
    return string.replace("-", "_").replace(" ", "").replace(".", "_")


def generate_file_structure(
    env: str, env_kwargs_string: str, model: str, model_kwargs_string: str, exploration: str, exploration_kwargs_string: str
) -> tuple[str, str, str]:
    """
    Generates a file structure
    Returns:
        paths: A tuple of the paths to the results,images, models, and videos directories
    """
    formatted_env_kwargs = fnrp(env_kwargs_string)
    env_tag = fnrp(env)
    if formatted_env_kwargs != "":
        env_tag += f"_{formatted_env_kwargs}"

    formatted_model_kwargs = fnrp(model_kwargs_string)
    model_tag = fnrp(model)
    if formatted_model_kwargs != "":
        model_tag += f"_{formatted_model_kwargs}"

    formatted_exploration_kwargs = fnrp(exploration_kwargs_string)
    exploration_tag = fnrp(exploration)
    if formatted_exploration_kwargs != "":
        exploration_tag += f"_{formatted_exploration_kwargs}"

    model_tag = model_tag + "__" + exploration_tag
    paths = (results_dir(env_tag, model_tag), images_dir(env_tag, model_tag), models_dir(env_tag, model_tag), videos_dir(env_tag, model_tag))
    create_file_structures(paths)
    return paths
