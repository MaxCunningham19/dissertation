import argparse


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=1000, help="number of episodes")
    parser.add_argument("--plot", action="store_true", default=False, help="Enable plotting the metrics calculated")
    parser.add_argument("--record", action="store_true", default=False, help="Enable recording of episodes")
    parser.add_argument("--num_record", type=int, default=25, help="number of episodes to record")

    parser.add_argument("--path_to_load_model", type=str, help="path to load a network from")
    parser.add_argument("--path_to_save_model", type=str, default="./agents/savedNets/", help="path to save the network to")
    parser.add_argument("--path_to_csv_save", type=str, default="./output.csv", help="path to save the csv results to")

    parser.add_argument("--model", type=str, default="democratic", help="MORL model to use: dwn, democratic, dueling")
    parser.add_argument(
        "--model_kwargs", type=str, nargs="*", help="model specific parameters not provided below e.g. --model_kwargs arg1=value1 arg2=value2"
    )

    parser.add_argument("--exploration", type=str, default="epsilon", help="exploration strategy for deciding which action to take")
    parser.add_argument(
        "--exploration_kwargs",
        type=str,
        nargs="*",
        help="exploration strategy specific parameters not provided below e.g. --exploration_kwargs arg1=value1 arg2=value2",
    )

    parser.add_argument("--env", type=str, default="deep-sea-treasure-v0", help="the mo-gymnasium environment to train the agent on")
    parser.add_argument("--env_kwargs", type=str, nargs="*", help="the key value pair arguments for the environment e.g. key1=value1")

    parser.add_argument(
        "--not_training",
        action="store_true",
        default=False,
        help="set this flag if you do not want the agent to train, used to evaluate a trained agents execution",
    )

    return parser
