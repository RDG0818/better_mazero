import argparse
import datetime
import math
import os

import torch
import numpy as np

def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Base Config MCTS + SMAC"
    )

    # ----------------------------
    # General / Experiment params
    # ----------------------------
    group_general = parser.add_argument_group("General parameters")
    group_general.add_argument(
        "--opr",
        type=str,
        default="train",
        choices=["train", "test", "eval"],
        help="Operation mode: train, test, or eval",
    )
    group_general.add_argument(
        "--env_name",
        type=str,
        default="3m",
        help="Name of the environment or scenario (e.g., '3m' for 3m SMAC)",
    )
    group_general.add_argument(
        "--exp_name",
        default="test",
        type=str,
        help="Identifier for this experiment (used in logging/checkpoints)",
    )
    group_general.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for numpy/torch (default: %(default)s)",
    )
    group_general.add_argument(
        "--discount",
        type=float,
        default=0.99,
        help="Discount factor gamma (default: %(default)s)",
    )
    group_general.add_argument(
        "--result_dir",
        type=str,
        default=os.path.join(os.getcwd(), "logs"),
        help="Directory to store results and logs (default: %(default)s)",
    )
    group_general.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="If set, log metrics to Weights & Biases (default: %(default)s)",
    )

    # ---------------------
    # Environment parameters
    # ---------------------
    group_env = parser.add_argument_group("Environment parameters")
    group_env.add_argument(
        "--smac_map",
        type=str,
        default="3m",
        help="SMAC map name (only used if case=='smac')",
    )
    group_env.add_argument(
        "--n_agents",
        type=int,
        default=3,
        help="Number of agents (only used if case=='smac')",
    )
    group_env.add_argument(
        "--max_steps",
        type=int,
        default=50,
        help="Maximum steps per episode (default: %(default)s)",
    )

    # -----------------------
    # MCTS & UCB parameters
    # -----------------------
    group_mcts = parser.add_argument_group("MCTS & UCB parameters")
    group_mcts.add_argument(
        "--num_simulations",
        type=int,
        default=20,
        help="Number of MCTS simulations per move (default: %(default)s)",
    )
    group_mcts.add_argument(
        "--c_puct",
        type=float,
        default=1.0,
        help="PUCT exploration constant (default: %(default)s)",
    )
    group_mcts.add_argument(
        "--rollout_depth",
        type=int,
        default=10,
        help="Maximum rollout depth for a single simulation (default: %(default)s)",
    )
    group_mcts.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor used inside rollouts (default: %(default)s)",
    )

    # ------------------
    # Training parameters
    # ------------------
    group_train = parser.add_argument_group("Training parameters")
    group_train.add_argument(
        "--train_on_gpu",
        action="store_true",
        default=False,
        help="If set, perform training on GPU (default: %(default)s)",
    )
    group_train.add_argument(
        "--training_steps",
        type=int,
        default=100000,
        help="Number of training steps (default: %(default)s)",
    )
    group_train.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for any neural network updates (default: %(default)s)",
    )
    group_train.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate for optimizer (default: %(default)s)",
    )
    group_train.add_argument(
        "--max_grad_norm",
        type=float,
        default=5.0,
        help="Max gradient norm for clipping (default: %(default)s)",
    )

    # ---------------------
    # Logging & Checkpoints
    # ---------------------
    group_log = parser.add_argument_group("Logging & checkpoint parameters")
    group_log.add_argument(
        "--save_interval",
        type=int,
        default=10000,
        help="How many steps between saving model checkpoints (default: %(default)s)",
    )
    group_log.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="How many steps between printing training logs (default: %(default)s)",
    )
    group_log.add_argument(
        "--checkpoint_dir",
        type=str,
        default=os.path.join(os.getcwd(), "checkpoints"),
        help="Directory to save model checkpoints (default: %(default)s)",
    )

    return parser.parse_args(args)
