from absl import flags
flags.FLAGS(['main']) 

import argparse
import numpy as np

from smac.smac_wrapper import new_instance
from config import parse_args


def main():
    args = parse_args()
    np.random.seed(args.seed)

    env = new_instance(
        map_name=args.smac_map,
        replay_dir=args.result_dir,
        seed=args.seed,
        save_video=True,
    )

    obs = env.reset()
    done = False
    info = {}

    while not done:
        avail_mask = env.legal_actions()
        actions = []
        for i in range(env.n_agents):
            legal_actions = np.nonzero(avail_mask[i])[0] 
            move = np.random.choice(legal_actions).item()
            actions.append(move)

        obs, reward, done, info = env.step(actions)

    print(f"Episode finished. Team won? {info['battle_won']}")
    env.close()


if __name__ == "__main__":
    main()
