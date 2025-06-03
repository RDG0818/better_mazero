import gymnasium as gym
from typing import List, Tuple, Dict
import numpy as np
from .mappo_smac.StarCraft2_Env import StarCraft2Env
from gymnasium.spaces import Discrete, Box


class SMACWrapper():

    def __init__(self, env: StarCraft2Env, save_video=True):
        """SMAC Wrapper

        Parameters
        ----------
        env: StarCraft2Env
            StarCraft2Env instance
        discount: float
            discount of env
        """
        self.env = env
        self.n_agents = env.n_agents
        self.obs_size = self.env.get_obs_size()[0]
        self.action_space_size = env.n_actions
        '''
        n_action defines all potential actions for a single agent, includes:
            0: no operation (valid only when dead)
            1: stop
            2: move north
            3: move south
            4: move east
            5: move west
            6~: specific enemy_id to attack
        So n_action = 6 + n_enemies
        '''
        self.save_video = save_video
        self.action_space: List[Discrete] = env.action_space
        self.observation_space: List[Tuple[int]] = env.observation_space


    def legal_actions(self) -> List[List[int]]:
        return self.env.get_avail_actions()

    def get_max_episode_steps(self) -> int:
        return self.env.episode_limit

    def step(self, action: List[int]):
        local_obs, global_state, rewards, dones, infos, available_actions = self.env.step(action)
        observation = np.asarray(local_obs)[:, :, None, None]
        reward = float(np.mean(rewards))
        done = bool(np.all(dones))
        info = {
            "battle_won": infos[0]["won"]
        }
        return observation, reward, done, info

    def reset(self, **kwargs):
        local_obs, global_state, available_actions = self.env.reset()
        observation = np.asarray(local_obs)[:, :, None, None]
        return observation

    def close(self):
        if self.save_video:
            self.env.save_replay()
        self.env.close()

def new_instance(map_name, replay_dir, seed=None, save_video=True, **kwargs):
    args = type("Config", (object,), {
        "map_name": map_name,
        "use_stacked_frames": False,
        "stacked_frames": 1,
        "add_local_obs": False,
        "add_move_state": False,
        "add_visible_state": False,
        "add_distance_state": False,
        "add_xy_state": False,
        "add_enemy_action_state": False,
        "add_agent_id": False,
        "use_state_agent": False,
        "use_mustalive": True,
        "add_center_xy": True,
        "use_obs_instead_of_state": False,
    })
    env = StarCraft2Env(args, seed=seed, replay_dir=replay_dir)
    return SMACWrapper(env, save_video=save_video)