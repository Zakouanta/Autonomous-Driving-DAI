# Imports
import gymnasium as gym
import numpy as np
import highway_env
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy

# Hyper parameters
total_timesteps = 100
n_eval_episodes = 100
config = {
    "observations": {"type": "Kinematics"},
    "actions": {"type": "DiscreteMetaAction"},
    "lanes_count": 2,
    "vehicles_count": 60,  # Set to 1 for the sake of visualization when using more than 2 controlled vehicles
    "controlled_vehicles": 1,
    "duration": 40,  # [s]
    "initial_spacing": 2,
    "collision_reward": -2,  # The reward received when colliding with a vehicle.
    "reward_speed_range": [
        90,
        110,
    ],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False,
}
env = gym.make("highway-v0", render_mode="rgb_array")
env.unwrapped.configure(config)
env.reset()
# Insert the model
model = PPO(
    MlpPolicy,
    env,
    # verbose=0,
    policy_kwargs=dict(net_arch=[256, 256]),
    learning_rate=5e-4,
    # buffer_size=15000,
    # learning_starts=200,
    batch_size=32,
    gamma=0.8,
    # train_freq=1,
    # gradient_steps=1,
    # target_update_interval=50,
    verbose=1,
    tensorboard_log="highway_ppo/",  # where the tensorboard will be stored
)


def evaluate(
    model: BaseAlgorithm,
    num_episodes: int = 100,
    deterministic: bool = True,
) -> float:
    """_Evaluate The PPO  on an agent for 'num_episodes'_

    Args:
        model (BaseAlgorithm): _description_
        num_episodes (int, optional): _description_. Defaults to 100.
        deterministic (bool, optional): _description_. Defaults to True. whether to use stochastic or deterministic actions

    Returns:
        float: _Mean reward for the las 'num_episodes'_
    """
    vec_env = model.get_env()
    obs = vec_env.reset()
    all_episodes_rewards = []
    for _ in range(num_episodes):
        episodes_rewards = []
        done = False
        # The env is reset auto
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, _info = vec_env.step(action)
            episodes_rewards.append(reward)
            env.render()
        all_episodes_rewards.append(sum(all_episodes_rewards))
    mean_episode_reward = np.mean(all_episodes_rewards)
    print(f"Mean reward is: {mean_episode_reward:.2f} +/- Num episodes: {num_episodes}")
    return mean_episode_reward


# Evaluate a random agent before training
mean_reward_before_training = evaluate(model, num_episodes=100, deterministic=True)
# Same result as before just simpler since it's given by the stable baseline
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, warn=False)
print(f"Mean reward is: {mean_reward:.2f} +/- Num episodes: {std_reward:.2f}")
### TRAINING

model.learn(total_timesteps=total_timesteps)
model.save("highway_ppo/model")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"Mean reward is: {mean_reward:.2f} +/- Num episodes: {std_reward:.2f}")
model = PPO.load("highway_ppo/model")
