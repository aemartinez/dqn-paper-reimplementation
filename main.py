import argparse
import time
from collections import deque
import random

import numpy as np
import ale_py
import gymnasium as gym
import cv2
import torch
import torch.nn as nn
import wandb
gym.register_envs(ale_py)

def preprocess_obs(obs0, obs1):
    maxed = np.max(np.array([obs0, obs1]), axis=0)
    gray = cv2.cvtColor(maxed, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (84, 84))

class ReplayBuffer:                                                                                          
    def __init__(self, capacity):
        self._capacity = capacity
        self._buffer = deque(maxlen=capacity)                                                                                        
                
    def push(self, stacked_obs, action, reward, next_stacked_obs, done):
        obs0, obs1, obs2, obs3 = stacked_obs
        _, _, _, obs4 = next_stacked_obs
        self._buffer.append((obs0, obs1, obs2, obs3, obs4, action, reward, done))
                                                                                                            
    def sample(self, batch_size):
        samples = random.sample(self._buffer, batch_size)
        obs0, obs1, obs2, obs3, obs4, actions, rewards, dones = zip(*samples)
        stacked_obs = np.stack([np.stack([o0, o1, o2, o3]) for o0, o1, o2, o3 in zip(obs0, obs1, obs2, obs3)])
        next_stacked_obs = np.stack([np.stack([o1, o2, o3, o4]) for o1, o2, o3, o4 in zip(obs1, obs2, obs3, obs4)])
        return (
            stacked_obs,
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            next_stacked_obs,
            np.array(dones, dtype=bool),
        )
        
    def capacity(self):
        return self._capacity

    def __len__(self):                                                                                       
        return len(self._buffer)

class QNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        x = nn.functional.relu(self.fc1(x))
        return self.fc2(x)

def reset_env(env):
    obs, info = env.reset()
    prev_obs = obs
    processed = preprocess_obs(obs, obs)
    frame_stack = deque([processed] * 4, maxlen=4)
    return obs, prev_obs, frame_stack


def main(algorithm):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_steps = 10_000_000
    q_net = QNetwork(n_actions=4).to(device)
    target_q_net = QNetwork(n_actions=4).to(device)
    target_q_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.RMSprop(q_net.parameters(), lr=0.00025, momentum=0.95, alpha=0.95, eps=0.01)
    criterion = nn.SmoothL1Loss()
    epsilon_start = 1.0
    epsilon_decay_steps = 100_000
    epsilon_end = 0.1
    target_update_interval = 1000
    discount_factor = 0.99
    replay_buffer_capacity = 1_000_000

    replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)
    replay_start_size = 50_000

    wandb.init(
        project="dqn-atari", name=f"{algorithm}-breakout",
        config={
            "training_steps": training_steps,
            "epsilon_start": epsilon_start,
            "epsilon_decay_steps": epsilon_decay_steps,
            "epsilon_end": epsilon_end,
            "target_update_interval": target_update_interval,
            "discount_factor": discount_factor,
            "replay_start_size": replay_start_size,
            "replay_buffer_capacity": replay_buffer_capacity,
            "algorithm": algorithm,
        }
    )

    env = gym.make("ALE/Breakout-v5")#, render_mode="human")

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    obs, prev_obs, frame_stack = reset_env(env)
    prev_stacked_obs = np.stack(frame_stack)
    print(f"Observation shape: {obs.shape}")

    total_reward = 0
    episode_count = 0
    loss_history = deque(maxlen=100)
    for step in range(training_steps):
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (step / epsilon_decay_steps))
        if len(replay_buffer) < replay_start_size:
            action = env.action_space.sample()
        elif random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(prev_stacked_obs, dtype=torch.float32).unsqueeze(0) / 255.0
                obs_tensor = obs_tensor.to(device)
                q_values = q_net(obs_tensor)
                action = q_values.squeeze(0).argmax().item()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        # print(f"Step {step + 1:3d} | action={action} | reward={reward} | shape={obs.shape}")
        # time.sleep(0.05)

        if terminated or truncated:
            episode_count += 1
            print(f"Episode ended, resetting. Total reward: {total_reward}")
            wandb.log(
                {"episode_reward": total_reward, "episode_count": episode_count},
                step=step
            )
            obs, prev_obs, frame_stack = reset_env(env)
            prev_stacked_obs = np.stack(frame_stack)
            total_reward = 0
            continue

        processed_obs = preprocess_obs(obs, prev_obs)
        frame_stack.append(processed_obs)
        stacked_obs = np.stack(frame_stack)
        reward = np.clip(reward, -1, 1)
        replay_buffer.push(prev_stacked_obs, action, reward, stacked_obs, terminated or truncated)
        prev_obs = obs
        prev_stacked_obs = stacked_obs

        if len(replay_buffer) >= replay_start_size:
            stacked_obs, actions, rewards, next_stacked_obs, dones = replay_buffer.sample(batch_size=32)
            actions = torch.tensor(actions, dtype=torch.long).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            dones = torch.tensor(dones, dtype=torch.float32).to(device)
            obs_tensor = torch.tensor(stacked_obs, dtype=torch.float32) / 255.0
            obs_tensor = obs_tensor.to(device)
            next_obs_tensor = torch.tensor(next_stacked_obs, dtype=torch.float32) / 255.0
            next_obs_tensor = next_obs_tensor.to(device)
            q_values = q_net(obs_tensor)
            q_values = q_values.gather(1, actions.unsqueeze(1))
            q_values = q_values.squeeze(1)
            with torch.no_grad():
                if algorithm == "dqn":
                    next_q = target_q_net(next_obs_tensor)
                    max_next_q = next_q.max(dim=1)[0]
                    target_q_values = rewards + (1 - dones) * discount_factor * max_next_q                
                elif algorithm == "ddqn":
                    next_q_online = q_net(next_obs_tensor)                                                                       
                    best_actions = next_q_online.argmax(dim=1)
                    next_q_target = target_q_net(next_obs_tensor)                                                                
                    max_next_q = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
                    target_q_values = rewards + (1 - dones) * discount_factor * max_next_q
            loss = criterion(q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())

            if step % 100 == 0:
                print(f"Step {step + 1:3d} | loss={np.mean(loss_history):.4f}")
                wandb.log(
                    {
                        "loss": np.mean(loss_history),
                        "epsilon": epsilon,
                        "buffer_size": len(replay_buffer),
                        "episode_count": episode_count,
                        "mean_target_q": target_q_values.mean().item(),
                    },
                    step=step
                )
            if step % target_update_interval == 0:
                target_q_net.load_state_dict(q_net.state_dict())

    env.close()
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="ddqn", choices=["dqn", "ddqn"])
    args = parser.parse_args()
    main(args.algorithm)
