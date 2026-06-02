import argparse
import time
from collections import deque
import random
import os
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

    def state_dict(self):
        return {
            "buffer": list(self._buffer),
        }

    def load_state_dict(self, state_dict):
        self._buffer = deque(state_dict["buffer"], maxlen=self._capacity)

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

class DuelingQNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.advantage_fc1 = nn.Linear(64 * 7 * 7, 512)
        self.advantage_fc2 = nn.Linear(512, n_actions)
        self.value_fc1 = nn.Linear(64 * 7 * 7, 512)
        self.value_fc2 = nn.Linear(512, 1)
        

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        adv = nn.functional.relu(self.advantage_fc1(x))
        adv = self.advantage_fc2(adv)
        val = nn.functional.relu(self.value_fc1(x))
        val = self.value_fc2(val)
        return val + adv - adv.mean(dim=1, keepdim=True)

def reset_env(env, max_no_op_steps):
    no_op_steps = random.randint(1, max_no_op_steps)
    obs, info = env.reset()
    for _ in range(no_op_steps):
        prev_obs = obs
        obs, _, _, _, _ = env.step(0)
    processed = preprocess_obs(prev_obs, obs)
    frame_stack = deque([processed] * 4, maxlen=4)
    return obs, prev_obs, frame_stack

def save_checkpoint(path, step, q_net, target_q_net, optimizer, replay_buffer):
    torch.save({
        "step": step,
        "q_net_state_dict": q_net.state_dict(),
        "target_q_net_state_dict": target_q_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "replay_buffer_state_dict": replay_buffer.state_dict(),
    }, path)

def load_checkpoint(
        path, 
        device,
        config
    ) -> tuple[int, QNetwork, QNetwork, torch.optim.Optimizer, ReplayBuffer]:
    checkpoint = torch.load(path)
    step = checkpoint["step"]
    if config["algorithm"] == "dqn" or config["algorithm"] == "ddqn":
        q_net = QNetwork(n_actions=4).to(device)
        q_net.load_state_dict(checkpoint["q_net_state_dict"])
        target_q_net = QNetwork(n_actions=4).to(device)
        target_q_net.load_state_dict(checkpoint["target_q_net_state_dict"])
    elif config["algorithm"] == "dueling-dqn":
        q_net = DuelingQNetwork(n_actions=4).to(device)
        q_net.load_state_dict(checkpoint["q_net_state_dict"])
        target_q_net = DuelingQNetwork(n_actions=4).to(device)
        target_q_net.load_state_dict(checkpoint["target_q_net_state_dict"])
    else:
        raise ValueError(f"Invalid algorithm: {config['algorithm']}")
    optimizer = torch.optim.RMSprop(
        q_net.parameters(),
        lr=config["learning_rate"],
        momentum=config["gradient_momentum"],
        alpha=config["squared_gradient_momentum"],
        eps=config["min_squared_gradient"]
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    replay_buffer = ReplayBuffer(capacity=config["replay_buffer_capacity"])
    replay_buffer.load_state_dict(checkpoint["replay_buffer_state_dict"])
    return step, q_net, target_q_net, optimizer, replay_buffer

def main(algorithm, n_steps_td):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("checkpoints", exist_ok=True)

    config = {
        "algorithm": algorithm,
        "discount_factor": 0.99,
        "epsilon_start": 1.0,
        "epsilon_decay_steps": 1_000_000,
        "epsilon_end": 0.1,
        "gradient_momentum": 0.95,
        "learning_rate": 0.00025,
        "min_squared_gradient": 0.01,
        "mini_batch_size": 32,
        "n_steps_td": n_steps_td,
        "no_op_max_steps": 30,
        "replay_start_size": 50_000,
        "replay_buffer_capacity": 1_000_000,
        "squared_gradient_momentum": 0.95,
        "target_net_update_freq": 10_000,
        "training_steps": 10_000_000,
        "update_freq": 4,
        "checkpoint_freq": 1_000_000,
    }

    if algorithm == "dqn" or algorithm == "ddqn":
        q_net = QNetwork(n_actions=4).to(device)
        target_q_net = QNetwork(n_actions=4).to(device)
    elif algorithm == "dueling-dqn":
        q_net = DuelingQNetwork(n_actions=4).to(device)
        target_q_net = DuelingQNetwork(n_actions=4).to(device)
    else:
        raise ValueError(f"Invalid algorithm: {algorithm}")
    target_q_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.RMSprop(
        q_net.parameters(),
        lr=config["learning_rate"],
        momentum=config["gradient_momentum"],
        alpha=config["squared_gradient_momentum"],
        eps=config["min_squared_gradient"]
    )
    criterion = nn.SmoothL1Loss()

    replay_buffer = ReplayBuffer(capacity=config["replay_buffer_capacity"])

    n_steps_td_buffer = list()

    wandb.init(
        project="dqn-atari", name=f"{algorithm}-breakout",
        config=config
    )

    env = gym.make("ALE/Breakout-v5")#, render_mode="human")

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    obs, prev_obs, frame_stack = reset_env(env, config["no_op_max_steps"])
    prev_stacked_obs = np.stack(frame_stack)
    print(f"Observation shape: {obs.shape}")

    total_reward = 0
    episode_count = 0
    loss_history = deque(maxlen=100)
    for step in range(config["training_steps"]):
        epsilon = max(config["epsilon_end"], config["epsilon_start"] - (config["epsilon_start"] - config["epsilon_end"]) * (step / config["epsilon_decay_steps"]))
        if len(replay_buffer) < config["replay_start_size"]:
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

        processed_obs = preprocess_obs(obs, prev_obs)
        frame_stack.append(processed_obs)
        stacked_obs = np.stack(frame_stack)
        reward = np.clip(reward, -1, 1)
        n_steps_td_buffer.append((prev_stacked_obs, action, reward, stacked_obs, terminated or truncated))

        if terminated or truncated:
            episode_count += 1

            truncated_return = 0
            _, _, _, target_stacked_obs, _ = n_steps_td_buffer[-1]
            for i in range(len(n_steps_td_buffer)):
                stacked_obs, action, reward, _, _ = n_steps_td_buffer[-i-1]
                truncated_return = reward + config["discount_factor"] * truncated_return
                replay_buffer.push(stacked_obs, action, truncated_return, target_stacked_obs, done=True)

            print(f"Episode ended, resetting. Total reward: {total_reward}")
            wandb.log(
                {"episode_reward": total_reward, "episode_count": episode_count},
                step=step
            )
            obs, prev_obs, frame_stack = reset_env(env, config["no_op_max_steps"])
            prev_stacked_obs = np.stack(frame_stack)
            total_reward = 0
            n_steps_td_buffer.clear()
            continue

        if len(n_steps_td_buffer) >= n_steps_td:
            stacked_obs, action, reward, next_stacked_obs, done = n_steps_td_buffer.pop(0)
            truncated_return = reward
            for i in range(n_steps_td - 1):
                _, _, rwd, _, _ = n_steps_td_buffer[i]
                truncated_return += rwd * (config["discount_factor"] ** (i + 1))
            
            if n_steps_td == 1:
                n_steps_next_stacked_obs = next_stacked_obs
            else:
                _, _, _, n_steps_next_stacked_obs, _ = n_steps_td_buffer[n_steps_td - 2]
            replay_buffer.push(stacked_obs, action, truncated_return, n_steps_next_stacked_obs, done)

        prev_obs = obs
        prev_stacked_obs = stacked_obs

        if len(replay_buffer) >= config["replay_start_size"]:
            if step % config["update_freq"] == 0:
                stacked_obs, actions, truncated_returns, n_steps_next_stacked_obs, dones = replay_buffer.sample(batch_size=config["mini_batch_size"])
                actions = torch.tensor(actions, dtype=torch.long).to(device)
                truncated_returns = torch.tensor(truncated_returns, dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).to(device)
                obs_tensor = torch.tensor(stacked_obs, dtype=torch.float32) / 255.0
                obs_tensor = obs_tensor.to(device)
                n_steps_next_obs_tensor = torch.tensor(n_steps_next_stacked_obs, dtype=torch.float32) / 255.0
                n_steps_next_obs_tensor = n_steps_next_obs_tensor.to(device)
                q_values = q_net(obs_tensor)
                q_values = q_values.gather(1, actions.unsqueeze(1))
                q_values = q_values.squeeze(1)
                with torch.no_grad():
                    if algorithm == "dqn":
                        next_q = target_q_net(n_steps_next_obs_tensor)
                        max_next_q = next_q.max(dim=1)[0]
                        target_q_values = truncated_returns + (1 - dones) * (config["discount_factor"] ** config["n_steps_td"]) * max_next_q                
                    elif algorithm == "ddqn" or algorithm == "dueling-dqn":
                        next_q_online = q_net(n_steps_next_obs_tensor)                                                                       
                        best_actions = next_q_online.argmax(dim=1)
                        next_q_target = target_q_net(n_steps_next_obs_tensor)                                                                
                        max_next_q = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
                        target_q_values = truncated_returns + (1 - dones) * (config["discount_factor"] ** config["n_steps_td"]) * max_next_q
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
            if step % config["target_net_update_freq"] == 0:
                target_q_net.load_state_dict(q_net.state_dict())
        if step % config["checkpoint_freq"] == 0 and step > 0:
            new_path = f"checkpoints/{algorithm}-breakout-{step}.pt"
            save_checkpoint(new_path, step, q_net, target_q_net, optimizer, replay_buffer)
            print(f"Checkpoint saved to {new_path}")
            prev_step = step - config["checkpoint_freq"]
            prev_path = f"checkpoints/{algorithm}-breakout-{prev_step}.pt"
            print(f"Removing checkpoint from {prev_path}")
            if os.path.exists(prev_path):
                os.remove(prev_path)
                print(f"Checkpoint removed from {prev_path}")
    env.close()
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="ddqn", choices=["dqn", "ddqn", "dueling-dqn"])
    parser.add_argument("--n-steps-td", type=int, default=1)
    args = parser.parse_args()
    main(args.algorithm, args.n_steps_td)
