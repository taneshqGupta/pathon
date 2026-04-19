import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import torch.multiprocessing as mp
import pathon_env
import os
import json
from collections import deque
import time

# Maximize Ampere Tensor Core performance
torch.backends.cudnn.benchmark = True

# --- 1. Shared Memory Worker Logic ---
class PathonEnvWrapper:
    def __init__(self, max_steps=1000):
        self.env = pathon_env.GridEnv(300, 200)
        self.max_steps = max_steps
        self.step_count = 0
        self.fov_hist = deque(maxlen=4)
        self.lidar_hist = deque(maxlen=4)
        self.grad_hist = deque(maxlen=4)

    def reset(self):
        self.step_count = 0
        fov_raw, lidar_raw, grad_raw = self.env.reset_env()
        fov_np = np.frombuffer(fov_raw, dtype=np.uint8).reshape((15, 15)).copy()
        lidar_np = np.array(lidar_raw, dtype=np.float32)
        grad_np = np.array(grad_raw, dtype=np.float32)

        for _ in range(4):
            self.fov_hist.append(fov_np)
            self.lidar_hist.append(lidar_np)
            self.grad_hist.append(grad_np)
        
        return np.stack(self.fov_hist, axis=0), np.concatenate(self.lidar_hist, axis=0), np.concatenate(self.grad_hist, axis=0)

    def step(self, action):
        self.step_count += 1
        fov_raw, lidar_raw, grad_raw, reward, terminated, reached_goal = self.env.step_action(int(action))
        
        fov_np = np.frombuffer(fov_raw, dtype=np.uint8).reshape((15, 15)).copy()
        lidar_np = np.array(lidar_raw, dtype=np.float32)
        grad_np = np.array(grad_raw, dtype=np.float32)

        self.fov_hist.append(fov_np)
        self.lidar_hist.append(lidar_np)
        self.grad_hist.append(grad_np)

        truncated = self.step_count >= self.max_steps
        done = terminated or truncated
        
        return np.stack(self.fov_hist, axis=0), np.concatenate(self.lidar_hist, axis=0), np.concatenate(self.grad_hist, axis=0), float(reward), done, reached_goal

def shared_worker(env_idx, remote, parent_remote, shared_fov, shared_lidar, shared_grad, shared_reward, shared_done, shared_goal):
    parent_remote.close()
    env = PathonEnvWrapper(max_steps=1000)
    
    while True:
        cmd, action = remote.recv()
        if cmd == 'step':
            fov, lidar, grad, reward, done, reached_goal = env.step(action)
            if done:
                fov, lidar, grad = env.reset()
            
            # Write directly to RAM. No IPC transfer.
            shared_fov[env_idx] = torch.from_numpy(fov)
            shared_lidar[env_idx] = torch.from_numpy(lidar)
            shared_grad[env_idx] = torch.from_numpy(grad)
            shared_reward[env_idx] = reward
            shared_done[env_idx] = float(done)
            shared_goal[env_idx] = float(reached_goal)
            
            remote.send(True)
        
        elif cmd == 'reset':
            fov, lidar, grad = env.reset()
            shared_fov[env_idx] = torch.from_numpy(fov)
            shared_lidar[env_idx] = torch.from_numpy(lidar)
            shared_grad[env_idx] = torch.from_numpy(grad)
            remote.send(True)
            
        elif cmd == 'close':
            remote.close()
            break

# --- 2. Shared Vector Environment ---
class SharedMemoryVecEnv:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        
        # Pre-allocate RAM and share memory across processes
        self.shared_fov = torch.zeros((num_envs, 4, 15, 15), dtype=torch.float32).share_memory_()
        self.shared_lidar = torch.zeros((num_envs, 32), dtype=torch.float32).share_memory_()
        self.shared_grad = torch.zeros((num_envs, 8), dtype=torch.float32).share_memory_()
        self.shared_reward = torch.zeros((num_envs,), dtype=torch.float32).share_memory_()
        self.shared_done = torch.zeros((num_envs,), dtype=torch.float32).share_memory_()
        self.shared_goal = torch.zeros((num_envs,), dtype=torch.float32).share_memory_()

        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes = []
        for i in range(num_envs):
            p = mp.Process(
                target=shared_worker, 
                args=(i, self.work_remotes[i], self.remotes[i], self.shared_fov, self.shared_lidar, self.shared_grad, self.shared_reward, self.shared_done, self.shared_goal)
            )
            p.daemon = True
            p.start()
            self.processes.append(p)
            
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        
        # Wait for tiny confirmation signals
        [remote.recv() for remote in self.remotes]
        
        return (self.shared_fov.clone(), self.shared_lidar.clone(), self.shared_grad.clone()), self.shared_reward.clone(), self.shared_done.clone(), self.shared_goal.clone()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        [remote.recv() for remote in self.remotes]
        return self.shared_fov.clone(), self.shared_lidar.clone(), self.shared_grad.clone()

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()

# --- 3. Neural Network ---
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(1608, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.actor = nn.Linear(256, 8)
        self.critic = nn.Linear(256, 1)

    def forward(self, fov, lidar, grad):
        fov_norm = fov / 255.0
        cnn_out = self.cnn(fov_norm)
        combined = torch.cat([cnn_out, lidar, grad], dim=1)
        features = self.fc(combined)
        return self.actor(features), self.critic(features)

    def get_action_and_value(self, fov, lidar, grad, action=None):
        logits, value = self.forward(fov, lidar, grad)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value

# --- 4. Training Loop ---
def train_ppo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device.type.upper()} with Shared CPU Memory & AMP")

    num_envs = 30 
    env = SharedMemoryVecEnv(num_envs=num_envs)
    
    agent = ActorCritic().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)
    scaler = torch.amp.GradScaler('cuda') # Automatic Mixed Precision Scaler

    total_timesteps = 2_000_000
    num_steps = 1024 
    batch_size = 4096 
    n_epochs = 10
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    record_freq = 300000
    replay_dir = "./replays"
    os.makedirs(replay_dir, exist_ok=True)
    
    # Storage Buffers in VRAM
    obs_fov = torch.zeros((num_steps, num_envs, 4, 15, 15), dtype=torch.float32, device=device)
    obs_lidar = torch.zeros((num_steps, num_envs, 32), dtype=torch.float32, device=device)
    obs_grad = torch.zeros((num_steps, num_envs, 8), dtype=torch.float32, device=device)
    actions = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    logprobs = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    dones = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    values = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    
    # Metric Tracking Buffers
    ep_returns = deque(maxlen=100)
    ep_lengths = deque(maxlen=100)
    ep_goals = deque(maxlen=100)
    current_returns = np.zeros(num_envs)
    current_lengths = np.zeros(num_envs)

    global_step = 0
    start_time = time.time()
    
    next_fov, next_lidar, next_grad = env.reset()
    next_fov = next_fov.to(device, non_blocking=True)
    next_lidar = next_lidar.to(device, non_blocking=True)
    next_grad = next_grad.to(device, non_blocking=True)
    next_done = torch.zeros(num_envs, device=device)

    num_updates = total_timesteps // (num_steps * num_envs)

    for update in range(1, num_updates + 1):
        for step in range(num_steps):
            global_step += num_envs
            obs_fov[step] = next_fov
            obs_lidar[step] = next_lidar
            obs_grad[step] = next_grad
            dones[step] = next_done

            with torch.no_grad(), torch.amp.autocast('cuda'):
                action, logprob, _, value = agent.get_action_and_value(next_fov, next_lidar, next_grad)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            (raw_fov, raw_lidar, raw_grad), reward, done, reached_goal = env.step(action.cpu().numpy())
            
            # Metric Tracking
            current_returns += reward.numpy()
            current_lengths += 1
            done_np = done.numpy()
            goal_np = reached_goal.numpy()
            
            for i in range(num_envs):
                if done_np[i]:
                    ep_returns.append(current_returns[i])
                    ep_lengths.append(current_lengths[i])
                    ep_goals.append(goal_np[i])
                    current_returns[i] = 0
                    current_lengths[i] = 0

            next_fov = raw_fov.to(device, non_blocking=True)
            next_lidar = raw_lidar.to(device, non_blocking=True)
            next_grad = raw_grad.to(device, non_blocking=True)
            
            rewards[step] = reward.to(device, non_blocking=True)
            next_done = done.to(device, non_blocking=True)

            # --- REPLAY RECORDING BLOCK ---
            if global_step % record_freq < num_envs:
                print(f"\n[Callback] Recording evaluation episode at step {global_step}...")
                record_env = pathon_env.GridEnv(300, 200)
                record_env.reset_env()
                frames = []
                r_step, r_done = 0, False
                r_fov_hist, r_lidar_hist, r_grad_hist = deque(maxlen=4), deque(maxlen=4), deque(maxlen=4)
                
                fv, lv, (gx, gy) = record_env.reset_env()
                for _ in range(4):
                    r_fov_hist.append(np.frombuffer(fv, dtype=np.uint8).reshape((15, 15)).copy())
                    r_lidar_hist.append(np.array(lv, dtype=np.float32))
                    r_grad_hist.append(np.array([gx, gy], dtype=np.float32))
                
                while not r_done and r_step < 1000:
                    st_fov = torch.tensor(np.stack(r_fov_hist, axis=0), dtype=torch.float32).unsqueeze(0).to(device)
                    st_lidar = torch.tensor(np.concatenate(r_lidar_hist, axis=0), dtype=torch.float32).unsqueeze(0).to(device)
                    st_grad = torch.tensor(np.concatenate(r_grad_hist, axis=0), dtype=torch.float32).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        r_act, _, _, _ = agent.get_action_and_value(st_fov, st_lidar, st_grad)
                    
                    fv, lv, (gx, gy), _, term, _ = record_env.step_action(int(r_act.item()))
                    r_done = term
                    frames.append(json.loads(record_env.get_state_json()))
                    
                    r_fov_hist.append(np.frombuffer(fv, dtype=np.uint8).reshape((15, 15)).copy())
                    r_lidar_hist.append(np.array(lv, dtype=np.float32))
                    r_grad_hist.append(np.array([gx, gy], dtype=np.float32))
                    r_step += 1
                
                filepath = os.path.join(replay_dir, f"replay_step_{global_step}.json")
                with open(filepath, "w") as f:
                    json.dump(frames, f)
                print(f"[Callback] Replay saved: {filepath} ({len(frames)} frames)\n")

        with torch.no_grad(), torch.amp.autocast('cuda'):
            _, next_value = agent(next_fov, next_lidar, next_grad)
            next_value = next_value.flatten()
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs_fov = obs_fov.reshape((-1, 4, 15, 15))
        b_obs_lidar = obs_lidar.reshape((-1, 32))
        b_obs_grad = obs_grad.reshape((-1, 8))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(num_steps * num_envs)
        for epoch in range(n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, num_steps * num_envs, batch_size):
                end = start + batch_size
                mb_inds = b_inds[start:end]

                # Utilize AMP for the entire backward pass
                with torch.amp.autocast('cuda'):
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs_fov[mb_inds], b_obs_lidar[mb_inds], b_obs_grad[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    approx_kl = ((ratio - 1) - logratio).mean()

                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()

        sps = int(global_step / (time.time() - start_time))
        avg_ret = np.mean(ep_returns) if ep_returns else 0.0
        avg_len = np.mean(ep_lengths) if ep_lengths else 0.0
        goal_rate = np.mean(ep_goals) if ep_goals else 0.0
        
        print(f"Update: {update}/{num_updates} | Steps: {global_step} | SPS: {sps} | "
              f"Ret: {avg_ret:.2f} | Len: {avg_len:.1f} | Goal: {goal_rate*100:.1f}% | "
              f"v_loss: {v_loss.item():.4f} | approx_kl: {approx_kl.item():.4f}")

    env.close()
    return agent

class OnnxWrapper(nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.model = actor_critic
    def forward(self, fov, lidar, grad):
        logits, _ = self.model(fov, lidar, grad)
        return torch.argmax(logits, dim=1)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    trained_agent = train_ppo()
    
    # Save PyTorch weights
    torch.save(trained_agent.state_dict(), "ppo_pathon.pth")
    
    # Export to ONNX
    onnx_model = OnnxWrapper(trained_agent).cpu()
    dummy_obs = {
        "fov": torch.zeros((1, 4, 15, 15), dtype=torch.float32),
        "lidar": torch.zeros((1, 32), dtype=torch.float32),
        "grad": torch.zeros((1, 8), dtype=torch.float32),
    }
    torch.onnx.export(
        onnx_model,
        (dummy_obs["fov"], dummy_obs["lidar"], dummy_obs["grad"]),
        "backend/model.onnx",
        opset_version=14,
        input_names=["fov", "lidar", "grad"],
        output_names=["action"],
    )
    print("Successfully exported parallel PyTorch model to ONNX.")