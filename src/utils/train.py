import torch
import time
import torch.nn as nn
import numpy as np

def train(
    env, 
    name, 
    q_network, 
    target_network, 
    optimizer, 
    timesteps, 
    replay, 
    metrics, 
    train_freq, 
    batch_size, 
    decay_start,
    C,
    gamma=0.99,
    device='cuda',
    save_step=850000,
):
    loss_func = nn.MSELoss()
    start_time = time.time()
    episode_count = 0
    best_avg_reward = -float('inf')
    
    obs, _ = env.reset()
    
    for step in range(1, timesteps+1):
        batched_obs = np.expand_dims(obs.squeeze(), axis=0)
        action = q_network.epsilon_greedy(torch.as_tensor(batched_obs, dtype=torch.float32, device=device)).cpu().item()
        obs_prime, reward, terminated, truncated, _ = env.step(action)

        
        replay.store((obs.squeeze(), action, reward, obs_prime.squeeze(), terminated or truncated))
        metrics.add_step_reward(reward)
        obs = obs_prime

        if step % train_freq == 0:
            observations, actions, rewards, observation_primes, dones = replay.sample(batch_size)
            
            with torch.no_grad():
                q_values_minus = target_network(observation_primes)
                boostrapped_values = torch.amax(q_values_minus, dim=1, keepdim=True)

            y_trues = torch.where(dones, rewards, rewards + gamma * boostrapped_values)
            y_preds = q_network(observations)

            loss = loss_func(y_preds.gather(1, actions), y_trues)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        scheduler.step()
            
        if step > decay_start: 
            q_network.epsilon_decay(step)
            target_network.epsilon_decay(step)
        
        if terminated or truncated:
            elapsed_time = time.time() - start_time
            steps_per_sec = step / elapsed_time
            metrics.end_episode()
            episode_count += 1
            
            obs, _ = env.reset()
            
            if metrics.avg_reward > best_avg_reward and step > save_step:
                best_avg_reward = metrics.avg_reward
                torch.save({
                    'step': step,
                    'model_state_dict': q_network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'avg_reward': metrics.avg_reward,
                }, f"models/{name}_dqn_best_{step}.pth")
                
            print(f"\rStep: {step:,}/{timesteps:,} | "
                    f"Episodes: {episode_count} | "
                    f"Avg Reward: {metrics.avg_reward:.1f} | "
                    f"Epsilon: {q_network.epsilon:.3f} | "
                    f"Steps/sec: {steps_per_sec:.1f}", end="\r")
            
        if step % C == 0:
            target_network.load_state_dict(q_network.state_dict())