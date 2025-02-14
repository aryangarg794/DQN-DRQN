import os
import argparse
import numpy as np
import time
import torch
import torch.nn as nn

from models.dqn import DQN
from utils.replay import load_buffer, MetricTracker
from utils.train import train

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument('-s', '--steps', type=int, default=1000000, help='number of steps')
parser.add_argument('-save', '--save', type=int, default=500000, help='when to save')
parser.add_argument('-l', '--lr', type=float, default=2.5e-4, help='learning rate')
parser.add_argument('-c', '--C', type=int, default=1000, help='when to update the target network')
parser.add_argument('-cap', '--capacity', type=int, default=100000, help='capacity of the experience replay')
parser.add_argument('-p', '--preload', type=int, default=10000, help='amount of experience to preload in the buffer ')
parser.add_argument('-f', '--freq', type=int, default=4, help='at which frame to do update step')
parser.add_argument('-fsteps', '--finalsteps', type=int, default=500000, help='when to stop the epsilon')
parser.add_argument('-b', '--batch', type=int, default=64, help='batch size')
parser.add_argument('-d', '--discount', type=float, default=0.95, help='discount factor')
parser.add_argument('-m', '--model', type=str, default='DQN', help='model to train')
parser.add_argument('-e', '--env', type=str, default='PongNoFrameskip-v4', help='select atari game to run on')
parser.add_argument('-dc', '--decaystart', type=int, default=0, help='at which point to start epsilon decay')
parser.add_argument('-dev', '--device', type=str, default='cuda', help='device')


args = parser.parse_args()
args_dict = vars(args)

model_name = args.model
preload = args.preload
capacity = args.capacity
atari_env = args.env
device = args.device

metrics = MetricTracker()
terminal_width = os.get_terminal_size().columns


def print_block(text):    
    padding = (terminal_width - len(text)) // 2
    print('\n' + '=' * terminal_width)
    print(' ' * padding + text)
    print('=' * terminal_width + '\n')

if __name__ == "__main__":
    
    if model_name == 'DRQN':
        raise NotImplementedError('Still being developed :(')
    
    
    print_block(f'Training Model {model_name}')
    
    
    for setting, value in args_dict.items():
        setting_text = f'{setting}: {value}'
        padding_item = (terminal_width - len(setting_text)) // 2
        print(' ' * padding_item + setting_text) 
        
    print_block(f'Loading Buffer...')
    buffer, env = load_buffer(preload, capacity, game=atari_env, device=device)
    
    if model_name == 'DQN':
        q_model = DQN(env, decay_steps=args.finalsteps).to(device)
        target_model = DQN(env, decay_steps=args.finalsteps).to(device)
        target_model.load_state_dict(q_model.state_dict())
        
    optimizer = torch.optim.Adam(q_model.parameters(), lr=args.lr)
    
    print_block(f'Training Loop...')
    
    train(
        env=env,
        q_network=q_model, 
        name=atari_env,
        target_network=target_model, 
        optimizer=optimizer, 
        timesteps=args.steps, 
        replay=buffer, 
        metrics=metrics, 
        train_freq=args.freq, 
        batch_size=args.batch, 
        gamma=args.discount, 
        decay_start=args.decaystart,
        C=args.C,
        device=device,
        save_step=args.save 
    )
    
    print_block(f'Training Complete!')