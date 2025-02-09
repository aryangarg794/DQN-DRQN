# Implementation of DQN (2013) and DRQN (2015)
---------------------------

To run the repo ensure that you have the following libraries installed. 

```
tqdm 
torch
ale
gymnasium
numpy
matplotlib
stable_baselines3
```
If you have `uv` installed you can just run the following command. 

```
uv venv
source .venv/bin/activate # .venv\Scripts\activate on Win
uv sync
```

--------------------------

To train an agent please run 

```
python src/main.py 
```
The possible CLI arguments include: 

USAGE:
```bash
python src/main.py [FLAGS] [OPTIONS]

OPTIONS:

-s, --steps <integer>        Number of steps to run during training. (Default: 1000000)
-save, --save <integer>      The step at which to save the model. (Default: 500000)
-l, --lr <float>             Learning rate for training. (Default: 2.5e-4)
-c, --C <integer>            Frequency (in steps) to update the target network. (Default: 1000)
-cap, --capacity <integer>   Capacity of the experience replay buffer. (Default: 100000)
-p, --preload <integer>      Number of experiences to preload into the buffer. (Default: 10000)
-f, --freq <integer>         Frequency (in frames) at which a training update is performed. (Default: 4)
-fsteps, --finalsteps <integer>  Step count at which epsilon and learning rate decay stop. (Default: 500000)
-b, --batch <integer>        Batch size used during training. (Default: 64)
-d, --discount <float>       Discount factor (gamma) used in Q-learning. (Default: 0.95)
-m, --model <string>         Model type to train (e.g., "DQN"). (Default: "DQN")
-e, --env <string>           Atari game environment to run (e.g., "PongNoFrameskip-v4"). (Default: "PongNoFrameskip-v4")
-dc, --decaystart <integer>  Step at which to begin epsilon decay. (Default: 0)
-dev, --device <string>      Device on which to run training (e.g., "cuda" or "cpu"). (Default: "cuda")


```

---------------------------
The models are named with the number of timesteps in mind i.e. `pong_dqn_best_6M` indicates that the agent was trained for 6 million steps. 

### Examples

![image info](/media/pong.gif)

![image info](/media/breakout.gif)

--------------------------
## References 
[1] Mnih, Volodymyr. "Playing atari with deep reinforcement learning." arXiv preprint arXiv:1312.5602 (2013).\
[2] Mnih, V., Kavukcuoglu, K., Silver, D. et al. Human-level control through deep reinforcement learning. Nature 518, 529–533 (2015). https://doi.org/10.1038/nature14236 \
[3] Bailey, Jay. Deep Q-Networks Explained. 13 Sept. 2022, www.lesswrong.com/posts/kyvCNgx9oAwJCuevo/deep-q-networks-explained. \
[4] Hausknecht, M., & Stone, P. (2015). Deep Recurrent Q-Learning for Partially Observable MDPs. arXiv preprint arXiv:1507.06527. 