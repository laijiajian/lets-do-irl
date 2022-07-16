import os
import gym
import pickle
import argparse
import numpy as np
from collections import deque

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter 
from ppo_algo.PPO import PPO
from utils.utils import *
from utils.zfilter import ZFilter
from model import  Discriminator
from train_model import train_discrim

parser = argparse.ArgumentParser(description='PyTorch GAIL')
parser.add_argument('--env_name', type=str, default="Hopper-v2", 
                    help='name of the environment to run')
parser.add_argument('--load_model', type=str, default='ckpt_1091_3814.pth.tar',
                    help='path to load the saved model')
parser.add_argument('--render', action="store_true", default=True,
                    help='if you dont want to render, set this to False')
parser.add_argument('--gamma', type=float, default=0.99, 
                    help='discounted factor (default: 0.99)')
parser.add_argument('--lamda', type=float, default=0.98, 
                    help='GAE hyper-parameter (default: 0.98)')
parser.add_argument('--hidden_size', type=int, default=100, 
                    help='hidden unit size of actor, critic and discrim networks (default: 100)')
parser.add_argument('--learning_rate', type=float, default=3e-4, 
                    help='learning rate of models (default: 3e-4)')
parser.add_argument('--l2_rate', type=float, default=1e-3, 
                    help='l2 regularizer coefficient (default: 1e-3)')
parser.add_argument('--clip_param', type=float, default=0.2, 
                    help='clipping parameter for PPO (default: 0.2)')
parser.add_argument('--discrim_update_num', type=int, default=2, 
                    help='update number of discriminator (default: 2)')
parser.add_argument('--actor_critic_update_num', type=int, default=10, 
                    help='update number of actor-critic (default: 10)')
parser.add_argument('--total_sample_size', type=int, default=2048, 
                    help='total sample size to collect before PPO update (default: 2048)')
parser.add_argument('--batch_size', type=int, default=64, 
                    help='batch size to update (default: 64)')
parser.add_argument('--suspend_accu_exp', type=float, default=0.8,
                    help='accuracy for suspending discriminator about expert data (default: 0.8)')
parser.add_argument('--suspend_accu_gen', type=float, default=0.8,
                    help='accuracy for suspending discriminator about generated data (default: 0.8)')
parser.add_argument('--max_iter_num', type=int, default=4000,
                    help='maximal number of main iterations (default: 4000)')
parser.add_argument('--seed', type=int, default=500,
                    help='random seed (default: 500)')
parser.add_argument('--logdir', type=str, default='logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()
device = torch.device('cpu')
if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

def main():
    env = gym.make(args.env_name)
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    running_state = ZFilter((num_inputs,), clip=5)

    print('state size:', num_inputs) 
    print('action size:', num_actions)
    # PPO agent
    has_continuous_action_space = True
    action_std = 0.6
    action_std_decay_rate = 0.05
    min_action_std = 0.1
    action_std_decay_freq = int(2.5e5)
    ################ PPO hyperparameters ################


    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network
    if args.load_model is not None:
        action_std=0.1
    ppo_agent = PPO(num_inputs,num_actions, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std)

    discrim = Discriminator(num_inputs + num_actions, args)
    discrim = discrim.to(device)

    discrim_optim = optim.Adam(discrim.parameters(), lr=args.learning_rate)
    
    # load demonstrations
    expert_demo, _ = pickle.load(open('./expert_demo/expert_demo.p', "rb"))
    demonstrations = np.array(expert_demo)
    print("demonstrations.shape", demonstrations.shape)
    
    writer = SummaryWriter(args.logdir)

    if args.load_model is not None:
        action_std=0.1
        saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
        ckpt = torch.load(saved_ckpt_path)
        discrim.load_state_dict(ckpt['discrim'])
        running_state.rs.n = ckpt['z_filter_n']
        running_state.rs.mean = ckpt['z_filter_m']
        running_state.rs.sum_square = ckpt['z_filter_s']
        ppo_agent.load('./save_model/agent_ckpt_1091_3814.pth.tar')
        print("Loaded OK ex. Zfilter N {}".format(running_state.rs.n))

    
    episodes = 0
    train_discrim_flag = False

    while True:
        state = env.reset()
        score = 0
        done = False
        state = running_state(state)
            
        while not done:
            if args.render:
                env.render()

            action = ppo_agent.select_action(state)

            next_state, reward, done, _ = env.step(action)
            irl_reward = get_reward(discrim, state, action)


            ppo_agent.buffer.rewards.append(irl_reward)
            ppo_agent.buffer.is_terminals.append(done)

            next_state = running_state(next_state)
            state = next_state

            score += reward




if __name__=="__main__":
    main()