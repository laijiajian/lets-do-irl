import torch
import numpy as np
from utils.utils import get_entropy, log_prob_density
device = torch.device('cpu')
if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")
def train_discrim(discrim, ppo_agent, discrim_optim, demonstrations, args):
    states = torch.squeeze(torch.stack(ppo_agent.buffer.states, dim=0)).detach().to(device)
    actions = torch.squeeze(torch.stack(ppo_agent.buffer.actions, dim=0)).detach().to(device)
        
    criterion = torch.nn.BCELoss()
    demonstrations = torch.Tensor(demonstrations).to(device)
    for _ in range(args.discrim_update_num):
        learner = discrim(torch.cat([states, actions], dim=1))
        expert = discrim(demonstrations)
        # BCEloss = ( xi,yi) first is predict, second is target
        discrim_loss = criterion(learner, torch.ones((states.shape[0], 1)).to(device)) + \
                        criterion(expert, torch.zeros((demonstrations.shape[0], 1)).to(device))
        # in this setting , the discrim for learner is converging to 1 and expert to zeros ?
        discrim_optim.zero_grad()
        discrim_loss.backward()
        discrim_optim.step()
    # expert_acc -> reward < 0.5?
    expert_acc = ((discrim(demonstrations) < 0.5).float()).mean()
    learner_acc = ((discrim(torch.cat([states, actions], dim=1)) > 0.5).float()).mean()
    # learner_acc -> reward > 0.5 ?
    return expert_acc, learner_acc


def train_actor_critic(actor, critic, memory, actor_optim, critic_optim, args):
    memory = np.array(memory) 
    states = np.vstack(memory[:, 0]) 
    actions = list(memory[:, 1]) 
    rewards = list(memory[:, 2]) 
    masks = list(memory[:, 3]) 

    old_values = critic(torch.Tensor(states))
    returns, advants = get_gae(rewards, masks, old_values, args)
    
    mu, std = actor(torch.Tensor(states))
    old_policy = log_prob_density(torch.Tensor(actions), mu, std)

    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)

    for _ in range(args.actor_critic_update_num):
        np.random.shuffle(arr)

        for i in range(n // args.batch_size): 
            batch_index = arr[args.batch_size * i : args.batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index)
            
            inputs = torch.Tensor(states)[batch_index]
            actions_samples = torch.Tensor(actions)[batch_index]
            returns_samples = returns.unsqueeze(1)[batch_index]
            advants_samples = advants.unsqueeze(1)[batch_index]
            oldvalue_samples = old_values[batch_index].detach()
            
            values = critic(inputs)
            clipped_values = oldvalue_samples + \
                             torch.clamp(values - oldvalue_samples,
                                         -args.clip_param, 
                                         args.clip_param)
            critic_loss1 = criterion(clipped_values, returns_samples)
            critic_loss2 = criterion(values, returns_samples)
            critic_loss = torch.max(critic_loss1, critic_loss2).mean()

            loss, ratio, entropy = surrogate_loss(actor, advants_samples, inputs,
                                         old_policy.detach(), actions_samples,
                                         batch_index)
            clipped_ratio = torch.clamp(ratio,
                                        1.0 - args.clip_param,
                                        1.0 + args.clip_param)
            clipped_loss = clipped_ratio * advants_samples
            actor_loss = -torch.min(loss, clipped_loss).mean()

            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

            critic_optim.zero_grad()
            loss.backward(retain_graph=True) 
            critic_optim.step()

            actor_optim.zero_grad()
            loss.backward()
            actor_optim.step()

def get_gae(rewards, masks, values, args):
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)
    
    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + (args.gamma * running_returns * masks[t])
        returns[t] = running_returns

        running_delta = rewards[t] + (args.gamma * previous_value * masks[t]) - \
                                        values.data[t]
        previous_value = values.data[t]
        
        running_advants = running_delta + (args.gamma * args.lamda * \
                                            running_advants * masks[t])
        advants[t] = running_advants

    advants = (advants - advants.mean()) / advants.std()
    return returns, advants

def surrogate_loss(actor, advants, states, old_policy, actions, batch_index):
    mu, std = actor(states)
    new_policy = log_prob_density(actions, mu, std)
    old_policy = old_policy[batch_index]

    ratio = torch.exp(new_policy - old_policy)
    surrogate_loss = ratio * advants
    entropy = get_entropy(mu, std)

    return surrogate_loss, ratio, entropy