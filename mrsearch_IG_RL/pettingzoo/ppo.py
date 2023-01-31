import numpy as np
import torch

import util

## Device 
if torch.cuda.is_available():
    device = torch.device("cuda")
#elif torch.backends.mps.is_available(): #MPS is buggy
#    device = torch.device("mps")
else:
    device = torch.device("cpu")

def collect_rollouts(env, policy, n_steps, rollout, device=device):
    next_obs = env.reset()
    total_episodic_return = 0
    for step in range(0, n_steps):
        obs = util.dict2torch(next_obs, device)
        action_dict = {}
        for i in range(len(obs)):
            actions, logprobs, _, values = policy.get_action_and_value(obs[i])
            action_dict[env.possible_agents[i]] = actions

            rollout["obs_img"][step, i] = obs[i][0]
            rollout["obs_vec"][step, i] = obs[i][1]
            rollout["logprobs"][step, i] = logprobs
            rollout["values"][step, i] = values.flatten()

        next_obs, rewards, terms, truncs, infos = env.step(action_dict)

        for i in range(len(obs)):
            if step > 0:
                rewards[env.possible_agents[i]] += policy.get_intrinsic_reward(
                    (rollout["obs_img"][step-1, i],rollout["obs_vec"][step-1, i]),
                    (rollout["obs_img"][step, i],rollout["obs_vec"][step, i]),
                    action_dict[env.possible_agents[i]])
            rollout["rewards"][step, i] = torch.tensor(rewards[env.possible_agents[i]])
            rollout["terms"][step, i] = terms[env.possible_agents[i]]
            rollout["actions"][step, i] = action_dict[env.possible_agents[i]]

        total_episodic_return += rollout["rewards"][step].cpu().numpy()

        if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
            end_step = step
            break
    else:
        end_step = step
    return rollout, total_episodic_return, end_step

def bootstrap_value(rollout, end_step, gamma=0.99, device=device):
    rollout["advantages"] = torch.zeros_like(rollout["rewards"]).to(device)
    for t in reversed(range(end_step)):
        delta = (
            rollout["rewards"][t]
            + gamma * rollout["values"][t + 1] * rollout["terms"][t + 1]
            - rollout["values"][t]
            )
        rollout["advantages"][t] = delta + gamma * gamma * rollout["advantages"][t + 1]
        rollout["returns"] = rollout["advantages"] + rollout["values"]
    return rollout

def batchify(end_step, rollout):
    batched_rollout = {
        k : torch.flatten(v[:end_step], start_dim=0, end_dim=1) 
        for k,v in rollout.items()
        }
    return batched_rollout

def train(env, policy, optimizer, batch_size, epochs, end_step, rollout, loss_coef=0.1, ent_coef=0.1, vf_coef=0.1, clip_coef=0.2):
    updates = 0
    batched_rollout = batchify(end_step, rollout)
    
    b_index = np.arange(1,len(batched_rollout["obs_img"])) #start at 1 so last_state indexing doesnt throw err
    clip_fracs = []
    for epoch in range(epochs):
        print("Epoch {}:".format(epoch))
        np.random.shuffle(b_index)
        for start in range(0, len(batched_rollout["obs_img"]), batch_size):
            end = start + batch_size
            batch_index = b_index[start:end]

            _, newlogprob, entropy, value = policy.get_action_and_value(
                    (batched_rollout["obs_img"][batch_index],batched_rollout["obs_vec"][batch_index]), batched_rollout["actions"].long()[batch_index])
            logratio = newlogprob - batched_rollout["logprobs"][batch_index]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clip_fracs += [ ((ratio - 1.0).abs() > clip_coef).float().mean().item() ]

            # normalize advantaegs
            advantages = batched_rollout["advantages"][batch_index]
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

            # Policy loss
            pg_loss1 = -batched_rollout["advantages"][batch_index] * ratio
            pg_loss2 = -batched_rollout["advantages"][batch_index] * torch.clamp(
                ratio, 1 - clip_coef, 1 + clip_coef
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            value = value.flatten()
            v_loss_unclipped = (value - batched_rollout["returns"][batch_index]) ** 2
            v_clipped = batched_rollout["values"][batch_index] + torch.clamp(
                value - batched_rollout["values"][batch_index],
                -clip_coef,
                clip_coef,
            )
            v_loss_clipped = (v_clipped - batched_rollout["returns"][batch_index]) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()

            entropy_loss = entropy.mean()

            icm_loss = policy.get_icm_loss(
                (batched_rollout["obs_img"][batch_index-1],batched_rollout["obs_vec"][batch_index-1]),
                (batched_rollout["obs_img"][batch_index],batched_rollout["obs_vec"][batch_index]),
                batched_rollout["actions"].long()[batch_index])

            policy_loss = loss_coef * (pg_loss - ent_coef * entropy_loss + v_loss * vf_coef)
            loss = policy_loss + torch.sum(icm_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            updates += batch_size

        print("Policy Loss: {}".format(policy_loss.item()))
        print("ICM Loss: {}".format(torch.sum(icm_loss).item()))
        print("Training Loss: {}".format(loss.item()))
        print("Clip Fraction: {}".format(np.mean(clip_fracs)))
    return updates


def PPO(env, policy, optimizer, max_steps, epochs, batch_size, frame_size, stack_size, num_agents=2, n_steps=10_000,
    vec_obs_size=3, loss_coef=0.1, ent_coef=0.1, vf_coef=0.1, clip_coef=0.2, gamma=0.99, device=device):
    total_steps = 0
    while total_steps < max_steps:
    
        # storage
        rollout = {}
        rollout["obs_img"] = torch.zeros((n_steps, num_agents, stack_size, *frame_size)).to(device)
        rollout["obs_vec"]= torch.zeros((n_steps, num_agents, vec_obs_size)).to(device)
        rollout["actions"] = torch.zeros((n_steps, num_agents)).to(device)
        rollout["logprobs"] = torch.zeros((n_steps, num_agents)).to(device)
        rollout["rewards"] = torch.zeros((n_steps, num_agents)).to(device)
        rollout["terms"] = torch.zeros((n_steps, num_agents)).to(device)
        rollout["values"] = torch.zeros((n_steps, num_agents)).to(device)

        print("\n-------------------------------------------\n")
        print("Collectiing Rollout:")
        with torch.no_grad():
            rollout, total_episodic_return, end_step = collect_rollouts(env, policy, n_steps, rollout, device)
            rollout = bootstrap_value(rollout, end_step, gamma, device)
        
        if end_step < batch_size:
            print("Skipping Early Termination...\n")
            continue
        print("Episode Length: {}\n".format(end_step+1))
        
        print("Training for {} Epochs".format(epochs))
        total_steps += train(env, policy, optimizer, batch_size, epochs, end_step, rollout, loss_coef, ent_coef, vf_coef, clip_coef)
        torch.save(policy.state_dict(), "./log/model_{}.pth".format(env.date))
        print("\n...{}/{}".format(total_steps,max_steps))


if __name__ == "__main__":
    from env import icm_env
    from policy import Network

    ## Device 
    if torch.cuda.is_available():
        device = torch.device("cuda")
    #elif torch.backends.mps.is_available(): #MPS is buggy
    #    device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ## PPO Params
    #total_steps = 2_000_000
    batch_size = 20
    stack_size = 5
    frame_size = (96, 96)
    n_steps = 10_000 # steps per episode [max 10 mins]
    cfg_file = "cfg.yaml"
    env = icm_env(headless=True,record=False,cfg=cfg_file,output_dir="train_1_30_23")

    net = Network().to(device)

    max_steps = 10_000
    epochs = 5
    num_agents = 2

    opt = torch.optim.Adam(net.parameters(), lr=0.001, eps=1e-5)

    PPO(env, net, opt, max_steps, epochs, n_steps, num_agents, frame_size, batch_size, stack_size, loss_coef=0.1, device=device)












