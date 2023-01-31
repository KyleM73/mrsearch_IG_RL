import torch

import env
import policy
import util

## Create Env
cfg_file = "cfg.yaml"
output_dir = "test_1_30_23"
env = env.icm_env(headless=True,record=True,cfg=cfg_file,output_dir=output_dir)

model = "./model.pth"
net = policy.Network()
net.load_state_dict(torch.load(model))
net.eval()

print("simulating...")
obs = env.reset()
action_dict = {}
for _ in range(env.max_steps):
    obs = util.dict2torch(obs, "cpu")
    for i in range(len(obs)):
        actions, _, _, _ = net.get_action_and_value(obs[i])
        action_dict[env.possible_agents[i]] = actions
    obs, _, terms, truncs, _ = env.step(action_dict)
    if any([terms[a] for a in terms]):
        print("terminated.")
        break
    if any([truncs[a] for a in truncs]):
        print("truncated.")
        break
print("done.")
