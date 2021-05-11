import CBEngine
import gym
import agent.gym_cfg as gym_cfg
simulator_cfg_file = './cfg/simulator.cfg'
mx_step = 360
gym_cfg_instance = gym_cfg.gym_cfg()

#gym
env = gym.make(
    'CBEngine-v0',
    simulator_cfg_file=simulator_cfg_file,
    thread_num=1,
    gym_dict=gym_cfg_instance.cfg,
    metric_period = 3600
)

for i in range(mx_step):
    print("{}/{}".format(i,mx_step))
    obs, rwd, dones, info = env.step({})
    for k,v in obs.items():
        print("{}:{}".format(k,v))
    print(info)
    for k,v in info.items():
        print("{}:{}".format(k,v))
