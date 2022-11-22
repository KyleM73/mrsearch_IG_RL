if __name__=="__main__":
    try:
        import gym
        import numpy
        import pybullet
        import pybullet_data
        import pybullet_utils
        import time
        import datetime
        import sys
        import os
        import stable_baselines3
        import torch
        print("SETUP PASSED")
    except:
        print("ERROR: SETUP FAILED")
