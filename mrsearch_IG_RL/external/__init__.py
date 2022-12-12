## built-ins
from copy import deepcopy
import json
import math
import multiprocessing as mp
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

## numpy
import numpy as np

## scipy
from scipy.sparse.csgraph import shortest_path as shortest_path

## matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

## datetime
import datetime

## pyyaml
import yaml

## argparse
import argparse

## pybullet
import pybullet as p
import pybullet_data
from pybullet import getEulerFromQuaternion as Q2E
from pybullet import getQuaternionFromEuler as E2Q
import pybullet_utils.bullet_client as bc

## torch
import torch
import torch.nn as nn

## torchviz
from torchviz import make_dot

## gym
import gym
from gym import Env, spaces

## stable_baselines3
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import obs_as_tensor, explained_variance
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn