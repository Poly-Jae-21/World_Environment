from gymnasium import Env, spaces
import numpy as np
from pyogrio import read_dataframe
from typing_extensions import Optional
import gemgis as gg
import random
import pygame
from os import path
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from template.env_name.envs.utils.action import Action
from template.env_name.envs.utils.data_conversion import Polygon_to_matrix, Density

from gymnasium.utils import EzPickle
WINDOW_SIZE = [3420, 4207]

def generate_partially_observable_env(current_position, main_map):
    partial_observable_map = np.zeros([9, 100, 100])
    for i in range(0, 100):
        for j in range(0, 100):
            x, y = int(current_position[0]), int(current_position[1])
            if x + i - 50 <= 0 or x + i - 50>= WINDOW_SIZE[1] or y + j <= 0 or y + j - 50 >= WINDOW_SIZE[0]:
                continue
            else:
                partial_observable_map[:,i,j] = main_map[:,x+i-50,y+j-50]
    return partial_observable_map

class UrbanPlanning(Env, EzPickle):
    metadata = {
        'render.modes': ['human'],
        "render_fps": 4,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.time_step = 0

        low = np.array(
            [0,
             0,
             0,
             0,
             0,
             0,
             0,
             0,
             0,
             0,

            ]
        ).astype(np.float32)

        high = np.array(
            [1,
             1,
             1,
             1,
             1,
             1,
             1,
             1,
             1,
             1,
             ]
        ).astype(np.float32)

        self.observation_space = spaces.Box(low=low, high=high)
        self.action_space = spaces.Box(-1, +1, (3,), dtype=np.float32)

        self.main