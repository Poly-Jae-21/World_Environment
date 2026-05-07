from gymnasium import Env, spaces
import numpy as np
from pyogrio import read_dataframe
from typing_extensions import Optional
import gemgis as gg
import random
import pygame
import math
from os import path
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import copy
from typing import Dict, Any, List

from template.env_name.envs.utils.action import Action
from template.env_name.envs.utils.data_conversion import Polygon_to_matrix, Density

#WINDOW_SIZE = [3420, 4207]
WINDOW_SIZE = [3426, 4211]

def generate_partial_observation(agent_position, MAP):
    p_observation_map = np.zeros([15, 100, 100])

    for i in range(0, 100):

        for j in range(0, 100):

            x_val, y_val = int(agent_position[0]), int(agent_position[1])

            if x_val + i - 50 <= 0 or x_val + i + 50 >= WINDOW_SIZE[1] or y_val + j - 50 <= 0 or y_val + j + 50 >= WINDOW_SIZE[0]:
                continue
            else:
                p_observation_map[:, i, j] = MAP[:, x_val + i - 50, y_val + j - 50]

    return p_observation_map

"""
def generate_partial_observation(agent_position, MAP):
    p_observation_map = np.zeros([9, 100, 100])
    for i in range(0, 100):
        for j in range(0, 100):
            x_val, y_val = int(agent_position[0]), int(agent_position[1])
            if x_val + i - 50 <= 0 or x_val + i + 50 >= WINDOW_SIZE[1] or y_val + j - 50 <= 0 or y_val + j + 50 >= WINDOW_SIZE[0]:
                continue
            else:
                p_observation_map[:, i, j] = MAP[:, x_val + i - 50, y_val + j - 50]
    return p_observation_map
"""

import pickle

def load_map(file_path, file_name):

    full_path = path.join(file_path, file_name)
    with open(full_path, 'rb') as f:
        data = pickle.load(f)
    return data


class ChicagoMultiPolicyMapv2(Env):
    """
    The charging network planning involves investigating optimal distribution urban charging network planning in a Chicago environment area (grid world),
    solving sequential multiple criteria decision-making problem and distributing charging stations.

    ## Description
    There are 4 layers designated 1) , 2) land information, 3) electricity demand, and 4) charging demand in the () grid world.

    The starting point for investigation is a crucial to distribute charging networks, and so we use two steps of the starting point for investigation.
    1) Select a community area between -- communities depending on the distribution probability. The Distribution probability map is based on the simple random over 1 ~ 30 episodes.
    Over 30 episodes, the distribution probability map is changed by spatial auto-correlation of charging stations in communities.
    2) After selecting a community, starting point is randomly selected within the community, where the land-use is satisfied.

    There are three policies for three criteria to which main-objective should be all satisfied. However, the calculation of three criteria at once makes difficult convergent to optimization in non-convex problem.
    Thus, we separate single policy problem into three multi-policies problems using meta-learning approach. In the step function involves four conditions to calculate rewards.

    ## Action space
    The action space is (3) in the range 0 to 1 indicating which direction to investigate the optimal sites and to determine charging capacity in those charging stations.

    env.action_space.shape[0] = 3
    - 0: x position of the charging station
    - 1: y position of the charging station
    - 2: capacity of the charging station

    ## Observation Space
    The observation space is extracted to the environment MAP to the center of the charging station. The size of this is 100 x 100 x 4 numpy array
    """

    metadata = {
        "render_modes": ["human"],
        "render_fps": 4,
    }

    def __init__(self, render_mode: Optional[str] = None):

        self.time_step = 1

        self.boundary_x, self.boundary_y = WINDOW_SIZE[0], WINDOW_SIZE[1]
        self.min_x, self.min_y = 42213, 461044

        low = np.array(
            [0, # x
             0, # y
             0, # boundary map
             0, # landuse map
             0, # Alpha = Percentage of electricity by capacity
             0, # Minimum distance of power grid to target site (current position)
             0, # Minimum distance of main road to target site
             0, # Average percentage of vegetation in the observation
             0, # Total VMT values in the observation
             0, # Total potential electricity values in the observation
             0, # Average distance of other potential sites in the observation
            ]
        ).astype(np.float32)

        high = np.array(
            [1,
             1,
             77,
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

        self.next_state = [0,0,0,0,0,0,0,0,0,0,0]

        self.observation_space = spaces.Box(low, high)
        self.action_space = spaces.Box(-1, +1, (3,), dtype=np.float32)

        self.scalar_VMT_ = MinMaxScaler(feature_range=(0, 1))
        self.scalar_PE_ = MinMaxScaler(feature_range=(0, 1))

        self.main_MAP_ = self.load_map()


        self.episode = 0

        self.initial_position = 0
        self.position_record = []
        self.capacity_record = [0]

        self.probability_list = []  ## This is for the starting point distribution probability. It can be updated over episodes after 32 self.episode
        self.max_steps = 200
        self.factor = None
        self.service_radius_list = []

        #self.select_community = random.randint(1, 77)

        self.converted_action = 0

        # pygame utils
        self.window = None
        self.clock = None
        self.cell_size = 1
        self.evcs_imgs = None

        self.render_mode = render_mode

        self.total_reward = 0

        self.system_total_env_reward = 0
        self.system_total_eco_reward = 0
        self.system_total_urb_reward = 0

        self.income_q33 = 0
        self.income_q66 = 0

        self.weights = None

        self.reward_history: List[Dict[str, Any]] = []

        self.temp_reward_log = {
        "env": {
            "sub": {"r_avm": 0.0, "r_viss": 0.0, "r_TER": 0.0, 'R_e': 0.0}
        },
        "eco": {
            "sub": {"R_eck": 0.0, 'R_ec': 0.0}
        },
        "urb": {
            "sub": {"r_drn": 0.0, "r_dg": 0.0, "r_lu": 0.0, "r_sc": 0.0, 'R_ue': 0.0}
        },
        "overall": {"total": 0.0, "average": 0.0}}

        self.return_reward_log: Dict[str, Any] = {}

        self.ewci_result = []

    def load_map(self):
        map = load_map(file_path="C:/Users/S2HubLab/PycharmProjects/World_Environment/template/env_name/envs/data",file_name="main_MAP.pkl")

        VMT_layer = map[5]
        VMT_mask = VMT_layer != 0
        VMT_values = VMT_layer[VMT_mask].reshape(-1,1)

        self.scalar_VMT_.fit(VMT_values)
        VMT_scaled = self.scalar_VMT_.transform(VMT_values)
        VMT_layer[VMT_mask] = VMT_scaled.ravel()

        map[5] = VMT_layer

        PE_layer = map[6]
        PE_mask = PE_layer != 0
        PE_values = PE_layer[PE_mask].reshape(-1, 1)

        self.scalar_PE_.fit(PE_values)
        PE_scaled = self.scalar_PE_.transform(PE_values)
        PE_layer[PE_mask] = PE_scaled.ravel()

        map[6] = PE_layer

        return map

    def conversion_into_extent(self, action_record):
        """
        Converting grid coordinate into x,y extent to visualize the output on ArcGIS Pro
        """
        capacity_value = action_record[..., 2]
        capacity = np.reshape(capacity_value, (len(capacity_value), 1))

        x_extent = 10 * (action_record[..., 1]  + self.min_x) # self.min_x = 42213
        x_extent = np.reshape(x_extent, (len(x_extent), 1))

        y_extent = 10 * (self.boundary_y - action_record[..., 0] + self.min_y) # self.min_y = 461044
        y_extent = np.reshape(y_extent, (len(y_extent), 1))

        output_action = np.hstack((x_extent, y_extent, capacity))
        return output_action
    def reset(self,
              seed: Optional[int] = None,
              options: Optional[list] = None, ):

        """
        This is to create environment or set up an initial position and initial partial observation
        """

        self.time_step = 1
        self.total_reward = 0
        self.system_total_env_reward, self.system_total_eco_reward, self.system_total_urb_reward = 0, 0, 0
        self.initial_position = np.array([0, 0])
        self.episode = options[0] + 1
        self.weights = options[1]

        self.temp_reward_log = {
        "env": {
            "sub": {"r_avm": 0.0, "r_viss": 0.0, "r_TER": 0.0, 'R_e': 0.0}
        },
        "eco": {
            "sub": {"R_eck": 0.0, 'R_ec': 0.0}
        },
        "urb": {
            "sub": {"r_drn": 0.0, "r_dg": 0.0, "r_lu": 0.0, "r_sc": 0.0, 'R_ue': 0.0}
        },
        "overall": {"total": 0.0, "average": 0.0}}
        self.reward_history = []
        # self.main_MAP_ = self.main_MAP.copy()

        if not hasattr(self, 'action_converter'):
            self.action_converter = Action(self.boundary_x, self.boundary_y)
        ## Simplify the training model
        if self.episode <= 10000:
            self.select_community = random.randint(1, 77)
            initial_position_list = np.argwhere(self.main_MAP_[0] == self.select_community)
            if initial_position_list.size > 0:
                selected_initial_starting_point = random.choice(initial_position_list)
                selected_initial_starting_point = np.array(selected_initial_starting_point)
                self.initial_position = selected_initial_starting_point
                self.temp_action_record = np.hstack(([self.initial_position, np.array([12000])]))[np.newaxis, :]

                start_action = (self.temp_action_record.squeeze(), "reset", False, True, False)

                info = {"community": self.select_community, "initial_position": self.initial_position.tolist()}

                if self.episode == 1:
                    self.action_record_environment = np.hstack(
                        ([self.initial_position, np.array([12000]), np.array([0]), np.array([0]), np.array([self.episode])])).reshape(1, -1)
                    self.action_record_economy = np.hstack(
                        ([self.initial_position, np.array([12000]), np.array([0]), np.array([0]), np.array([self.episode])])).reshape(1, -1)
                    self.action_record_urbanity = np.hstack(
                        ([self.initial_position, np.array([12000]), np.array([0]), np.array([0]), np.array([self.episode])])).reshape(1, -1)

                    self.action_record_overall = np.hstack(([self.initial_position, np.array([12000]), np.array([0]),
                                                             np.array([0]), np.array([0]), np.array([0]), np.array([0]),
                                                             np.array([0]), np.array([self.episode])])).reshape(1, -1)
                    self.action_record_system = np.hstack(([self.initial_position, np.array([12000]), np.array([0]),
                                                            np.array([0]), np.array([0]), np.array([0]), np.array([0]),
                                                            np.array([0]), np.array([self.episode])])).reshape(1, -1)

                    self.average_action_record = np.hstack(([self.initial_position, np.array([12000]), np.array([0]),
                                                             np.array([0]), np.array([0]), np.array([0]), np.array([0]),
                                                             np.array([0]), np.array([self.episode])])).reshape(1, -1)
                    self.meta_action_record = np.hstack(([self.initial_position, np.array([12000]), np.array([0]),
                                                          np.array([0]), np.array([0]), np.array([0]), np.array([0]),
                                                          np.array([0]), np.array([self.episode])])).reshape(1, -1)

                return self.step(start_action)[0], info
            else:
                print("No positions with the value 3 found")
                info = {"error": f"No valid positions in community {self.select_community}"}
                return None, info

        else:
            Density_weight = self.Den.KernelDensity(self.radius, self.main_MAP_)
            updated_weight_list = 1 / (np.exp(Density_weight) + 77)  ## 77 = The number of community areas in Chicago
            self.probability_list = updated_weight_list / np.sum(updated_weight_list)
            selected_community = \
            random.choices(population=[i + 1 for i in range(77)], weights=self.probability_list, k=1)[0]
            high_positions_list = np.argwhere(self.main_MAP_[0] == selected_community)
            selected_high_position = np.array(random.choice(high_positions_list))

            while selected_high_position.size > 0:
                if selected_high_position.size > 0:

                    self.initial_position = selected_high_position
                    self.temp_action_record = np.hstack((self.initial_position, np.array([12000])))[np.newaxis, :]
                    info = {"community": selected_community, "initial_position": self.initial_position.tolist()}

                    start_action = (self.temp_action_record.squeeze(), "reset", False, True, False)

                    return self.step(start_action)[0], info
                else:
                    print("Non valid positions")
                    selected_community = \
                    random.choices(population=[i + 1 for i in range(77)], weights=self.probability_list, k=1)[0]
                    high_positions_list = np.argwhere(self.main_MAP_[0] == selected_community)
                    selected_high_position = np.array(random.choice(high_positions_list))

    def update_reward_log_per_timestep(self, env_sub_rewards: dict,
                                       eco_sub_rewards: dict,
                                       urb_sub_rewards: dict,
                                       step_reward: float = None):

        def update_objective(obj_key, sub_rewards):
            for k, v in sub_rewards.items():
                self.temp_reward_log[obj_key]["sub"][k] = float(v)

        update_objective("env", env_sub_rewards)
        update_objective("eco", eco_sub_rewards)
        update_objective("urb", urb_sub_rewards)

        if step_reward is None:
            step_reward = (
                self.temp_reward_log['env']['sub']['R_e']
                + self.temp_reward_log['eco']['sub']['R_ec']
                + self.temp_reward_log['urb']['sub']['R_ue']
            )
        self.temp_reward_log['overall']['total'] = float(step_reward)
        self.temp_reward_log['overall']['average'] = float(step_reward)

    def record_step_reward_log(self):
        self.reward_history.append(copy.deepcopy(self.temp_reward_log))

    def build_return_reward_log_from_history(self):
        logs = self.reward_history
        n = len(logs)

        template = copy.deepcopy(logs[0])

        def zero_numeric_leaves(d: Dict[str, Any]):
            for k, v in d.items():
                if isinstance(v, dict):
                    zero_numeric_leaves(v)
                else:
                    d[k] = 0.0

        sum_log = copy.deepcopy(template)
        zero_numeric_leaves(sum_log)

        def add_numeric_leaver(dst: Dict[str, Any], src: Dict[str, Any]):
            for k, v in src.items():
                if isinstance(v, dict):
                    add_numeric_leaver(dst[k], v)
                else:
                    dst[k] += float(v)

        for log in logs:
            add_numeric_leaver(sum_log, log)

        return_log = copy.deepcopy(template)

        def div_numeric_leaves(d: Dict[str, Any], denom: float):
            for k, v in d.items():
                if isinstance(v, dict):
                    div_numeric_leaves(v, denom)
                else:
                    d[k] = float(v) / denom

        div_numeric_leaves(return_log, float(n))

        # Keep episode-level overall aligned with step reward r.
        episode_total = float(sum_log["overall"].get("total", 0.0))
        return_log["overall"]['total'] = episode_total
        return_log['overall']['average'] = episode_total / float(n)

        return return_log

    def step(self, action_with_factor):

        """
        :param action_with_factor: action ((x,y),capacity,update[True, False], reset[True, False])
        Update boolean is to identify whether environment has to update in current step.
        Reset boolean refers that current step is from the reset function.
        :return:
        """

        self.temp_reward_log = {
        "env": {
            "sub": {"r_avm": 0.0, "r_viss": 0.0, "r_TER": 0.0, 'R_e': 0.0}
        },
        "eco": {
            "sub": {"R_eck": 0.0, 'R_ec': 0.0}
        },
        "urb": {
            "sub": {"r_drn": 0.0, "r_dg": 0.0, "r_lu": 0.0, "r_sc": 0.0, 'R_ue': 0.0}
        },
        "overall": {"total": 0.0, "average": 0.0}}

        current_position = self.temp_action_record[-1][0:2]

        action = action_with_factor[0]

        if self.time_step == 1:
            done = False

        self.factor = action_with_factor[1]

        env_update = action_with_factor[2]

        reset_step = action_with_factor[3]

        save_result = action_with_factor[4]

        ### self.time_step != 0 and reset_step == False -> action is from output in the training model, and thus it should be needed to convert into the extent. On the other hand, action does not need to convert into real extent.
        if not reset_step:
            converted_action = self.action_converter.local_action_converter(current_position,action) # next position
        else:
            converted_action = action

        action_group = converted_action[0:2].astype(int)
        x, y, capacity = converted_action[0].astype(int), converted_action[1].astype(int), converted_action[2].astype(int)

        self.converted_action_a = self.conversion_into_extent(np.array([x, y, capacity]).reshape(1, -1)) # [Rows, Columns, Capacity]

        next_observation = generate_partial_observation(action_group, self.main_MAP_)

        observation_position = np.array((50, 50))

        VMT_indices, VMT_values, raw_VMT_values, VMT, raw_VMT = self._process_indices(next_observation, observation_position, -1)

        PE_indices, PE_values, raw_PE_values, PE, raw_PE = self._process_indices(next_observation, observation_position, -8)

        # potential EVCS
        indices_all_1 = np.argwhere(next_observation[13,:] != 0)
        av_of_ops = np.mean(np.linalg.norm(observation_position - indices_all_1, axis=1)) / 50 if len(indices_all_1) > 0 else 0

        # power grid
        indices_all_2 = np.argwhere(next_observation[4,:] != 0)
        mg = 1 - (np.min(np.linalg.norm(observation_position - indices_all_2, axis=1)) / (50*math.sqrt(2))) if len(indices_all_2) > 0 else 0

        # main road
        indices_all_3 = np.argwhere(next_observation[3,:] != 0)
        mr = 1 - (np.min(np.linalg.norm(observation_position - indices_all_3, axis=1)) / (50*math.sqrt(2))) if len(indices_all_3) > 0 else 0

        Alpha = 1 if raw_PE >= capacity else raw_PE / capacity

        avm = np.mean(next_observation[2][next_observation[2] != 0]) / 100 if len(next_observation[2][next_observation[2] != 0]) != 0 else 0
        VMT = 0 if len(VMT_indices) == 0 else VMT/len(VMT_indices)
        PE_ = 0 if len(PE_indices) == 0 else PE/len(PE_indices)


        if reset_step:
            self.next_state = [
                x / self.boundary_x,
                y / self.boundary_y,
                self.select_community / 77,
                self.main_MAP_[1, x, y],
                Alpha,
                mg,
                mr,
                avm,
                VMT,
                PE_,
                0
            ]

        if 0 <= x < self.main_MAP_.shape[1] and 0 <= y < self.main_MAP_.shape[2]:
            self.next_state = [
                x / self.boundary_x,
                y / self.boundary_y,
                self.select_community / 77,
                self.main_MAP_[1, x, y],
                Alpha,
                mg,
                mr,
                avm,
                VMT,
                PE_,
                av_of_ops
            ]
        else:
            self.next_state = self.next_state


        if len(VMT_indices) == 0 or int(self.main_MAP_[0,x,y]) is not self.select_community:
            return self._handle_invalid_action(next_observation, self.next_state, reset_step, action_group, capacity, x, y, VMT_indices, PE_indices, raw_VMT, save_result)

        else:
            r, info, _ = self._calculate_reward(self.factor, raw_VMT, raw_PE, Alpha, capacity, next_observation)
            done, terminate, return_log = self._update_environment(self.factor, action_group, VMT_indices, PE_indices, capacity, r, x, y, env_update, info, save_result)

            return np.array(self.next_state, dtype=np.float32), r, done, terminate, return_log

    def render(self):
        if self.render_mode == "human":
            return self._render_gui()

    def _render_gui(self):

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(WINDOW_SIZE)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.evcs_imgs is None:
            file_name = path.join(path.dirname(__file__), "img/EVCS.png")
            self.evcs_imgs = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)

        for y in range(self.main_MAP_.shape[1]):
            for x in range(self.main_MAP_.shape[2]):
                cell = (x * self.cell_size, y * self.cell_size)
                color = (255, 255, 255)
                if self.main_MAP_[1, y, x] == -1:
                    color = (255, 0, 0)  # Red
                elif self.main_MAP_[1, y, x] == -2:
                    color = (0, 0, 255)  # Blue
                elif self.main_MAP_[1, y, x] == - 8:
                    PE_value = int((self.main_MAP_[0, y, x] / 100) * 255)
                    color = (PE_value, 0, 0)
                elif self.main_MAP_[1, y, x] == -16:
                    self.window.blit(self.evcs_imgs, cell)
                pygame.draw.rect(self.window, color,
                                 pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size))
        pygame.display.flip()

    def _process_indices(self, observation, observation_position, target_value):
        layer = 5 if target_value == -1 else 6
        indices_all = np.argwhere(observation[layer] != 0)
        filtered_indices = indices_all[np.linalg.norm(observation_position - indices_all, axis=1) < (50*math.sqrt(2))]

        if target_value == -1:
            total_values = []
            Inverse_total_values = []
            for x, y in filtered_indices:
                obs_ = observation[5, x, y]
                total_values.append(obs_)
                obs_ = self.scalar_VMT_.inverse_transform(np.array(obs_).reshape(-1, 1)).item()
                Inverse_total_values.append(obs_)

            total_indices = np.unique(filtered_indices, axis=0)

            Inverse_value = 0.28 * np.sum(Inverse_total_values)
            value = 0.28 * np.sum(total_values) / len(total_values) if len(total_values) > 0 else 0

        else:
            total_values = []
            Inverse_total_values = []
            for x, y in filtered_indices:
                obs_ = observation[6, x, y]
                total_values.append(obs_)
                obs_ = self.scalar_PE_.inverse_transform(np.array(obs_).reshape(-1, 1)).item()
                Inverse_total_values.append(obs_)

            total_indices = np.unique(filtered_indices, axis=0)

            value = np.sum(total_values)
            Inverse_value = np.sum(Inverse_total_values) / len(total_values) if len(total_values) > 0 else 0

        return total_indices, np.array(total_values).reshape(-1, 1), np.array(Inverse_total_values).reshape(-1,
                                                                                                            1), value, Inverse_value

    def _handle_invalid_action(self, next_observation, next_state, reset_step, action_group, capacity, x, y, VMT_indices, PE_indices, raw_VMT, save_result):

        if self.time_step == 1:
            done = False

        if self.time_step == self.max_steps:
            done = True
        else:
            done = False
        terminate = (self.episode + 1 == 20000) if done else False

        r = -6

        self.info = {}

        env_sub = {"r_avm": -6/3, "r_viss": -6/3, "r_TER": -6/3, 'R_e': -6}

        eco_sub = {"R_eck": -6, 'R_ec': -6}

        urb_sub = {"r_drn": -6/4, "r_dg": -6/4, "r_lu": -6/4, "r_sc": -6/4, 'R_ue': -6}

        self.update_reward_log_per_timestep(env_sub, eco_sub, urb_sub, step_reward=r)

        self.record_step_reward_log()

        if done:
            converted_action = np.array([x, y, capacity]).reshape(1, -1)
            raw_converted_action = self.conversion_into_extent(converted_action)

            if self.factor == 'environment':
                self.total_reward -= 6
                self.info = {'reward_environment': -6, 'total_reward': self.total_reward,'average_reward': self.total_reward / self.time_step}
                self.reward_environment = np.array([-6, self.select_community]).reshape(1, -1)
                trajectory_environment = np.hstack((raw_converted_action, self.reward_environment, np.array([[self.episode]])))
                self.action_record_environment = np.append(self.action_record_environment, trajectory_environment, axis=0)
                if save_result:
                    np.savetxt("out/result/reward/test/" + "environment_action.csv", self.action_record_environment,delimiter=",", header="x,y,capacity,reward,community_label")

            elif self.factor == 'economic':
                self.total_reward -= 6
                self.info = {'reward_economy': -6, 'total_reward': self.total_reward,'average_reward': self.total_reward / self.time_step}
                self.reward_economy = np.array([-6, self.select_community]).reshape(1, -1)
                trajectory_economy = np.hstack((raw_converted_action, self.reward_economy, np.array([[self.episode]])))
                self.action_record_economy = np.append(self.action_record_economy, trajectory_economy, axis=0)
                if save_result:
                    np.savetxt("out/result/reward/test/" + "economy_action.csv", self.action_record_economy,delimiter=",", header="x,y,capacity,reward,community_label")

            elif self.factor == 'urbanity':
                self.total_reward -= 6
                self.info = {'reward_urbanity': -6, 'total_reward': self.total_reward,'average_reward': self.total_reward / self.time_step}
                self.reward_urbanity = np.array([-6, self.select_community]).reshape(1, -1)
                trajectory_urbanity = np.hstack((raw_converted_action, self.reward_urbanity, np.array([[self.episode]])))
                self.action_record_urbanity = np.append(self.action_record_urbanity, trajectory_urbanity, axis=0)
                if save_result:
                    np.savetxt("out/result/reward/test/" + "urbanity_action.csv", self.action_record_urbanity,delimiter=",", header="x,y,capacity,reward,community_label")

            elif self.factor == 'overall':
                self.total_reward -= 6
                self.info = {'environment reward': -6, 'economic reward': -6, 'urbanity reward': -6,'total reward': self.total_reward, 'average_reward': self.total_reward / self.time_step,'raw_VMT': raw_VMT}
                self.reward_overall = np.array([-6,-6,-6,self.total_reward,self.total_reward/self.time_step, self.select_community]).reshape(1, -1)
                trajectory_overall = np.hstack((raw_converted_action, self.reward_overall, np.array([[self.episode]])))
                self.action_record_overall = np.append(self.action_record_overall, trajectory_overall, axis=0)
                if save_result:
                    np.savetxt("out/result/reward/test/" + "overall_action.csv", self.action_record_overall,delimiter=",", header="x,y,capacity,en_reward, eco_reward, urb_reward, total_reward, avr_reward, community_label")

            elif self.factor == 'system':
                self.total_reward -= ((self.weights['w1']*(-6)+self.weights['w2']*(-6)+self.weights['w3']*(-6)) / (self.weights['w1']+self.weights['w2']+self.weights['w3']))
                self.system_total_env_reward -= 6
                self.system_total_eco_reward -= 6
                self.system_total_urb_reward -= 6
                self.info = {'environment_reward': self.system_total_env_reward/self.time_step, 'economic_reward': self.system_total_eco_reward/self.time_step, 'urbanity_reward': self.system_total_urb_reward/self.time_step, 'total_reward': self.total_reward,'average_reward': self.total_reward/self.time_step, 'raw_VMT': raw_VMT}
                self.reward_system = np.array([self.system_total_env_reward/self.time_step, self.system_total_eco_reward/self.time_step, self.system_total_urb_reward/self.time_step, self.total_reward,self.total_reward/self.time_step, self.select_community]).reshape(1, -1)
                trajectory_system = np.hstack((raw_converted_action, self.reward_system, np.array([[self.episode]])))
                self.action_record_system = np.append(self.action_record_system, trajectory_system, axis=0)
                if save_result:
                    np.savetxt("out/result/reward/test/" + "system_action.csv", self.action_record_system, delimiter=",", header="x,y,capacity,en_reward, eco_reward, urb_reward, total_reward, avr_reward, community_label")


                '''
                self.reward_average = (self.reward_environment + self.reward_economy + self.reward_urbanity) / 3
                self.reward_meta = np.array([-1])
                meta_converted_action = np.array([y, x, capacity,-1,-1,-1,-1,-1, self.select_community]).reshape(1, -1)
                average_action_record = np.array([np.mean(np.array(
                    [self.action_record_environment[-1], self.action_record_economy[-1],
                     self.action_record_urbanity[-1]]), axis=0)]).astype(int)
                average_reward = np.array([
                    self.reward_environment.item(),
                    self.reward_economy.item(),
                    self.reward_urbanity.item(),
                    self.reward_average.item(),
                    self.reward_meta.item(),
                    self.select_community
                ]).reshape(1, -1)
                average_value = np.hstack((average_action_record, average_reward))

                self.average_action_record = np.append(self.average_action_record, average_value, axis=0)
                self.meta_action_record = np.append(self.meta_action_record, meta_converted_action, axis=0)
                '''

            '''
            if int(next_observation[0,50,50]) == int(self.select_community):
                self._apply_map_updates(self.factor, action_group, VMT_indices, PE_indices, capacity, x, y)
            '''



            self.temp_action_record = np.hstack(([self.initial_position, np.array([12000])]))[np.newaxis, :]
            self.time_step = 1
            self.total_reward = 0
            self.system_total_env_reward, self.system_total_eco_reward, self.system_total_urb_reward = 0, 0, 0
            '''
            np.savetxt("out/result/reward/test/" + "average_action_record.csv", self.average_action_record, delimiter=",")
            np.savetxt("out/result/reward/test/" + "meta_action_record.csv", self.meta_action_record, delimiter=",")          
            '''

        elif not done:
            last_action = self.temp_action_record[-1]
            self.temp_action_record = np.append(self.temp_action_record, last_action.reshape(1, -1).astype(int),axis=0)
            self.time_step += 1

        else:
            self.temp_action_record = np.hstack(([self.initial_position, np.array([12000])]))[np.newaxis, :]
            self.time_step = 1
            self.total_reward = 0
            self.system_total_env_reward, self.system_total_eco_reward, self.system_total_urb_reward = 0, 0, 0

        if done:
            return_log = self.build_return_reward_log_from_history()

        else:
            return_log = self.temp_reward_log

        return np.array(next_state, dtype=np.float32), r, done, terminate, return_log

    def _calculate_reward(self, factor, raw_VMT, PE, Alpha, capacity, observation_map):
        if factor == 'environment':
            R_e, info, env_sub = self._calculate_environment_reward(raw_VMT, Alpha, observation_map)
            self.total_reward += R_e
            return R_e, info, env_sub
        elif factor == 'economic':
            R_ec, info, eco_sub = self._calculate_economic_reward(raw_VMT, Alpha, capacity)
            self.total_reward += R_ec
            return R_ec, info, eco_sub
        elif factor == 'urbanity':
            R_u, info, urb_sub = self._calculate_urbanity_reward(raw_VMT, Alpha, capacity, observation_map)
            self.total_reward += R_u
            return R_u, info, urb_sub
        else:
            return self._calculate_composite_reward(raw_VMT, Alpha, capacity, observation_map)

    def _calculate_environment_reward(self, raw_VMT, Alpha, observation_map):
        avm = np.mean(observation_map[2][observation_map[2] != 0]) / 100
        r_avm = np.exp(-avm)
        viss = observation_map[2, 50, 50] / 100
        r_viss = np.exp(-viss)

        r_apr = raw_VMT * 23.7 / 21.79 - raw_VMT * 0.72576 / 4.56
        r_eser = Alpha * raw_VMT * 0.72576 / 4.56
        r_TER = (r_apr + r_eser) * 0.0005
        r_TER = 1 - np.exp(-r_TER)

        R_a = (0.3 * r_avm + 0.3 * r_viss + 0.4 * r_TER) * 10

        bonus = (1 - self.time_step / self.max_steps) * R_a * 0.2

        R_a += bonus if R_a >= 6 else - bonus

        R_e = np.clip(R_a, -12, 12)

        info = {'reward_environment': R_e, 'total_reward': self.total_reward, 'average_reward': self.total_reward/self.time_step}

        env_sub = {
            "r_avm": r_avm,
            "r_viss": r_viss,
            "r_TER": r_TER,
            "R_e": R_e
        }

        return R_e, info, env_sub

    def _calculate_economic_reward(self, raw_VMT, Alpha, capacity):
        z = round(capacity / 6000) if capacity > 0 else 0
        F_z = z * (20600*0.8 + 800*12)
        P_G = (Alpha * (0.28 - 0.06) + (1 - Alpha) * (0.28 - 0.0405)) * raw_VMT * 365 / (4.56*5)
        ### commercial electricity rate (Chicago): 0.0405$/kwh, Supercharger cost (0.2~0.6): 0.28$/kwh, PV electricity rate(SolarReviews): 0.06$/kwh
        R_eck = P_G * 12 / F_z

        R_ecb = 10 * np.tanh((R_eck - 1) * 2)

        if R_ecb > 0:
            R_ecb += (1 - self.time_step / self.max_steps) * 2
        else:
            R_ecb -= 2

        R_ec = np.clip(R_ecb, -12, 12)

        info = {'reward_economy': R_ec, 'total_reward': self.total_reward, 'average_reward': self.total_reward/self.time_step}

        eco_sub = {
            "R_eck": R_eck,
            "R_ec": R_ec,
        }

        return R_ec, info, eco_sub

    def _calculate_urbanity_reward(self, raw_VMT, Alpha, capacity, observation_map):
        r_drn = 1 if len(observation_map[3][observation_map[3] != 0]) != 0 else 0
        r_dg = 1 if Alpha == 1 else (0.5 if len(observation_map[4][observation_map[4] != 0]) != 0 else 0)
        r_lu = 1 if observation_map[1, 50, 50] != 0 else 0
        r_sc = 1 if capacity >= (raw_VMT / 4.56) else 0
        R_ue =  (r_drn + r_dg + r_lu + r_sc) / 4 * 10
        R_ua = R_ue * 10

        if R_ua >= 7.5:
            R_ua += (1 - self.time_step / self.max_steps) * 2

        if self.time_step == self.max_steps and R_ua < 5:
            R_ua -= 2

        R_u = np.clip(R_ua, -12, 12)

        info = {'reward_urbanity': R_u, 'total_reward': self.total_reward, 'average_reward': self.total_reward/self.time_step}

        urb_sub = {
            "r_drn": r_drn,
            "r_dg": r_dg,
            "r_lu": r_lu,
            "r_sc": r_sc,
            "R_ue": R_u,
        }

        return R_u, info, urb_sub

    def _calculate_composite_reward(self, raw_VMT, Alpha, capacity, observation_map):
        R_e, _, env_sub = self._calculate_environment_reward(raw_VMT, Alpha, observation_map)
        R_ec, _, eco_sub = self._calculate_economic_reward(raw_VMT, Alpha, capacity)
        R_u, _, urb_sub = self._calculate_urbanity_reward(raw_VMT, Alpha, capacity, observation_map)

        w_env, w_eco, w_urb = self.weights['w1'], self.weights['w2'], self.weights['w3']
        R = (w_env * R_e + w_eco * R_ec + w_urb * R_u)/( w_env + w_eco + w_urb)
        self.update_reward_log_per_timestep(env_sub, eco_sub, urb_sub, step_reward=R)

        self.system_total_env_reward += R_e
        self.system_total_eco_reward += R_ec
        self.system_total_urb_reward += R_u

        self.total_reward += R

        info = {'environment reward': self.system_total_env_reward/self.time_step, 'economic reward': self.system_total_eco_reward/self.time_step, 'urbanity reward': self.system_total_urb_reward/self.time_step, 'total reward': self.total_reward, 'average_reward': self.total_reward/self.time_step, 'raw_VMT': raw_VMT}

        self.record_step_reward_log()

        return R, info, None

    def _update_environment(self, factor, action_group, VMT_indices, PE_indices, capacity, reward, x, y, env_update, reward_info, save_result):

        if self.time_step == 1:
            done = False


        if reward >= 10 or self.time_step == self.max_steps:

            if self.time_step < 11:
                done = False
                terminate = False
                converted_action = np.array([x, y, capacity]).reshape(1, -1)

                self.temp_action_record = np.append(self.temp_action_record, converted_action, axis=0)
                self.time_step += 1
                return_log = self.temp_reward_log
            else:
                converted_action = np.array([x, y, capacity]).reshape(1, -1)
                raw_converted_action = self.conversion_into_extent(converted_action)

                if factor == 'environment':
                    self.reward_environment = np.array([reward_info['reward_environment'], self.select_community]).reshape(1, -1)
                    trajectory_environment = np.hstack((raw_converted_action, self.reward_environment, np.array([[self.episode]])))
                    self.action_record_environment = np.append(self.action_record_environment, trajectory_environment, axis=0)
                    if save_result:
                        np.savetxt("out/result/reward/test/" + "environment_action.csv", self.action_record_environment,delimiter=",", header="x,y,capacity,reward,community_label")

                elif factor == 'economic':
                    self.reward_economy = np.array([reward_info['reward_economy'], self.select_community]).reshape(1, -1)
                    trajectory_economy = np.hstack((raw_converted_action, self.reward_economy, np.array([[self.episode]])))
                    self.action_record_economy = np.append(self.action_record_economy, trajectory_economy,axis=0)
                    if save_result:
                        np.savetxt("out/result/reward/test/" + "economy_action.csv", self.action_record_economy,delimiter=",", header="x,y,capacity,reward,community_label")

                elif factor == 'urbanity':
                    self.reward_urbanity = np.array([reward_info['reward_urbanity'], self.select_community]).reshape(1, -1)
                    trajectory_urbanity = np.hstack((raw_converted_action, self.reward_urbanity, np.array([[self.episode]])))
                    self.action_record_urbanity = np.append(self.action_record_urbanity, trajectory_urbanity, axis=0)
                    if save_result:
                        np.savetxt("out/result/reward/test/" + "urbanity_action.csv", self.action_record_urbanity,delimiter=",", header="x,y,capacity,reward,community_label")

                elif factor == "overall":
                    self.reward_overall = np.array([reward_info['environment reward'], reward_info['economic reward'], reward_info['urbanity reward'], reward_info['total reward'], reward_info['average_reward'], self.select_community]).reshape(1, -1)
                    trajectory_overall = np.hstack((raw_converted_action, self.reward_overall, np.array([[self.episode]])))
                    self.action_record_overall = np.append(self.action_record_overall, trajectory_overall, axis=0)
                    if save_result:
                        np.savetxt("out/result/reward/test/" + "overall_action.csv", self.action_record_overall,delimiter=",", header="x,y,capacity,en_reward, eco_reward, urb_reward, total_reward, avr_reward, community_label")

                elif factor == "system":
                    self.reward_system = np.array([reward_info['environment reward'], reward_info['economic reward'], reward_info['urbanity reward'], reward_info['total reward'], reward_info['average_reward'], self.select_community]).reshape(1, -1)
                    trajectory_system = np.hstack((raw_converted_action, self.reward_system, np.array([[self.episode]])))
                    self.action_record_system = np.append(self.action_record_system, trajectory_system, axis=0)
                    if save_result:
                        np.savetxt("out/result/reward/test/" + "system_action.csv", self.action_record_system,delimiter=",",header="x,y,capacity,en_reward, eco_reward, urb_reward, total_reward, avr_reward, community_label")

                    '''
                    self.reward_average = (self.reward_environment + self.reward_economy + self.reward_urbanity) / 3
                    self.reward_meta = np.array([reward_info['reward_meta']])
                    meta_converted_action = np.array([
                        x, y, capacity,
                        reward_info['environment reward'],
                        reward_info['economic reward'],
                        reward_info['urbanity reward'],
                        self.reward_average.item(), reward_info['average_reward'], self.select_community
                    ], dtype=object).reshape(1, -1)

                    average_action_record = np.array([np.mean(np.array([self.action_record_environment[-1], self.action_record_economy[-1],self.action_record_urbanity[-1]]), axis=0)]).astype(int)
                    average_reward = np.array([
                        self.reward_environment.item(),
                        self.reward_economy.item(),
                        self.reward_urbanity.item(),
                        self.reward_average.item(),
                        self.reward_meta.item(),
                        self.select_community
                    ]).reshape(1,-1)
                    average_value = np.hstack((average_action_record, average_reward))
                    self.average_action_record = np.append(self.average_action_record, average_value, axis=0)
                    self.meta_action_record = np.append(self.meta_action_record, meta_converted_action, axis=0)
                    '''

                if env_update:
                    self._apply_map_updates(factor, action_group, VMT_indices, PE_indices, capacity, x, y)

                '''
                np.savetxt("action_record.csv", self.average_action_record, delimiter=",")
                np.savetxt("out/result/reward/test/" + "meta_action_record.csv", self.meta_action_record,delimiter=",")
                
                '''

                return_log = self.build_return_reward_log_from_history()

                done = True
                terminate = self.episode + 1 == 20000
                self.temp_action_record = np.hstack(([self.initial_position, np.array([12000])]))[np.newaxis, :]
                self.time_step = 1
                self.total_reward = 0
                self.system_total_env_reward, self.system_total_eco_reward, self.system_total_urbanity_reward = 0, 0, 0

        else:
            done = False
            terminate = False
            converted_action = np.array([x, y, capacity]).reshape(1, -1)
            self.temp_action_record = np.append(self.temp_action_record, converted_action, axis=0)
            self.time_step += 1
            return_log = self.temp_reward_log

        return done, terminate, return_log

    def _apply_map_updates(self, factor, action_group, VMT_indices, PE_indices, capacity, x, y):
        if factor == 'overall' or factor == 'system':
            self.main_MAP_[2, x, y] = 0  # vegetation destruction
            self._update_demand(action_group, VMT_indices, PE_indices, capacity)
            self.main_MAP_[13, x, y] += capacity / self.action_converter.capacity() # new charging station layer

    def _update_demand(self, action_group, VMT_indices, PE_indices, capacity):
        if len(VMT_indices) != 0:
            self._update_VMT_demand(action_group, VMT_indices, capacity)
        if len(PE_indices) != 0:
            self._update_PE_demand(action_group, PE_indices, capacity)

    def _update_VMT_demand(self, action_group, VMT_indices, capacity):

        converted_VMT_indices = (action_group + (VMT_indices - np.array([50, 50]))).astype(int)
        remaining_capacity = capacity

        x_indices, y_indices = converted_VMT_indices.T
        vmt_values = self.main_MAP_[5, x_indices, y_indices]

        raw_reductions = self.scalar_VMT_.inverse_transform((0.28 * vmt_values).reshape(-1, 1)).flatten()

        for i, (x, y, r_reduction) in enumerate(
                zip(x_indices, y_indices, raw_reductions)):
            if remaining_capacity <= 0:
                break

            reduction_amount = min(r_reduction, remaining_capacity)

            remaining_capacity -= reduction_amount
            delta = self.scalar_VMT_.transform([[float(reduction_amount)]])[0,0]
            self.main_MAP_[5, x, y] -= delta

    def _update_PE_demand(self, action_group, PE_indices, capacity):

        converted_PE_indices = (action_group + (PE_indices - np.array([50, 50]))).astype(int)
        remaining_capacity = capacity

        x_indices, y_indices = converted_PE_indices.T
        pe_values = self.main_MAP_[6, x_indices, y_indices]

        raw_reductions = self.scalar_PE_.inverse_transform(pe_values.reshape(-1, 1)).flatten()

        for i, (x, y, reduction) in enumerate(zip(x_indices, y_indices, raw_reductions)):
            if remaining_capacity <= 0:
                break

            reduction_amount = min(reduction, remaining_capacity)

            remaining_capacity -= reduction_amount

            delta = self.scalar_PE_.transform([[float(reduction_amount)]])[0,0]

            self.main_MAP_[6, x, y] -= delta

    def equity_evaluation(self):
        """
        Compute nearest-station distance for communities.

        Parameters
        H, W: grid shape
        station_rc: (N, 2) array of (row, col) station indices
        metric: "manhattan (L1)
        return_for: "all" compute for all cells or "pop>0" only where pop>0
        pop: community required if return_for = "pop>0"

        :return:
        dist: distance in cells for selected points
        pts_rc: corresponding points (M,2) as (row, col)
        """

        station_layer = self.main_MAP_[13]
        s_rows, s_cols = np.where(station_layer != 0)
        capacity_values = station_layer[s_rows, s_cols]

        station_list = np.column_stack((s_rows, s_cols, capacity_values))

        existing_station_layer = self.main_MAP_[7]
        e_rows, e_cols = np.where(existing_station_layer != 0)

        existing_station_list = np.column_stack((e_rows, e_cols))

        layer = self.main_MAP_[10] # income
        layer_2 = self.main_MAP_[14] # population density
        rows, cols = np.where(layer != 0)
        income_values = layer[rows, cols]
        population_density_values = layer_2[rows, cols]

        self.income_q33 = np.quantile(income_values, 0.33)
        self.income_q66 = np.quantile(income_values, 0.66)

        weights = np.select(
            [
                income_values <= self.income_q33,
                (income_values > self.income_q33) & (income_values <= self.income_q66),
                income_values > self.income_q66
            ],
            [
                3.0,
                2.0,
                1.7
            ]
        )

        EWCI_result = np.zeros_like(income_values)

        community_list = np.column_stack((rows, cols, income_values, population_density_values, weights, EWCI_result)) #low=3, medium=2, high=1.7

        comm_coords, station_coords, e_station_coords = community_list[:,:3], station_list[:,:3], existing_station_list

        dist = (
            np.abs(comm_coords[:,None,0] - station_coords[None,:,0]) +
            np.abs(comm_coords[:,None,1] - station_coords[None,:,1])
        )

        e_dist = (
            np.abs(comm_coords[:,None,0] - e_station_coords[None,:,0]) +
            np.abs(comm_coords[:,None,1] - e_station_coords[None,:,1])
        )

        threshold = self.radius_cells_for_minutes() # 75 cells

        within_75 = np.any(dist <=threshold, axis=1)
        e_within_75 = np.any(e_dist <=threshold, axis=1)

        filtered_community_list = community_list[within_75]
        e_filtered_community_list = community_list[e_within_75]

        station_within_75 = np.any(dist <=threshold, axis=0)
        e_station_within_75 = np.any(e_dist <= threshold, axis=0)

        filtered_station_list = station_list[station_within_75]
        e_filtered_station_list = existing_station_list[e_station_within_75]

        lowincome_community_list, lowincome_filtered_community_list, e_lowincome_filtered_community_list = community_list[community_list[:, 4] == 3.0], filtered_community_list[filtered_community_list[:, 4] == 3.0], e_filtered_community_list[e_filtered_community_list[:, 4] == 3.0]
        midincome_community_list, midincome_filtered_community_list, e_midincome_filtered_community_list  = community_list[community_list[:, 4] == 2.0], filtered_community_list[filtered_community_list[:, 4] == 2.0], e_filtered_community_list[e_filtered_community_list[:, 4] == 2.0]
        highincome_community_list, highincome_filtered_community_list, e_highincome_filtered_community_list = community_list[community_list[:, 4] == 1.7], filtered_community_list[filtered_community_list[:, 4] == 1.7], e_filtered_community_list[e_filtered_community_list[:, 4] == 1.7]

        if len(filtered_community_list) > 0:
            EWCI_ = sum(filtered_community_list[:,3]*filtered_community_list[:,-2]) /sum(community_list[:,3]*community_list[:,-2]) * 100 # %
        else:
            EWCI_ = 0
        if len(lowincome_community_list) > 0:
            low_EWCI_ = sum(lowincome_filtered_community_list[:, 3]) / sum(lowincome_community_list[:, 3]) * 100
        else:
            low_EWCI_ = 0
        if len(midincome_community_list) > 0:
            mid_EWCI_ = sum(midincome_filtered_community_list[:, 3]) / sum(midincome_community_list[:, 3]) * 100
        else:
            mid_EWCI_ = 0
        if len(highincome_community_list) > 0:
            high_EWCI_ = sum(highincome_filtered_community_list[:, 3]) / sum(highincome_community_list[:, 3]) * 100
        else:
            high_EWCI_ = 0

        if len(e_filtered_community_list) > 0:
            e_EWCI_ = sum(e_filtered_community_list[:,3]*e_filtered_community_list[:,-2]) /sum(community_list[:,3]*community_list[:,-2]) * 100 # %

        else:
            e_EWCI_ = 0

        if len(e_lowincome_filtered_community_list) > 0:
            e_low_EWCI_ = sum(e_lowincome_filtered_community_list[:,3]) / sum(lowincome_community_list[:, 3]) * 100
        else:
            e_low_EWCI_ = 0
        if len(e_midincome_filtered_community_list) > 0:
            e_mid_EWCI_ = sum(e_midincome_filtered_community_list[:,3]) / sum(midincome_community_list[:, 3]) * 100
        else:
            e_mid_EWCI_ = 0
        if len(e_highincome_filtered_community_list) > 0:
            e_high_EWCI_ = sum(e_highincome_filtered_community_list[:,3]) / sum(highincome_community_list[:, 3]) * 100
        else:
            e_high_EWCI_ = 0

        Improved_EWCI_ = EWCI_ / e_EWCI_ #times
        lowincome_Improved_EWCI_ = low_EWCI_ / e_low_EWCI_
        midincome_Improved_EWCI_ = mid_EWCI_ / e_mid_EWCI_
        highincome_Improved_EWCI_ = high_EWCI_ / e_high_EWCI_


        current_ewci = np.array([self.episode, len(station_list), len(filtered_station_list),EWCI_, e_EWCI_, Improved_EWCI_,low_EWCI_, mid_EWCI_, high_EWCI_,  e_low_EWCI_, e_mid_EWCI_, e_high_EWCI_,  lowincome_Improved_EWCI_, midincome_Improved_EWCI_, highincome_Improved_EWCI_])

        print("=" * 50)
        print(f"episode={self.episode}, EWCI={EWCI_}, e_EWCI={e_EWCI_}, Improved_EWCI={Improved_EWCI_}, new_station={len(station_list)}, available_station={len(filtered_station_list)}")
        print(f"Improved_lowincome_EWCI={lowincome_Improved_EWCI_}, Improved_midincome_EWCI={midincome_Improved_EWCI_}, Improved_highincome_EWCI={highincome_Improved_EWCI_}")
        print("=" * 50)

        self.ewci_result.append(current_ewci)

        np.savetxt('ewci_result.csv', self.ewci_result, delimiter=',', header='episode, potential_station, available_station, EWCI, existing_EWCI, Improved_EWCI, lowincome_EWCI, midincome_EWCI, highincome_EWCI, existing_lowincome_EWCI, existing_midincome_EWCI, existing_highincome_EWCI, Improved_lowincome_EWCI, Improved_midincome_EWCI, Improved_highincome_EWCI')

    def radius_cells_for_minutes(self, speed_kmh=30, threshold_min=15, cell_size_m=100):
        """
        Convert a travel-time threshold (minutes) into a grid-radius (cells),
        given average speed (30 km/h) and grid cell size (100 meters).


        :return:
        """

        km_per_min = speed_kmh / 60.0
        dist_km = threshold_min * km_per_min # s = v*t
        cell_km = cell_size_m / 1000.0

        return dist_km / cell_km # if speed = 30 km/h, threshold = 15min, cell_size = 100m -> return 75 cells

    def plot(self, args, rank, meta=False):
        if rank == 0:
            filename = args.reward_folder + '/test/potential_sites_by_{}_map.png'.format("environment")
            x_coords = self.action_record_environment[:, 0]
            y_coords = self.action_record_environment[:, 1]

            plt.figure(figsize=(6, 8))
            plt.hexbin(x_coords, y_coords, gridsize=20, cmap='viridis')

            plt.colorbar(label='Point Density of Selected Charging Stations')
            plt.xlabel('Columns Coordinates (Extent X)')
            plt.ylabel('Rows Coordinates (Extent Y)')
            plt.title('Potential Sites Frequency Map by {} factor in {} community'.format("Environment",
                                                                                          self.select_community))
            plt.savefig(filename)

            plt.close()


        elif rank == 1:
            filename = args.reward_folder + '/test/potential_sites_by_{}_map.png'.format("economy")
            x_coords = self.action_record_economy[:, 0]
            y_coords = self.action_record_economy[:, 1]

            plt.figure(figsize=(6, 8))
            plt.hexbin(x_coords, y_coords, gridsize=20, cmap='viridis')

            plt.colorbar(label='Point Density of Selected Charging Stations')
            plt.xlabel('Columns Coordinates (Extent X)')
            plt.ylabel('Rows Coordinates (Extent Y)')
            plt.title('Potential Sites Frequency Map by {} factor in {} community'.format("Economy",
                                                                                          self.select_community))
            plt.savefig(filename)

            plt.close()

        elif rank == 2:
            filename = args.reward_folder + '/test/potential_sites_by_{}_map.png'.format("urbanity")
            x_coords = self.action_record_urbanity[:, 0]
            y_coords = self.action_record_urbanity[:, 1]

            plt.figure(figsize=(6, 8))
            plt.hexbin(x_coords, y_coords, gridsize=20, cmap='viridis')

            plt.colorbar(label='Point Density of Selected Charging Stations')
            plt.xlabel('Columns Coordinates (Extent X)')
            plt.ylabel('Rows Coordinates (Extent Y)')
            plt.title('Potential Sites Frequency Map by {} factor in {} community'.format("Urbanity",
                                                                                          self.select_community))
            plt.savefig(filename)

            plt.close()

        elif rank == 3:
            filename = args.reward_folder + '/test/potential_sites_by_{}_map.png'.format("overall")
            x_coords = self.action_record_overall[:, 0]
            y_coords = self.action_record_overall[:, 1]

            plt.figure(figsize=(6, 8))
            plt.hexbin(x_coords, y_coords, gridsize=20, cmap='viridis')

            plt.colorbar(label='Point Density of Selected Charging Stations')
            plt.xlabel('Columns Coordinates (Extent X)')
            plt.ylabel('Rows Coordinates (Extent Y)')
            plt.title('Potential Sites Frequency Map by {} factor in {} community'.format("overall",
                                                                                          self.select_community))
            plt.savefig(filename)

            plt.close()

        elif rank == 4:
            filename = args.reward_folder + '/test/potential_sites_by_{}_map.png'.format("system")
            x_coords = self.action_record_system[:, 0]
            y_coords = self.action_record_system[:, 1]

            plt.figure(figsize=(6, 8))
            plt.hexbin(x_coords, y_coords, gridsize=20, cmap='viridis')

            plt.colorbar(label='Point Density of Selected Charging Stations')
            plt.xlabel('Columns Coordinates (Extent X)')
            plt.ylabel('Rows Coordinates (Extent Y)')
            plt.title('Potential Sites Frequency Map by {} factor in {} community'.format("system",
                                                                                          self.select_community))
            plt.savefig(filename)

            plt.close()

    def close(self):
        pygame.quit()
















