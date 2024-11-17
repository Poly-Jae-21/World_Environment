import math
import random
from collections.abc import Mapping

import numpy as np
import pandas as pd
import gemgis as gg
from pyogrio import read_dataframe
from sklearn import preprocessing
import gymnasium as gym
import random

from env.utils.data_conversion import Polygon_to_matrix
from example import selected_position


class ChicagoEnv():

    def __init__(self, config):
        self.max_steps = config['max_steps'] # default: 100
        self.episode_count = 0
        self.action_space = 0

        self.time_step = 0
        self.capacity = 0

        self.fully_environment = None
        self.normalized_fully_environment = None

        self.boundary_x = 0
        self.boundary_y = 0

        self.test = config['test'] #default: False

        self.spdp_list = [1/76] * 76 # the list of starting_point_distribution_probability

        self.boundary_gdf = None

    def Chicago_data(self):

        # Boundary map data (Shape file)
        read_community_boundary_data = read_dataframe("env/MultiCity/Chicago/data/community_boundary_map/geo_export_b5a56d3a9_Project.shp")
        boundary_numpy, boundary_gdf, boundary_minX, boundary_minY, boundary_maxX, boundary_maxY = Polygon_to_matrix().transform_data_landuse(read_community_boundary_data)

        # Landuse data (matrix format)
        read_landuse_data = read_dataframe('env/MultiCity/Chicago/data/landuse_map/Landuse2018_Dissolve_Pr_Clip.shp')
        landuse_numpy, landuse_gdf, landuse_minX, landuse_minY, landuse_maxX, landuse_maxY = Polygon_to_matrix().transform_data_landuse(read_landuse_data)

        # Existing charging infrastructure location data (shape file) = test (validation)
        existing_charging_infra = read_dataframe("env/MultiCity/Chicago/data/existing_infrastructure_map/Alternative_Fuel_Loc_Pr_Clip.shp")
        existing_charging_infra = gg.vector.extract_xy(existing_charging_infra)
        existing_charging_infra.X, existing_charging_infra.Y = np.trunc(existing_charging_infra.X / 10), np.trunc(existing_charging_infra.Y / 10)

        # Traffic count data
        traffic_volume = read_dataframe("env/MultiCity/Chicago/data/traffic_flow_map/RasterT_Traffic3.shp")
        traffic_volume = gg.vector.extract_xy(traffic_volume)
        traffic_volume.X, traffic_volume.Y = np.trunc(traffic_volume.X / 10), np.trunc(traffic_volume.Y / 10)
        raw_traffic_volume = traffic_volume["grid_code"]
        traffic_volumne_lower, traffic_volumne_upper = np.percentile(raw_traffic_volume, 25, method="midpoint"), np.percentile(raw_traffic_volume, 75, method="midpoint")
        IQR = traffic_volumne_upper - traffic_volumne_lower
        traffic_volume_upper_outlier, traffic_volume_lower_outlier = traffic_volumne_upper + 1.5 * IQR, traffic_volumne_lower - 1.5 * IQR
        traffic_volume_upper_array = traffic_volume.index[(raw_traffic_volume >= traffic_volume_upper_outlier)]
        traffic_volume_lower_array = traffic_volume.index[(raw_traffic_volume <= traffic_volume_lower_outlier)]
        traffic_volume = traffic_volume.drop(traffic_volume_upper_array, axis=0)
        traffic_volume = traffic_volume.drop(traffic_volume_lower_array, axis=0)
        traffic_volume["grid_code"] = traffic_volume["grid_code"] / 2 # Consider the share of EV sales in estimating charging demand through traffic count data: 50% target goal of U.S. in 2030
        traffic_volume["grid_code"] = traffic_volume["grid_code"] * 0.28 # Consider the probability of visiting a charging station based on the traffic flow, 28% assumed by Liu et al. 2023 paper

        # Transmission line location data
        read_transmission_line_data = read_dataframe('env/MultiCity/Chicago/data/transmission_line_map/geo_export_d59_Polyg_Pr_Clip.shp')
        transmission_line_numpy, transmission_line_gdf, transmission_line_minX, transmission_line_minY, transmission_line_maxX, transmission_line_maxY = Polygon_to_matrix().transform_data_transmission(read_transmission_line_data)

        # Potential electricity data from zipped file
        file_path = "env/MultiCity/Chicago/data/potential_electricity_map/rooftop_vector-20230817T071422Z-001.zip"
        shapefile_name = "rooftop_vector/buildings_Proj_FeatureToPoin.shp"
        potential_electricity = read_dataframe(f'zip://{file_path}!{shapefile_name}')
        potential_electricity = gg.vector.extract_xy(potential_electricity)
        potential_electricity.X, potential_electricity.Y = np.trunc(potential_electricity.X / 10), np.trunc(potential_electricity.Y / 10)
        raw_potential_electricity = potential_electricity["grid_code"]
        potential_electricity_lower, potential_electricity_upper = np.percentile(raw_potential_electricity, 25, method="midpoint"), np.percentile(raw_potential_electricity, 75, method="midpoint")
        IQR = potential_electricity_upper - potential_electricity_lower
        potential_electricity_upper_outlier, potential_electricity_lower_outlier = potential_electricity_upper + 1.5 * IQR, potential_electricity_lower - 1.5 * IQR
        potential_electricity_upper_array = potential_electricity.index[(raw_potential_electricity >= potential_electricity_upper_outlier)]
        potential_electricity_lower_array = potential_electricity.index[(raw_potential_electricity <= potential_electricity_lower_outlier)]
        potential_electricity = potential_electricity.drop(potential_electricity_upper_array, axis=0)
        potential_electricity = potential_electricity.drop(potential_electricity_lower_array, axis=0)

        # raw (x,y) extent to grid coordinates ( 0 to max)
        self.min_x = int(np.min(boundary_minX,landuse_minX, transmission_line_minX, int(np.min(np.concatenate(traffic_volume.X, existing_charging_infra.X, potential_electricity.X), axis=0))))
        self.max_x = int(np.max(boundary_maxX, landuse_maxX, transmission_line_maxX, int(np.max(np.concatenate(traffic_volume.X, existing_charging_infra.X, potential_electricity.X), axis=0))))
        self.min_y = int(np.min(boundary_minY, landuse_minY, transmission_line_minY, int(np.min(np.concatenate(traffic_volume.Y, existing_charging_infra.Y, potential_electricity.Y), axis=0))))
        self.max_y = int(np.max(boundary_maxY, landuse_maxY, transmission_line_maxY, int(np.max(np.concatenate(traffic_volume.Y, existing_charging_infra.Y, potential_electricity.Y), axis=0))))

        traffic_volume.X, traffic_volume.Y = traffic_volume.X - self.min_x, traffic_volume.Y - self.min_y
        potential_electricity.X, potential_electricity.Y = potential_electricity.X - self.min_x, potential_electricity.Y - self.min_y

        return boundary_numpy, landuse_numpy, existing_charging_infra, transmission_line_numpy, traffic_volume, potential_electricity, boundary_gdf, landuse_gdf, transmission_line_gdf

    def Mapping(self):

        obj1, obj2, obj3, obj4, obj5, obj6, obj1_2, obj2_2, obj4_2 = self.Chicago_data() # boundary_data, landuse_data, existing_charging_infra, transmission_line, traffic_volume, potential_electricity

        """
        Data format
        obj1 (boundary), obj2 (landuse), obj4 (transmission): numpy matrix 
        obj3 (existing charging infra), obj5 (traffic volume), obj6 (potential electricity): gdf 
        oj1_2 (boundary), obj2_2 (landuse), obj4_2 (transmission): gdf
        
        obj1, obj2, obj4 do not need to convert it into numpy matrix.
        
        obj3, obj5, obj6 do need to convert it into numpy matrix.
        
        obj1_2, will use it on storing and managing the locations of potential fast charging stations to define the distribution probability of starting locations
        
        --- Updated MAP format (10/28/2024) ---
        We create 2 types of MAP: i) main MAP for environment, and ii) sub MAP for record & visualization of output in the render mode.
        
        i) Main MAP
        There are two 
        
        """
        Matrix_x_size, Matrix_y_size = int(self.max_x - self.min_x), int(self.max_y - self.min_y)
        self.boundary_x, self.boundary_y = Matrix_x_size, Matrix_y_size

        MAP = np.zeros(shape=(Matrix_y_size+1, Matrix_x_size+1, 4)) # astype = np.int32, Environment (MAP) has four layers, including (1) boundary map, (2) landuse, (3) various geometry data (e.g., potential CS locations, transmission lines, charging demand = road network), (4) potential electricity = building information

        obj1_slice = np.stack([obj1]*4, axis=2)
        obj2_slice = np.stack([obj2]*4, axis=2)
        obj4_slice = np.stack([obj4]*4, axis=2)

        MAP[0] = obj1_slice[:,:,0] # first layer of boundary map

        MAP[1] = obj2_slice[:,:,0] # second layer of landuse map

        MAP[2] = obj4_slice[:,:,1] + MAP[2] # third layer of geometric data for transmission lines

        for ii3 in range(len(obj5)):
            x_val_5, y_val_5 = int(obj5.iloc[ii3, -2]), int(obj5.iloc[ii3, -1])
            traffic_volume_count = obj5.iloc[ii3, -4]
            if x_val_5 > self.max_x or x_val_5 < self.min_x or y_val_5 > self.max_y or y_val_5 < self.min_y:
                continue
            else:
                MAP[Matrix_y_size - y_val_5, x_val_5 - self.min_x, 2] = traffic_volume_count

        for ii4 in range(len(obj6)):
            x_val_6, y_val_6 = int(obj6.iloc[ii4, -2]), int(obj6.iloc[ii4, -1])
            potential_electricity_value = obj6.iloc[ii4, -4]
            if x_val_6 > self.max_x or x_val_6 < self.min_x or y_val_6 > self.max_y or y_val_6 < self.min_y:
                continue
            else:
                MAP[Matrix_y_size - y_val_6, x_val_6 - self.min_x, 3] = potential_electricity_value
        if self.test == True:
            for ii5 in range(len(obj3)):
                x_val_3, y_val_3 = int(obj3.iloc[ii5, -2]), int(obj3.iloc[ii5, -1])
                if x_val_3 > self.max_x or x_val_3 < self.min_x or y_val_3 > self.max_y or y_val_3 < self.min_y:
                    continue
                else:
                    MAP[Matrix_y_size - y_val_3, x_val_3 - self.min_x, 3] = 2
        return MAP

    def Partial_Observation(self, agent_position, MAP):
        partial_observation = MAP[-50 + agent_position[0]: 50 + agent_position[0], -50 + agent_position[1]: 50 + agent_position[1]]
        return partial_observation

    def conversion(self, action_record):
        height = int(self.max_y - self.min_y)
        width = int(self.max_x - self.min_x)

        capacity_value = action_record[...,2]
        capacity = np.reshape(capacity_value, (len(capacity_value),1))

        x_extent = 10 * ((action_record[...,1] - self.min_x) + self.min_x)
        x_extent = np.reshape(x_extent, (len(x_extent),1))

        y_extent = 10 * (height - action_record[...,0] + self.min_y)
        y_extent = np.reshape(y_extent, (len(y_extent),1))

        output_action = np.hstack((x_extent, y_extent, capacity))
        return output_action

    def reset(self):

        self.fully_environment = self.Mapping()

        '''
        min_max_scaler_traffic = preprocessing.MinMaxScaler()
        normalized_traffic = min_max_scaler_traffic.fit_transform(self.fully_environment[:,:,1])

        min_max_scaler_electric = preprocessing.MinMaxScaler()
        normalized_electric = min_max_scaler_electric.fit_transform(self.fully_environment[:,:2])

        self.normalized_fully_environment = np.copy(self.fully_environment)
        self.normalized_fully_environment[:,:,1] = normalized_traffic
        self.normalized_fully_environment[:,:,2] = normalized_electric
        '''


        """
        ### New version ###
        We update the method of staring point definition from random/fixed to - distribution probability of starting point 
        Detailed explanation is in the reference paper (J Heo & SW Chang, 2025) 
         
        """


        initial_position_list = np.random.permutation(np.argwhere(self.fully_environment[:,:,0] == 1)).astype('int32')
        initial_position = initial_position_list[0]

        for ii in range(len(initial_position_list)):
            if (initial_position[0] >= 4207) or (initial_position[0] <= 0) or (initial_position[1] >= 3420) or (initial_position[1] <= 0):
                initial_position = initial_position_list[ii+1]
                continue
            else:
                break

        initial_position = np.array([initial_position])
        initial_observation = self.Partial_Observation(initial_position, self.normalized_fully_environment)

        return initial_position, initial_observation, self.fully_environment, self.normalized_fully_environment
    def reset_free(self, env, action_record, probability_list):
        self.time_step = 0
        self.capacity = 0
        boundary = env[0, :, ]
        if self.episode_count + 1 == 1:
            self.boundary_gdf['number_of_selection'] = 0
            self.boundary_gdf['capacity'] = 0
            self.boundary_gdf['probability'] = 0
            self.boundary_gdf['probability'] = self.spdp_list

            select_community = random.randint(1,76)
            initial_positions_list = np.argwhere(boundary == select_community)
            if initial_positions_list.size > 0:
                selected_initial_position = random.choice(initial_positions_list)
                selected_initial_position = np.array([selected_initial_position])
                initial_observation = self.Partial_Observation(selected_initial_position, env)
                return selected_initial_position, initial_observation
            else:
                print("No positions with the value 3 found.")

        elif 2 <= self.episode_count +1 <= 30:
            select_community = random.randint(1, 76)
            medium_positions_list = np.argwhere(boundary == select_community)
            if medium_positions_list.size > 0:
                selected_medium_position = random.choice(medium_positions_list)
                selected_medium_position = np.array([selected_medium_position])
                medium_observation = self.Partial_Observation(selected_medium_position, env)
                return selected_medium_position, medium_observation
            else:
                print("No positions with the value 3 found.")
                """
                Need to except condition to select the community where charging stations are excessively installed. 
                """

        else:
            updated_weight_list = 1 / (np.exp(probability_list) + len(self.boundary_gdf))
            updated_probability_list = updated_weight_list / sum(updated_weight_list)
            self.spdp_list.append(updated_probability_list)
            self.boundary_gdf['probability'] = np.average(self.spdp_list[-1], self.spdp_list[-2], self.spdp_list[-3])

            selected_community = random.choices(population=[i+1 for i in range(76)], weights=self.spdp_list, k=1)
            high_positions_list = np.argwhere(boundary == selected_community)
            if high_positions_list.size > 0:
                selected_high_position = random.choice(high_positions_list)
                selected_high_position = np.array([selected_high_position])
                high_observation = self.Partial_Observation(selected_high_position, env)
                return selected_high_position, high_observation
            else:
                print("No positions with the value 3 found.")

    def step(self, action_record):
        action_record = action_record
        action_record_ = np.squeeze(np.asarray(action_record))
        recent_action = action_record_[-1]
        recent_action_position = recent_action[0:2]
        action_position_record = action_record_[:-1, 0:2]

        self.capacity += recent_action[2]

        normalized_next_observation = self.Partial_Observation(recent_action, self.normalized_fully_environment)
        next_observation = self.Partial_Observation(recent_action, self.fully_environment)
        observation_position = np.array((50,50))

        """first reward: Check the areas whether it is suitable for the installation of EVFCS or not on land use map and transmission line"""
        landuse_map = next_observation[observation_position[0], observation_position[1], 0]
        if landuse_map == 1:
            reward_1 = 2
        else:
            reward_1 = -2

        """second reward: Existing charging stations and potential charging stations on map 
        to prevent the overlapping charging stations """
        first_layer_partially_observation = next_observation[:,:,0]
        existing_indices = np.argwhere(first_layer_partially_observation == 2)

        if len(existing_indices) == 0:
            reward_2 = +2
        elif 1 <= len(existing_indices) <= 3 :
            reward_2 = -1
        else:
            reward_2 = -2

        total_distance = []

        if len(action_position_record) == 0:
            reward_2 += 0
        else:
            # Calculate the distance from current charging station to each existing charging stations in action record array
            distances = np.linalg.norm(action_position_record - recent_action_position, axis=1)

            # Find the points in action record array where the distance from current charging stations is less than 50
            points_within_distance = action_position_record[distances < 50]

            if len(points_within_distance) == 0:
                reward_2 += 2
            elif 1 <= len(points_within_distance) <= 3:
                reward_2 -= 1
            else:
                reward_2 -= 2

        """
        Third reward: traffic volume is transformed to charging demand through reference. 
        This charging demand is represented by kWh. 
        Charging demand is compared with taken action, which is about potential charging station's capacities.
        If charging demand is higher than potential charging capacities, third reward is penalty to agent (-2)
        """

        traffic_volume = np.argwhere(next_observation[:,:,1] != 0)

        if len(traffic_volume) == 0: # There is no charging demand nearby potential sites.
            reward_3 = -2
        else:
            average_traffic_volume = np.average(traffic_volume)
            charging_demand = 0.2 * average_traffic_volume # The indicator (0.2) should be fixed by the references
            charging_demand = charging_demand * 60.0 * 0.8 # nominal capacity = 60.0 kWh, charging rate = 80 %
            if recent_action[2] > charging_demand:
                reward_3 = +2
            else:
                reward_3 = -2

        """
        Fourth Reward: Balance between potential electricity and capacity of potential EVFCS. 
        """
        potential_electricity = next_observation[:,:,2]
        center_p = np.array([observation_position[0], observation_position[1]])
        potential_electricity_indices = np.argwhere(potential_electricity != 0)
        threshold = 25 # 250m

        potential_electricity_position = potential_electricity_indices[np.where(center_p - potential_electricity_indices, axis=1) <= threshold]
        total_electricity = np.sum(potential_electricity[potential_electricity_position])
        adjustment_electricity = recent_action[2] - total_electricity

        converted_electricity_position = np.zeros((len(potential_electricity_position), 2))
        for ss in range(len(potential_electricity_position)):
            converted_electricity_position[ss] = np.asmatrix(recent_action_position) + (potential_electricity_position[ss] - 50)
        converted_electricity_position = converted_electricity_position.astype('int32')

        if len(converted_electricity_position) == 0:
            reward_4 = -10
        else:
            if total_electricity >= recent_action[2]:
                surplus_electricity = total_electricity - recent_action[2]
                reward_4 = - surplus_electricity / recent_action[2] # it should be modified by reference later.
            else:
                deficiency_electricity = recent_action[2] - total_electricity
                reward_4 = - deficiency_electricity / recent_action[2]

            for ss2 in range(len(potential_electricity_position)):
                next_observation[converted_electricity_position[ss2][0], converted_electricity_position[ss2][1], 2] = 0
                normalized_next_observation[converted_electricity_position[ss2][0], converted_electricity_position[ss2][1], 2] = 0

        reward = reward_1 + reward_2 + reward_3 + reward_4

        if self.time_step == self.max_steps:
            done = True
        else:
            done = False

        if not done:
            step_reward = reward
            self.time_step += 1
        else:
            step_reward = reward

            self.episode_count += 1

        info = {"reward_1": reward_1, "reward_2": reward_2, "reward_3": reward_3, "reward_4": reward_4, "step_reward": step_reward}



        return next_observation, step_reward, done, info, normalized_next_observation
