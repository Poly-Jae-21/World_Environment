import gymnasium as gym
from gymnasium import Env
import numpy as np
from pyogrio import read_dataframe
from typing_extensions import Optional
import math
import gemgis as gg
import random
import pygame
from os import path

from utils.action import Action
from utils.data_conversion import Polygon_to_matrix, Density

WINDOW_SIZE = [3420, 4207]

def generate_partial_observation(agent_position, MAP):
    p_observation_map = MAP[-50 + agent_position[0]: 50 + agent_position[0], -50 + agent_position[1]: 50 + agent_position[1],0]
    information_position_map = MAP[-50 + agent_position[0]: 50 + agent_position[0], -50 + agent_position[1]: 50 + agent_position[1],1]
    return p_observation_map, information_position_map

class ChicagoMultiPolicyMap(Env):
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
        "render.modes": ["human", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.time_step = 0
        self.main_MAP = None
        self.sub_MAP = None
        self.episode = 0
        self.initial_position = 0
        self.position_record = []
        self.capacity_record = [0]
        self.temp_action_record = np.array([[0,0,0]])
        self.action_record = np.array([[0,0,0]])
        self.probability_list = [] ## This is for the starting point distribution probability. It can be updated over episodes after 32 self.episode
        self.max_steps = 100

        # pygame utils
        self.window = None
        self.clock = None
        self.cell_size = 1
        self.evcs_imgs = None

    def Chicago_data(self):
        PtM = Polygon_to_matrix()
        # Import The landuse data (matrix format) and convert it into numpy format for sub_MAP
        read_landuse_data = read_dataframe('envs/data/landuse_map/Landuse2018_Dissolve_Pr_Clip.shp')
        landuse_numpy, landuse_minX, landuse_maxX, landuse_minY, landuse_maxY = PtM.transform_data_landuse(read_landuse_data)

        # Import the community boundary map data (matrix format) and convert it into numpy format for sub_MAP
        read_community_boundary_data = read_dataframe('envs/data/community_boundary_map/geo_export_b5a56d3a9_Project.shp')
        boundary_numpy, boundary_minX, boundary_maxX, boundary_minY, boundary_maxY = PtM.transform_data_community_boundary(read_community_boundary_data)

        # Import the vegetation map data (matrix format) and convert it into numpy format for sub_MAP
        read_vegetation_data = read_dataframe('envs/data/vegetation_map/SevenCensusWCommunit_Pr_Clip.shp')
        vegetation_numpy, vegetation_minX, vegetation_maxX, vegetation_minY, vegetation_maxY = PtM.transform_data_vegetation(read_vegetation_data)
        self.vegetation_percentage_max = np.max(vegetation_numpy)

        # Import the main road map data (shape file)
        read_main_road_data = read_dataframe('envs/data/road_map/geo_export_90a38541d_Pr_Clip.shp')
        main_road_numpy, main_road_minX, main_road_maxX, main_road_minY, main_road_maxY = PtM.transform_data_mainroad(read_main_road_data)

        # Existing charging infrastructure locations data (shape file) -> only used in test
        existing_charging_infra = read_dataframe('envs/data/existing_infrastructure_map/alt_fuel_stationsSep_Pr_Clip1.shp')
        existing_charging_infra = gg.vector.extract_xy(existing_charging_infra)
        existing_charging_infra.X, existing_charging_infra.Y = np.trunc(existing_charging_infra.X / 10), np.trunc(existing_charging_infra.Y / 10)

        # Traffic AADT data -> Vehicle miles traveled (VMT) data (Point data) ##  AADT * foot * 0.000189394 = VMT
        VMT_data = read_dataframe('envs/data/VMT_point_map/Average_Annual_FeatureT_Clip.shp')
        VMT_data = gg.vector.extract_xy(VMT_data)
        VMT_data.X, VMT_data.Y = np.trunc(VMT_data.X / 10), np.trunc(VMT_data.Y / 10)
        VMT_lower, VMT_upper = np.percentile(VMT_data["VMT_mile"], 25, method='midpoint'), np.percentile(VMT_data["VMT_mile"], 75, method='midpoint')
        IQR = VMT_upper - VMT_lower
        VMT_upper_outlier, VMT_lower_outlier = VMT_upper + 1.5 * IQR, VMT_lower - 1.5 * IQR
        VMT_upper_array = VMT_data.index[(VMT_data["VMT_mile"] >= VMT_upper_outlier)]
        VMT_lower_array = VMT_data.index[(VMT_data["VMT_mile"] <= VMT_lower_outlier)]
        VMT_data = VMT_data.drop(VMT_upper_array, axis=0)
        VMT_data = VMT_data.drop(VMT_lower_array, axis=0)
        VMT_data["VMT_mile"] = VMT_data["VMT_mile"] / 2 # Consider the share of EV sales in estimating charging demand through traffic count data: 50% target goal of U.S. in 2030

        # Power Grid location data (Polyline format)
        read_PowerGrid_line_data = read_dataframe('envs/data/transmission_line_map/geo_export_d59_Polyg_Pr_Clip.shp')
        PowerGrid_line_numpy, PowerGrid_line_minX, PowerGrid_line_maxX, PowerGrid_line_minY, PowerGrid_line_maxY = PtM.transform_data_transmission(read_PowerGrid_line_data)

        # Potential electricity data from zipped file
        file_path = "envs/data/potential_electricity_map/rooftop_vector-20230817T071422Z-001.zip"
        shapefile_name = "rooftop_vector/buildings_Proj_FeatureToPoin.shp"
        potential_electricity = read_dataframe(f'zip://{file_path}!{shapefile_name}')
        potential_electricity = gg.vector.extract_xy(potential_electricity)
        potential_electricity.X, potential_electricity.Y = np.trunc(potential_electricity.X / 10), np.trunc(potential_electricity.Y / 10)
        raw_potential_electricity = potential_electricity["grid_code"]
        potential_electricity_lower, potential_electricity_upper = np.percentile(raw_potential_electricity, 25,method="midpoint"), np.percentile(raw_potential_electricity, 75, method="midpoint")
        IQR = potential_electricity_upper - potential_electricity_lower
        potential_electricity_upper_outlier, potential_electricity_lower_outlier = potential_electricity_upper + 1.5 * IQR, potential_electricity_lower - 1.5 * IQR
        potential_electricity_upper_array = potential_electricity.index[(raw_potential_electricity >= potential_electricity_upper_outlier)]
        potential_electricity_lower_array = potential_electricity.index[(raw_potential_electricity <= potential_electricity_lower_outlier)]
        potential_electricity = potential_electricity.drop(potential_electricity_upper_array, axis=0)
        potential_electricity = potential_electricity.drop(potential_electricity_lower_array, axis=0)

        # raw (x, y) extent to grid coordinates ( 0 to max )
        self.min_x = int(np.min(boundary_minX, landuse_minX, PowerGrid_line_minX, main_road_minX, vegetation_minX, int(np.min(np.concatenate(VMT_data.X, existing_charging_infra.X, potential_electricity.X), axis=0))))
        self.max_x = int(np.min(boundary_maxX, landuse_maxX, PowerGrid_line_maxX, main_road_maxX, vegetation_maxX, int(np.max(np.concatenate(VMT_data.X, existing_charging_infra.X, potential_electricity.X), axis=0))))
        self.min_y = int(np.min(boundary_minY, landuse_minY, PowerGrid_line_minY, main_road_minY, vegetation_minY, int(np.min(np.concatenate(VMT_data.Y, existing_charging_infra.Y, potential_electricity.Y), axis=0))))
        self.max_y = int(np.min(boundary_maxX, landuse_maxY, PowerGrid_line_maxY, main_road_maxY, vegetation_maxY, int(np.max(np.concatenate(VMT_data.Y, existing_charging_infra.Y, potential_electricity.Y), axis=0))))

        VMT_data.X, VMT_data.Y = VMT_data.X - self.min_x, VMT_data.Y - self.min_y
        potential_electricity.X, potential_electricity.Y = potential_electricity.X - self.min_x, potential_electricity.Y - self.min_y
        existing_charging_infra.X, existing_charging_infra.Y = existing_charging_infra.X - self.min_x, existing_charging_infra.Y - self.min_y


        return boundary_numpy, landuse_numpy, existing_charging_infra, PowerGrid_line_numpy, VMT_data, potential_electricity, main_road_numpy, vegetation_numpy

    def Mapping(self):

        obj1, obj2, obj3, obj4, obj5, obj6, obj7, obj8 = self.Chicago_data()

        """
        Data format
        obj1 (boundary), obj2 (landuse), obj4 (Power Grid line), obj7 (main road), obj8 (vegetation percentage): numpy matrix 
        obj3 (existing charging infra), obj5 (traffic VMT), obj6 (potential electricity): gdf 

        obj1, obj2, obj4, obj7, obj8 do not need to convert it into numpy matrix.
        obj3, obj5, obj6 do need to convert it into numpy matrix.

        --- Updated MAP format (10/28/2024) ---
        We create 2 types of MAP: i) main MAP for environment, and ii) sub MAP for record & visualization of output in the render mode.

        i) Main MAP
        There are two layers in the main MAP: i_1) Quantified information (e.g., VMT), and i_2) location information (e.g., -8 of PVs in (2000, 2000)).
        
        We use only the first layer of the main MAP in training the model as 2D array, which can be trained by CNN networks. 
        
        The second layer of the main MAP is only used to compute the reward function, such as be to calculate the distance between power lines and potential CS location. 
         
        main_MAP = ( , ,2) -> state = ( , ,0) = ( , )
        Vehicle Mileage Traffic (charging demand) = [VMT, -1]
        main road = [-2, -2]
        powerline = [-3, -3]
        PV_potential electricity = [potential electricity generation, -4]
        Potential EVCSs = [capacities, -5]
        
        ii) Sub MAP
        The sub MAP is for constraining the conditions of the installation of EVCSs, cannot be installed in un-available land-use, for figuring out the starting point by the boundary map,
        and for storing the selected potential EVCSs' location information and capacities. 
        
        There are three layers in the sub MAP: ii_1) land-use with available or non-available, ii_2) boundary of communities (1 - 76), and ii_3) potential EVCSs.  
        
        sub_MAP = ( , ) same size with main_MAP
        1 layer: boundary = [non-available/available] non-available = -1, available = +1
        2 layer: landuse = [community code] 1 ~ 76
        3 layer: vegetation = the percentage of vegetation
        4 layer: potential EVCSs [capacities] 
               : Existing EVCSs [location info=-1]
        """

        self.boundary_x, self.boundary_y = int(self.max_x - self.min_x), int(self.max_y - self.min_y)
        obj1['centroid_x'], obj1['centroid_y'] = obj1['centroid_x'] - self.min_x, obj1['centroid_y'] - self.min_y
        self.Den = Density(obj1['centroid_x'], obj1['centroid_y'])
        self.radius = self.Den.average_radius()
        MAP = np.zeros(shape=(self.boundary_y + 1, self.boundary_x + 1, 2))
        sub_MAP = np.zeros(shape=(self.boundary_y + 1, self.boundary_x + 1, 4))

        for ii in range(len(obj5)):
            x_val, y_val = int(obj5.iloc[ii, -2]), int(obj5.iloc[ii, -1])
            VMT = obj5.iloc[ii, -3]
            info = -1
            if x_val > self.max_x or x_val < self.min_x or y_val > self.max_y or y_val < self.min_y:
                continue
            else:
                MAP[self.boundary_y - y_val, x_val - self.min_x, 0] = VMT
                MAP[self.boundary_y - y_val, x_val - self.min_x, 1] = info

        for ii in range(len(obj6)):
            x_val, y_val = int(obj6.iloc[ii, -2]), int(obj6.iloc[ii, -1])
            PE = obj6.iloc[ii, -4]
            info = -8
            if x_val > self.max_x or x_val < self.min_x or y_val > self.max_y or y_val < self.min_y:
                continue
            else:
                MAP[self.boundary_y - y_val, x_val - self.min_x, 0] = PE
                MAP[self.boundary_y - y_val, x_val - self.min_x, 1] = info

        obj7_slice = np.stack([obj7] * 2, axis=2) #-2

        obj4_slice = np.stack([obj4] * 2, axis=2) # -4

        MAP[0] = MAP[0] + obj7_slice[:,:,0] + obj4_slice[:,:,0]
        MAP[1] = MAP[1] + obj7_slice[:,:,0] + obj4_slice[:,:,0]

        obj1_slice = np.stack([obj1] * 4, axis=2)
        obj2_slice = np.stack([obj2] * 4, axis=2)
        obj8_slice = np.stack([obj8] * 4, axis=2)

        sub_MAP[0] = obj1_slice[:,:,0]
        sub_MAP[1] = obj2_slice[:,:,0]
        sub_MAP[2] = obj8_slice[:,:,0]

        for ii in range(len(obj3)):
            x_val, y_val = int(obj3.iloc[ii, -2]), int(obj3.iloc[ii, -1])
            if x_val > self.max_x or x_val < self.min_x or y_val > self.max_y or y_val < self.min_y:
                continue
            else:
                sub_MAP[self.boundary_y - y_val, x_val - self.min_x, 3] = -1
        return MAP, sub_MAP

    def conversion_into_extent(self, action_record):
        """
        Converting grid coordinate into x,y extent to visualize the output on ArcGIS Pro
        """
        capacity_value = action_record[...,2]
        capacity = np.reshape(capacity_value, (len(capacity_value), 1))

        x_extent = 10 * ((action_record[...,1] - self.min_x) + self.min_x)
        x_extent = np.reshape(x_extent, (len(x_extent), 1))

        y_extent = 10 * (self.boundary_y - action_record[...,0] + self.min_y)
        y_extent = np.reshape(y_extent, (len(y_extent), 1))

        output_action = np.hstack((x_extent, y_extent, capacity))
        return output_action

    def reset(self):
        """
        This is to create environment or set up an initial position and initial partial observation
        """
        self.time_step = 0
        self.initial_position = 0

        if self.episode == 0:
            select_community = random.randint(1,77)
            self.main_MAP, self.sub_MAP = self.Mapping()
            initial_position_list = np.argwhere(self.sub_MAP[...,1] == select_community)
            if initial_position_list.size > 0:
                selected_initial_starting_point = random.choice(initial_position_list)
                selected_initial_starting_point = np.array(selected_initial_starting_point)
                self.initial_position = selected_initial_starting_point
                self.temp_action_record = np.hstack((self.initial_position, np.array([[0]])))

                initial_observation = generate_partial_observation(selected_initial_starting_point, self.main_MAP)
                return selected_initial_starting_point, initial_observation
            else:
                print("No positions with the value 3 found")

        elif 1 <= self.episode <= 31:
            select_community = random.randint(1,77)
            medium_position_list = np.argwhere( self.sub_MAP[...,1] == select_community)
            selected_medium_position = random.choice(medium_position_list)
            selected_medium_position = np.array(selected_medium_position)
            self.initial_position = selected_medium_position
            self.temp_action_record = np.hstack((self.initial_position, np.array([[0]])))

            medium_observation = generate_partial_observation(selected_medium_position, self.main_MAP)
            return selected_medium_position, medium_observation

        else:
            Density_weight = self.Den.KernelDensity(self.radius, self.sub_MAP)
            updated_weight_list = 1 / (np.exp(Density_weight) + 77) ## 77 = The number of community areas in Chicago
            self.probability_list = updated_weight_list / np.sum(updated_weight_list)

            selected_community = random.choices(population=[i+1 for i in range(77)], weights=self.probability_list, k=1)
            high_positions_list = np.argwhere(self.sub_MAP[...,1] == selected_community)
            if high_positions_list.size > 0:
                selected_high_position = random.choice(high_positions_list)
                selected_high_position = np.array(selected_high_position)
                self.initial_position = selected_high_position
                self.temp_action_record = np.hstack((self.initial_position, np.array([[0]])))
                high_observation = generate_partial_observation(selected_high_position, self.main_MAP)
                return selected_high_position, high_observation
            else:
                print("Problem 3.0")

    def step(self, action, factor=None):

        if self.time_step == 0:
            current_position = self.initial_position
            converted_action = Action(self.boundary_x, self.boundary_y).local_action_converter(current_position, action) # next position
            self.temp_action_record = np.append(self.temp_action_record, converted_action, axis=0)
        else:
            current_position = self.temp_action_record[-1][0:2]
            converted_action = Action(self.boundary_x, self.boundary_y).local_action_converter(current_position, action)
            self.temp_action_record = np.append(self.temp_action_record, converted_action, axis=0)

        converted_action = self.temp_action_record[-1]
        action_group = converted_action[0:2]
        x, y, capacity = converted_action[0], converted_action[1], converted_action[2]

        next_observation, next_observation_position = generate_partial_observation(action_group, self.main_MAP)
        observation_position = np.array((50,50))
        VMT_indices = np.argwhere(next_observation_position == -1) # charging demand from vehicle miles traveled, refers to the total number of miles traveled by vehicles in a partial observation map as daily.
        VMT_indices = VMT_indices[np.linalg.norm(observation_position - VMT_indices, axis=1) < 50]
        VMT_indices = np.sort(VMT_indices, axis=0)
        VMT = 0.28 * np.sum([next_observation[x, y] for x, y in VMT_indices])

        PE_indices = np.argwhere(next_observation_position == -8)
        PE_indices = PE_indices[np.linalg.norm(observation_position - PE_indices, axis=1) < 50]
        PE_indices = np.sort(PE_indices, axis=0)
        PE = np.sum([next_observation[x, y] for x, y in PE_indices])

        # alpha = the ratio of replaced electric resources to alternative sources.
        if PE >= capacity:
            Alpha = 1
        else:
            Alpha = (capacity - PE) / capacity # 0 ~ 1

        # Reward function for first policy of environment factor
        if factor == 'environment':

            observation_map_for_avm = generate_partial_observation(action_group, self.sub_MAP)
            avm_indices = np.argwhere( observation_map_for_avm != 0)
            avm = np.average([observation_map_for_avm[x, y] for x, y in avm_indices]) # average of vegetation in observation map
            viss = observation_map_for_avm[50, 50]  # loss of vegetation cover in selected site after installation EVCSs

            r_apr = VMT * 1/21.79 * 23.7 - VMT * 1/4.56 * 0.72576
            r_eser = Alpha * VMT * 1/4.56 * 0.72576
            r_TER = r_apr + r_eser # r_TER = total emission reduction, r_apr = Air pollution reduction, r_eser = electricity sources emission reduction by replacing with solar energy

            R_e = r_TER * math.exp(-avm) * math.exp(self.vegetation_percentage_max - viss)

            r = R_e

            info = None

        elif factor == 'economic':

            z = int(capacity / 150,000)
            rho = 800

            pc = 20,600

            F_z = z * rho * pc # rho = the maintenance and management cost coefficient = $ 800 / charger
            P_G = (Alpha * 0.00526 + (1 - Alpha) * 0.05) * VMT    # (alternative electricity + general electricity fees) * VMT
            P_z = (F_z / VMT) - P_G # input costs per charging demand and electricity profit
            R_ec = 1 / P_z # maximization of the profit of EV charging network investor

            r = R_ec

            info = None

        elif factor == 'urbanity':

            main_road_info = np.argwhere(next_observation_position == -2)
            if len(main_road_info) > 0:
                r_drn = 1
            else:
                r_drn = 0

            if Alpha == 1:
                r_dg = 1
            else:
                PowerLine_info = np.argwhere(next_observation_position == -4)
                PowerLine_info = PowerLine_info[np.linalg.norm(observation_position - PowerLine_info, axis=1) < 25]

                if len(PowerLine_info) > 0:
                    r_dg = 0.5
                else:
                    r_dg = 0

            if self.sub_MAP[x,y,1] == 1:
                r_lu = 1
            else:
                r_lu = 0

            if capacity >= (VMT / 4.56):
                r_sc = 1
            else:
                r_sc = 0


            R_u = r_drn + r_dg + r_lu + r_sc
            r = R_u

            info = None

        else:
            observation_map_for_avm = generate_partial_observation(action_group, self.sub_MAP)
            avm_indices = np.argwhere(observation_map_for_avm != 0)
            avm = np.average([observation_map_for_avm[x, y] for x, y in avm_indices])  # average of vegetation in observation map
            viss = observation_map_for_avm[50, 50]  # loss of vegetation cover in selected site after installation EVCSs

            r_apr = VMT * 1 / 21.79 * 23.7 - VMT * 1 / 4.56 * 0.72576
            r_eser = Alpha * VMT * 1 / 4.56 * 0.72576
            r_TER = r_apr + r_eser  # r_TER = total emission reduction, r_apr = Air pollution reduction, r_eser = electricity sources emission reduction by replacing with solar energy

            R_e = r_TER * math.exp(-avm) * math.exp(self.vegetation_percentage_max - viss)

            z = int(capacity / 150, 000)
            rho = 800

            pc = 20, 600

            F_z = z * rho * pc  # rho = the maintenance and management cost coefficient = $ 800 / charger
            P_G = (Alpha * 0.00526 + (1 - Alpha) * 0.05) * VMT  # (alternative electricity + general electricity fees) * VMT
            P_z = (F_z / VMT) - P_G  # input costs per charging demand and electricity profit
            R_ec = 1 / P_z  # maximization of the profit of EV charging network investor

            main_road_info = np.argwhere(next_observation_position == -2)
            if len(main_road_info) > 0:
                r_drn = 1
            else:
                r_drn = 0

            if Alpha == 1:
                r_dg = 1
            else:
                PowerLine_info = np.argwhere(next_observation_position == -4)
                PowerLine_info = PowerLine_info[np.linalg.norm(observation_position - PowerLine_info, axis=1) < 25]

                if len(PowerLine_info) > 0:
                    r_dg = 0.5
                else:
                    r_dg = 0

            if self.sub_MAP[x, y, 1] == 1:
                r_lu = 1
            else:
                r_lu = 0

            if capacity >= (VMT / 4.56):
                r_sc = 1
            else:
                r_sc = 0

            R_u = r_drn + r_dg + r_lu + r_sc

            r = R_e + R_ec + R_u
            info = {"reward_1": R_e, "reward_2": R_ec, "reward_3": R_u, "overall_reward": r}


        if self.time_step == self.max_steps:
            done = True
            r = r
            terminate = None
        elif factor is None and self.time_step != self.max_steps:
            done = False
            r = r
            self.time_step += 1
            terminate = False

        elif factor is None and self.time_step == self.max_steps:
            done = True
            r = r

            # Update the main and sub MAP environment through learning things.
            self.sub_MAP[x, y, 3] = 0
            self.main_MAP[x, y, 0] = capacity
            self.main_MAP[x, y, 1] = -16
            self.sub_MAP[x, y, 2] = capacity
            self.action_record = np.append(self.action_record, self.temp_action_record[-1], axis=0)
            converted_VMT_indices = current_position + (VMT_indices - observation_position)  ## VMT indices in main
            converted_PE_indices = current_position + (PE_indices - observation_position)  ## PE indices in main

            capacity_, capacity__ = capacity, capacity
            while capacity_ > 0:
                for a, b in converted_VMT_indices:
                    capacity_ -= (self.main_MAP[a, b, 0] * 0.28 / 4.56)
                    self.main_MAP[a, b, 0] *= (1 - 0.28)
                    if capacity_ <= 0 or converted_VMT_indices[-1] == (a, b):
                        break

            while capacity__ > 0:
                for a, b in converted_PE_indices:
                    capacity__ -= self.main_MAP[a, b, 0]
                    self.main_MAP[a, b, 0] = 0
                    self.main_MAP[a, b, 1] = 0
                    if capacity__ <= 0 or converted_PE_indices[-1] == (a, b):
                        break
            self.episode += 1
            if self.episode == 5000:
                terminate = True
            else:
                terminate = False
        else:
            done = False
            terminate = None
            r = r
            self.time_step += 1

        return next_observation, r, done, terminate, info

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

        for y in range(self.main_MAP.shape[0]):
            for x in range(self.main_MAP.shape[1]):
                cell = (x * self.cell_size, y * self.cell_size)
                color = (255,255,255)
                if self.main_MAP[y, x, 1] == -1:
                    color = (255,0,0) # Red
                elif self.main_MAP[y, x, 1] == -2:
                    color = (0,0,255) # Blue
                elif self.main_MAP[y, x, 1] == - 8:
                    PE_value = int((self.main_MAP[y, x, 0] / 100) * 255)
                    color = (PE_value, 0, 0)
                elif self.main_MAP[y, x, 1] == -16:
                    self.window.blit(self.evcs_imgs, cell)
                pygame.draw.rect(self.window, color, pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size))
        pygame.display.flip()

    def close(self):
        pygame.quit()













