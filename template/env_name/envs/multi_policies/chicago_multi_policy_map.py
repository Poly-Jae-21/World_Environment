from gymnasium import Env, spaces
import numpy as np
from pyogrio import read_dataframe
from typing_extensions import Optional
import math
import pyvista as pv
import gemgis as gg
import random
import pygame
from os import path

from sklearn.preprocessing import MinMaxScaler

from template.env_name.envs.utils.action import Action
from template.env_name.envs.utils.data_conversion import Polygon_to_matrix, Density

WINDOW_SIZE = [3420, 4207]

def generate_partial_observation(agent_position, MAP):
    p_observation_map = np.zeros([100, 100])
    information_position_map = np.zeros([100, 100])
    for i in range(0,100):
        for j in range(0,100):
            x_val, y_val = int(agent_position[0]), int(agent_position[1])
            if x_val +i - 50 <= 0 or x_val +i + 50 >= WINDOW_SIZE[1] or y_val +j - 50 <= 0 or y_val +j + 50 >= WINDOW_SIZE[0]:
                continue
            else:
                p_observation_map[i, j] = MAP[0, x_val+i-50, y_val+j-50]
                information_position_map[i, j] = MAP[1, x_val+i-50, y_val+j-50]
    return p_observation_map, information_position_map


def generate_partial_observation_sub(agent_position, sub_MAP):
    p_observation_map = np.zeros([4, 100, 100])
    for i in range(0, 100):
        for j in range(0, 100):
            x_val, y_val = int(agent_position[0]), int(agent_position[1])
            if x_val + i - 50 <= 0 or x_val + i + 50 >= WINDOW_SIZE[1] or y_val + j - 50 <= 0 or y_val + j + 50 >= WINDOW_SIZE[0]:
                continue
            else:
                p_observation_map[:, i, j] = sub_MAP[:, x_val + i - 50, y_val + j - 50]

    return p_observation_map

'''
def generate_partial_observation(agent_position, MAP):
    if agent_position[0] - 50 < 0 or agent_position[0] + 50 > WINDOW_SIZE[1] or agent_position[1] - 50 < 0 or agent_position[1] + 50 > WINDOW_SIZE[0]:
        p_observation_map = np.zeros([2,100,100])
        p_observation_map_ = p_observation_map + MAP[0, -50 + agent_position[0]: 50 + agent_position[0], -50 + agent_position[1]: 50 + agent_position[1]]
        information_position_map = p_observation_map + MAP[1, -50 + agent_position[0]: 50 + agent_position[0], -50 + agent_position[1]: 50 + agent_position[1]]
        return p_observation_map_, information_position_map
    else:
        p_observation_map_ =  MAP[0, -50 + agent_position[0]: 50 + agent_position[0],-50 + agent_position[1]: 50 + agent_position[1]]
        information_position_map = MAP[1, -50 + agent_position[0]: 50 + agent_position[0],-50 + agent_position[1]: 50 + agent_position[1]]
        return p_observation_map_, information_position_map

def generate_partial_observation_sub(agent_position, sub_MAP):
    p_observation_map = sub_MAP[:,-50 + agent_position[0]: 50 + agent_position[0], -50 + agent_position[1]: 50 + agent_position[1]]
    return p_observation_map

'''


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
        "render_modes": ["human"],
        "render_fps": 4,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(10000)

        self.time_step = 0

        self.main_MAP = None ## These are for communicating with local policies and meta policy and updating through meta policy.
        self.sub_MAP = None

        self.episode = 0
        self.initial_position = 0
        self.position_record = []
        self.capacity_record = [0]
        self.temp_action_record = np.array([[0,0,0]])
        self.action_record = np.array([[0,0,0]])
        self.probability_list = [] ## This is for the starting point distribution probability. It can be updated over episodes after 32 self.episode
        self.max_steps = 100
        self.factor = None
        self.service_radius_list = []

        self.Den = 0
        self.radius = 0

        self.scalar_VMT = MinMaxScaler(feature_range=(0, 1))
        self.scalar_PE = MinMaxScaler(feature_range=(0, 1))

        self.normalized_VMT = None
        self.normalized_PE = None

        # pygame utils
        self.window = None
        self.clock = None
        self.cell_size = 1
        self.evcs_imgs = None

        self.render_mode = render_mode

    def Chicago_data(self):
        PtM = Polygon_to_matrix()
        # Import The landuse data (matrix format) and convert it into numpy format for sub_MAP
        read_landuse_data = read_dataframe('C:/Users\S2HubLab\PycharmProjects\World_Environment/template\env_name\envs\data\landuse_map\Landuse2018_Dissolve_Pr_Clip.shp')
        landuse_numpy, landuse_minX, landuse_maxX, landuse_minY, landuse_maxY = PtM.transform_data_landuse(read_landuse_data)

        # Import the community boundary map data (matrix format) and convert it into numpy format for sub_MAP
        read_community_boundary_data = read_dataframe('C:/Users\S2HubLab\PycharmProjects\World_Environment/template\env_name\envs\data\community_boundary_map\geo_export_b5a56d3a9_Project.shp')
        boundary_numpy, boundary_minX, boundary_maxX, boundary_minY, boundary_maxY = PtM.transform_data_community_boundary(read_community_boundary_data)

        # Import the vegetation map data (matrix format) and convert it into numpy format for sub_MAP
        read_vegetation_data = read_dataframe('C:/Users\S2HubLab\PycharmProjects\World_Environment/template\env_name\envs\data/vegetation_map\SevenCensusWCommunit_Pr_Clip.shp')
        vegetation_numpy, vegetation_minX, vegetation_maxX, vegetation_minY, vegetation_maxY = PtM.transform_data_vegetation(read_vegetation_data)
        self.vegetation_percentage_max = np.max(vegetation_numpy)

        # Import the main road map data (shape file)
        read_main_road_data = read_dataframe('C:/Users\S2HubLab\PycharmProjects\World_Environment/template\env_name\envs\data/road_map\geo_export_90a38541d_Pr_Clip.shp')
        main_road_numpy, main_road_minX, main_road_maxX, main_road_minY, main_road_maxY = PtM.transform_data_mainroad(read_main_road_data)

        # Existing charging infrastructure locations data (shape file) -> only used in test
        existing_charging_infra = read_dataframe('C:/Users\S2HubLab\PycharmProjects\World_Environment/template\env_name\envs\data\existing_infrastructure_map/alt_fuel_stationsSep_Pr_Clip1.shp')
        existing_charging_infra = gg.vector.extract_xy(existing_charging_infra)
        existing_charging_infra.X, existing_charging_infra.Y = np.trunc(existing_charging_infra.X / 10), np.trunc(existing_charging_infra.Y / 10)

        # Traffic AADT data -> Vehicle miles traveled (VMT) data (Point data) ##  AADT * foot * 0.000189394 = VMT
        VMT_data = read_dataframe('C:/Users\S2HubLab\PycharmProjects\World_Environment/template\env_name\envs\data\VMT_point_map\Average_Annual_FeatureT_Clip.shp')
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
        read_PowerGrid_line_data = read_dataframe('C:/Users\S2HubLab\PycharmProjects\World_Environment/template\env_name\envs\data/transmission_line_map\geo_export_d59_Polyg_Pr_Clip.shp')
        PowerGrid_line_numpy, PowerGrid_line_minX, PowerGrid_line_maxX, PowerGrid_line_minY, PowerGrid_line_maxY = PtM.transform_data_transmission(read_PowerGrid_line_data)

        # Potential electricity data from zipped file
        file_path = "C:/Users\S2HubLab\PycharmProjects\World_Environment/template\env_name\envs\data\potential_electricity_map/rooftop_vector-20230817T071422Z-001.zip"
        shapefile_name = "rooftop_vector/buildings_Proj_FeatureToPoin.shp"
        potential_electricity = read_dataframe(f'zip://{file_path}!{shapefile_name}')
        potential_electricity = gg.vector.extract_xy(potential_electricity)
        potential_electricity.X, potential_electricity.Y = np.trunc(potential_electricity.X / 10), np.trunc(potential_electricity.Y / 10)
        raw_potential_electricity = potential_electricity["MEAN"]
        potential_electricity_lower, potential_electricity_upper = np.percentile(raw_potential_electricity, 25,method="midpoint"), np.percentile(raw_potential_electricity, 75, method="midpoint")
        IQR = potential_electricity_upper - potential_electricity_lower
        potential_electricity_upper_outlier, potential_electricity_lower_outlier = potential_electricity_upper + 1.5 * IQR, potential_electricity_lower - 1.5 * IQR
        potential_electricity_upper_array = potential_electricity.index[(raw_potential_electricity >= potential_electricity_upper_outlier)]
        potential_electricity_lower_array = potential_electricity.index[(raw_potential_electricity <= potential_electricity_lower_outlier)]
        potential_electricity = potential_electricity.drop(potential_electricity_upper_array, axis=0)
        potential_electricity = potential_electricity.drop(potential_electricity_lower_array, axis=0)

        # raw (x, y) extent to grid coordinates ( 0 to max )
        self.min_x = int(min(boundary_minX, landuse_minX, PowerGrid_line_minX, main_road_minX, vegetation_minX, int(np.min(np.concatenate((VMT_data.X, existing_charging_infra.X, potential_electricity.X), axis=0)))))
        self.max_x = int(max(boundary_maxX, landuse_maxX, PowerGrid_line_maxX, main_road_maxX, vegetation_maxX, int(np.max(np.concatenate((VMT_data.X, existing_charging_infra.X, potential_electricity.X), axis=0)))))
        self.min_y = int(min(boundary_minY, landuse_minY, PowerGrid_line_minY, main_road_minY, vegetation_minY, int(np.min(np.concatenate((VMT_data.Y, existing_charging_infra.Y, potential_electricity.Y), axis=0)))))
        self.max_y = int(max(boundary_maxX, landuse_maxY, PowerGrid_line_maxY, main_road_maxY, vegetation_maxY, int(np.max(np.concatenate((VMT_data.Y, existing_charging_infra.Y, potential_electricity.Y), axis=0)))))

        read_community_boundary_data['centroid_x'], read_community_boundary_data['centroid_y'] = \
        read_community_boundary_data['centroid_x'] - self.min_x, read_community_boundary_data['centroid_y'] - self.min_y
        self.Den = Density(read_community_boundary_data['centroid_x'], read_community_boundary_data['centroid_y'])
        self.radius = self.Den.average_radius()

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
        MAP = np.zeros(shape=(2, self.boundary_y + 1, self.boundary_x + 1))
        sub_MAP = np.zeros(shape=(4, self.boundary_y + 1, self.boundary_x + 1))

        VMT = np.array(obj5.iloc[:, -4]).reshape(-1, 1)
        self.normalized_VMT = self.scalar_VMT.fit_transform(VMT)

        PE = np.array(obj6.iloc[:, -4]).reshape(-1, 1)
        self.normalized_PE = self.scalar_PE.fit_transform(PE)


        for ii in range(len(obj5)):
            x_val, y_val = int(obj5.iloc[ii, -2]), int(obj5.iloc[ii, -1])

            info = -1
            if x_val > self.boundary_x or x_val < 0 or y_val > self.boundary_y or y_val < 0:
                continue
            else:
                MAP[0, self.boundary_y - y_val, x_val] = self.normalized_VMT[ii]
                MAP[1, self.boundary_y - y_val, x_val] = info

        for ii in range(len(obj6)):
            x_val, y_val = int(obj6.iloc[ii, -2]), int(obj6.iloc[ii, -1])

            info = -8
            if x_val > self.max_x or x_val < 0 or y_val > self.max_y or y_val < 0:
                continue
            else:
                MAP[0, self.boundary_y - y_val, x_val] = self.normalized_PE[ii]
                MAP[1, self.boundary_y - y_val, x_val] = info

        MAP[0, :obj7.shape[0], MAP.shape[2] - obj7.shape[1]:] = obj7/(-16) + MAP[0, :obj7.shape[0], MAP.shape[2]-obj7.shape[1]:]
        MAP[1, :obj7.shape[0], MAP.shape[2] - obj7.shape[1]:] = obj7 + MAP[1, :obj7.shape[0], MAP.shape[2] - obj7.shape[1]:]

        MAP[0, :obj4.shape[0], MAP.shape[2] - obj4.shape[1]:] = obj4/(-16) + MAP[0, :obj4.shape[0], MAP.shape[2] - obj4.shape[1]:]
        MAP[1, :obj4.shape[0], MAP.shape[2] - obj4.shape[1]:] = obj4 + MAP[1, :obj4.shape[0], MAP.shape[2] - obj4.shape[1]:]

        sub_MAP[0, :obj1.shape[0], sub_MAP.shape[2] - obj1.shape[1]:] = obj1

        sub_MAP[1, :obj2.shape[0], sub_MAP.shape[2] - obj2.shape[1]:] = obj2

        sub_MAP[2, :obj8.shape[0], sub_MAP.shape[2] - obj8.shape[1]:] = obj8

        for ii in range(len(obj3)):
            x_val, y_val = int(obj3.iloc[ii, -2]), int(obj3.iloc[ii, -1])
            if x_val > self.max_x or x_val < 0 or y_val > self.max_y or y_val < 0:
                continue
            else:
                sub_MAP[3, self.boundary_y - y_val, x_val] = -1
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
        self.initial_position = np.array([0, 0])

        if self.episode == 0:
            select_community = random.randint(1,77)
            self.main_MAP, self.sub_MAP = self.Mapping()
            initial_position_list = np.argwhere(self.sub_MAP[0,...] == select_community)
            if initial_position_list.size > 0:
                selected_initial_starting_point = random.choice(initial_position_list)
                selected_initial_starting_point = np.array(selected_initial_starting_point)
                self.initial_position = selected_initial_starting_point
                self.temp_action_record = np.hstack((self.initial_position, np.array([0])))

                initial_observation, _ = generate_partial_observation(selected_initial_starting_point, self.main_MAP)
                info = {"community": select_community, "initial_position": self.initial_position.tolist()}
                return initial_observation, info
            else:
                print("No positions with the value 3 found")
                info = {"error": f"No valid positions in community {select_community}"}
                return None, info

        elif 1 <= self.episode <= 31:
            while True:
                select_community = random.randint(1,77)
                medium_position_list = np.argwhere( self.sub_MAP[0,...] == select_community)
                np.random.shuffle(medium_position_list)
                for i in medium_position_list:
                    selected_medium_position = np.array(i)
                    medium_observation, medium_indices = generate_partial_observation(selected_medium_position, self.main_MAP)
                    VMT_indices = np.argwhere(medium_observation == -1)

                    VMT_values = [medium_observation[x, y] for x, y in VMT_indices]
                    VMT_values = np.array([value for value in VMT_values if value != 0]).reshape(-1, 1)
                    total_VMT = 0.28 * np.sum(VMT_values) / 4.56

                    if total_VMT >= 12000:
                        self.initial_position = selected_medium_position
                        self.temp_action_record = np.hstack((self.initial_position, np.array([0])))
                        info = {"community": select_community, "initial_position": self.initial_position.tolist()}
                        return medium_observation, info

        else:
            Density_weight = self.Den.KernelDensity(self.radius, self.sub_MAP)
            updated_weight_list = 1 / (np.exp(Density_weight) + 77) ## 77 = The number of community areas in Chicago
            self.probability_list = updated_weight_list / np.sum(updated_weight_list)

            while True:
                selected_community = random.choices(population=[i+1 for i in range(77)], weights=self.probability_list, k=1)[0]
                high_positions_list = np.argwhere(self.sub_MAP[0,...] == selected_community)
                np.random.shuffle(high_positions_list)
                for i in high_positions_list:
                    selected_high_position = np.array(i)
                    high_observation, high_indices = generate_partial_observation(selected_high_position, self.main_MAP)
                    VMT_indices = np.argwhere(high_observation == -1)
                    total_VMT = np.sum(self.scalar_VMT.inverse_transform([high_observation[x, y] for x, y in VMT_indices])) / 4.56
                    if total_VMT >= 12000 and i <= len(high_positions_list) - 1:
                        self.initial_position = selected_high_position
                        self.temp_action_record = np.hstack((self.initial_position, np.array([0])))
                        info = {"community": selected_community, "initial_position": self.initial_position.tolist()}
                        return high_observation, info

    def step(self, action_with_factor):
        action, factor = action_with_factor
        if self.time_step == 0:
            self.factor = factor
            current_position = self.initial_position
            converted_action = Action(self.boundary_x, self.boundary_y).local_action_converter(current_position, action) # next position
        else:
            current_position = self.temp_action_record[-1][0:2]
            converted_action = Action(self.boundary_x, self.boundary_y).local_action_converter(current_position, action)

        action_group = converted_action[0:2].astype(int)
        x, y, capacity = converted_action[0].astype(int), converted_action[1].astype(int), converted_action[2].astype(int)

        next_observation, next_observation_position = generate_partial_observation(action_group, self.main_MAP)
        observation_for_subMap = generate_partial_observation_sub(action_group, self.sub_MAP)
        observation_position = np.array((50,50))
        VMT_indices = np.argwhere(next_observation_position == -1) # charging demand from vehicle miles traveled, refers to the total number of miles traveled by vehicles in a partial observation map as daily.
        VMT_indices = VMT_indices[np.linalg.norm(observation_position - VMT_indices, axis=1) < 50]
        VMT_indices = np.sort(VMT_indices, axis=0)
        VMT_values = [next_observation[x, y] for x, y in VMT_indices]
        VMT_values = np.array([value for value in VMT_values if value != 0]).reshape(-1,1)

        if len(VMT_values) ==0:
            r = -1
            next_observation, _ = generate_partial_observation(current_position, self.main_MAP)
            info = {}

            if self.episode == 5000 and self.time_step == self.max_steps:
                self.temp_action_record = np.array([[0,0,0]])
                self.time_step = 0
                done = True
                terminate = True

            elif self.episode != 5000 and self.time_step == self.max_steps:
                self.temp_action_record = np.array([[0, 0, 0]])
                self.time_step = 0
                done = True
                terminate = False
                if factor is None:
                    self.episode += 1

            else:
                self.temp_action_record = np.append(self.temp_action_record, self.temp_action_record[-1].reshape(1,-1).astype(int), axis=0)
                done = False
                terminate = False
                self.time_step += 1

            return next_observation, r, done, terminate, info

        elif x < 60 or x > 3360 or y < 60 or y > 4147:
            r = -1
            next_observation, _ = generate_partial_observation(current_position, self.main_MAP)
            info = {}
            if self.episode == 5000 and self.time_step == self.max_steps:
                self.temp_action_record = np.array([[0,0,0]])
                self.time_step = 0
                done = True
                terminate = True

            elif self.episode != 5000 and self.time_step == self.max_steps:
                self.temp_action_record = np.array([[0, 0, 0]])
                self.time_step = 0
                done = True
                terminate = False
                if factor is None:
                    self.episode += 1

            else:
                self.temp_action_record = np.append(self.temp_action_record, self.temp_action_record[-1].reshape(1,-1).astype(int), axis=0)
                done = False
                terminate = False
                self.time_step += 1
            return next_observation, r, done, terminate, info

        else:
            VMT = 0.28 * np.sum(self.scalar_VMT.inverse_transform(VMT_values))

            PE_indices = np.argwhere(next_observation_position == -8)
            PE_indices = PE_indices[np.linalg.norm(observation_position - PE_indices, axis=1) < 50]
            PE_indices = np.sort(PE_indices, axis=0)
            PE_values = [next_observation[x, y] for x, y in PE_indices]
            PE_values = np.array([value for value in PE_values if value != 0]).reshape(-1,1)

            if len(PE_values) ==0:
                PE = 0
            else:
                PE = np.sum(self.scalar_PE.inverse_transform(PE_values))

            # alpha = the ratio of replaced electric resources to alternative sources.
            if PE >= capacity:
                Alpha = 1
            else:
                Alpha = (capacity - PE) / capacity # 0 ~ 1

            # Reward function for first policy of environment factor
            if factor == 'environment':

                observation_map_for_avm = observation_for_subMap[2,...]
                avm_indices = np.argwhere( observation_map_for_avm != 0)
                avm = np.average([observation_map_for_avm[x, y] for x, y in avm_indices]) # average of vegetation in observation map
                viss = observation_map_for_avm[50, 50]  # loss of vegetation cover in selected site after installation EVCSs

                r_apr = VMT * 1/21.79 * 23.7 - VMT * 1/4.56 * 0.72576
                r_eser = Alpha * VMT * 1/4.56 * 0.72576
                r_TER = r_apr + r_eser # r_TER = total emission reduction, r_apr = Air pollution reduction, r_eser = electricity sources emission reduction by replacing with solar energy

                R_e = np.log(r_TER * np.exp(-avm) * np.exp(self.vegetation_percentage_max - viss))

                r = R_e

                info = {}

            elif factor == 'economic':

                if capacity <= 0:
                    z = 0
                else:
                    z = round(capacity / 6000)
                rho = 800

                pc = 20600

                F_z = z * rho * pc # rho = the maintenance and management cost coefficient = $ 800 / charger
                P_G = (Alpha * 0.00526 + (1 - Alpha) * 0.05) * VMT    # (alternative electricity + general electricity fees) * VMT
                P_z = (F_z / VMT) - P_G # input costs per charging demand and electricity profit
                R_ec = 1 / P_z # maximization of the profit of EV charging network investor

                r = np.exp(R_ec)

                info = {}

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

                if self.sub_MAP[1, y,x] == 1:
                    r_lu = 1
                else:
                    r_lu = 0

                if capacity >= (VMT / 4.56):
                    r_sc = 1
                else:
                    r_sc = 0


                R_u = (r_drn + r_dg + r_lu + r_sc)/4
                r = R_u

                info = {}

            else:
                observation_map_for_avm = observation_for_subMap[2,...]
                avm_indices = np.argwhere(observation_map_for_avm != 0)
                avm = np.average([observation_map_for_avm[x, y] for x, y in avm_indices])  # average of vegetation in observation map
                viss = observation_map_for_avm[50, 50]  # loss of vegetation cover in selected site after installation EVCSs

                r_apr = VMT * 1 / 21.79 * 23.7 - VMT * 1 / 4.56 * 0.72576
                r_eser = Alpha * VMT * 1 / 4.56 * 0.72576
                r_TER = np.log(r_apr + r_eser) # r_TER = total emission reduction, r_apr = Air pollution reduction, r_eser = electricity sources emission reduction by replacing with solar energy

                R_e = r_TER * math.exp(-avm) * math.exp(self.vegetation_percentage_max - viss)

                if capacity <= 0:
                    z = 0
                else:
                    z = round(capacity / 3000)
                rho = 800

                pc = 20600

                F_z = z * rho * pc  # rho = the maintenance and management cost coefficient = $ 800 / charger
                P_G = (Alpha * 0.00526 + (1 - Alpha) * 0.05) * VMT  # (alternative electricity + general electricity fees) * VMT
                P_z = (F_z / VMT) - P_G  # input costs per charging demand and electricity profit
                R_ec = np.exp(1 / P_z)  # maximization of the profit of EV charging network investor

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

                if self.sub_MAP[1, x, y] == 1:
                    r_lu = 1
                else:
                    r_lu = 0

                if capacity >= (VMT / 4.56):
                    r_sc = 1
                else:
                    r_sc = 0

                R_u = (r_drn + r_dg + r_lu + r_sc)/4

                r = (R_e + R_ec + R_u)/3
                info = {"reward_1": R_e, "reward_2": R_ec, "reward_3": R_u, "overall_reward": r}

            if self.time_step < self.max_steps:
                if r >= 3: ## stop the timestep
                    done = True
                    self.time_step = 0

                    if factor is None: ## stop the rollout and update the environment
                        if self.episode == 5000: ## Terminate = True and stop the training
                            terminate = True

                            # Update the main and sub MAP environment through learning things.
                            self.sub_MAP[2, x, y] = 0
                            self.main_MAP[0, x, y] = capacity / 72000
                            self.main_MAP[1, x, y] = -16
                            self.sub_MAP[3, x, y] = capacity
                            self.action_record = np.append(self.action_record, self.temp_action_record[-1], axis=0)

                            converted_VMT_indices = (action_group + (
                                    VMT_indices - observation_position)).astype('int32')  ## VMT indices in main
                            capacity_ = capacity
                            while capacity_ > 0:
                                for a, b in converted_VMT_indices:
                                    if capacity_ <= 0:
                                        break
                                    capacity_ -= (self.scalar_VMT.inverse_transform(
                                        self.main_MAP[0, a, b].reshape(-1, 1)) * 0.28 / 4.56)
                                    self.main_MAP[0, a, b] *= (1 - 0.28)
                                    if capacity_ <= 0:
                                        service_radius = np.linalg.norm(action_group - (a, b))
                                        self.service_radius_list.append(service_radius)
                                else:
                                    break

                            if PE_indices is not None:
                                converted_PE_indices = (action_group + (
                                        PE_indices - observation_position)).astype('int32')  ## PE indices in main
                                capacity__ = capacity
                                while capacity__ > 0:
                                    for a, b in converted_PE_indices:
                                        if capacity__ <= 0:
                                            break
                                        capacity__ = capacity__ - self.scalar_PE.inverse_transform(
                                            self.main_MAP[0, a, b].reshape(-1, 1))
                                        if capacity__ <= 0:
                                            self.main_MAP[0, a, b] = self.scalar_PE.fit_transform(capacity__)
                                        else:
                                            self.main_MAP[0, a, b] = 0
                                            self.main_MAP[1, a, b] = 0
                                    else:
                                        break
                        else: ## update the environment in meta policy and keep doing the episode  and stop the rollout
                            self.episode += 1
                            self.action_record = np.append(self.action_record, self.temp_action_record, axis=0)
                            terminate = False

                            self.temp_action_record = np.array([[0, 0, 0]])

                            # Update the main and sub MAP environment through learning things.
                            self.sub_MAP[2, x, y] = 0
                            self.main_MAP[0, x, y] = capacity / 72000
                            self.main_MAP[1, x, y] = -16
                            self.sub_MAP[3, x, y] = capacity

                            converted_VMT_indices = (action_group + (
                                        VMT_indices - observation_position)).astype('int32')  ## VMT indices in main
                            capacity_ = capacity
                            while capacity_ > 0:
                                for a, b in converted_VMT_indices:
                                    if capacity_ <= 0:
                                        break
                                    capacity_ -= (self.scalar_VMT.inverse_transform(self.main_MAP[0, a, b].reshape(-1,1)) * 0.28 / 4.56)
                                    self.main_MAP[0, a, b] *= (1 - 0.28)
                                    if capacity_ <= 0:
                                        service_radius = np.linalg.norm(action_group - (a, b))
                                        self.service_radius_list.append(service_radius)
                                else:
                                    break

                            if PE_indices is not None:
                                converted_PE_indices = (action_group + (
                                            PE_indices - observation_position)).astype('int32')  ## PE indices in main
                                capacity__ = capacity
                                while capacity__ > 0:
                                    for a, b in converted_PE_indices:
                                        if capacity__ <= 0:
                                            break
                                        capacity__ = capacity__ - self.scalar_PE.inverse_transform(
                                            self.main_MAP[0, a, b].reshape(-1,1))
                                        if capacity__ <= 0:
                                            self.main_MAP[0, a, b] = self.scalar_PE.fit_transform(capacity__)
                                        else:
                                            self.main_MAP[0, a, b] = 0
                                            self.main_MAP[1, a, b] = 0
                                    else:
                                        break

                    else: ## stop the rollout in local policies
                        self.temp_action_record = np.array([[0, 0, 0]])
                        if self.episode == 5000:
                            terminate = True
                        else:
                            terminate = False
                else: ## Keep doing the rollout in both local policies and meta policy
                    terminate = False
                    converted_action = converted_action.reshape(1,-1)
                    self.temp_action_record = np.append(self.temp_action_record, converted_action, axis=0)
                    done = False
                    self.time_step += 1

            else: ## time_step == max_timestep
                self.time_step = 0
                done = True
                if factor is None: ## Meta learner -> Update the environment in max_timestep
                    if self.episode == 5000: ## Terminate = True
                        terminate = True

                        # Update the main and sub MAP environment through learning things.
                        self.sub_MAP[2, x, y] = 0
                        self.main_MAP[0, x, y] = capacity / 72000
                        self.main_MAP[1, x, y] = -16
                        self.sub_MAP[3, x, y] = capacity
                        self.action_record = np.append(self.action_record, self.temp_action_record[-1],
                                                       axis=0)
                        self.temp_action_record = np.array([[0, 0, 0]])

                        converted_VMT_indices = (action_group + (
                                VMT_indices - observation_position)).astype('int32')  ## VMT indices in main
                        capacity_ = capacity
                        while capacity_ > 0:
                            for a, b in converted_VMT_indices:
                                if capacity_ <= 0:
                                    break
                                capacity_ -= (self.scalar_VMT.inverse_transform(
                                    self.main_MAP[0, a, b].reshape(-1, 1)) * 0.28 / 4.56)
                                self.main_MAP[0, a, b] *= (1 - 0.28)
                                if capacity_ <= 0:
                                    service_radius = np.linalg.norm(action_group - (a, b))
                                    self.service_radius_list.append(service_radius)
                            else:
                                break

                        if PE_indices is not None:
                            converted_PE_indices = (action_group + (
                                    PE_indices - observation_position)).astype('int32')  ## PE indices in main
                            capacity__ = capacity
                            while capacity__ > 0:
                                for a, b in converted_PE_indices:
                                    if capacity__ <= 0:
                                        break
                                    capacity__ = capacity__ - self.scalar_PE.inverse_transform(
                                        self.main_MAP[0, a, b].reshape(-1, 1))
                                    if capacity__ <= 0:
                                        self.main_MAP[0, a, b] = self.scalar_PE.fit_transform(capacity__)
                                    else:
                                        self.main_MAP[0, a, b] = 0
                                        self.main_MAP[1, a, b] = 0
                                else:
                                    break

                    else: ## Terminate = False, but update the environment
                        terminate = False
                        self.episode += 1

                        self.action_record = np.append(self.action_record, self.temp_action_record, axis=0)
                        self.temp_action_record = np.array([[0, 0, 0]])
                        self.time_step = 0

                        # Update the main and sub MAP environment through learning things.
                        self.sub_MAP[2, x, y] = 0
                        self.main_MAP[0, x, y] = capacity / 72000
                        self.main_MAP[1, x, y] = -16
                        self.sub_MAP[3, x, y] = capacity

                        converted_VMT_indices = (action_group + (
                                VMT_indices - observation_position)).astype('int32')  ## VMT indices in main
                        capacity_ = capacity
                        while capacity_ > 0:
                            for a, b in converted_VMT_indices:
                                if capacity_ <= 0:
                                    break
                                capacity_ -= (self.scalar_VMT.inverse_transform(
                                    self.main_MAP[0, a, b].reshape(-1, 1)) * 0.28 / 4.56)
                                self.main_MAP[0, a, b] *= (1 - 0.28)
                                if capacity_ <= 0:
                                    service_radius = np.linalg.norm(action_group - (a, b))
                                    self.service_radius_list.append(service_radius)
                            else:
                                break

                        if PE_indices is not None:
                            converted_PE_indices = (action_group + (
                                    PE_indices - observation_position)).astype('int32')  ## PE indices in main
                            capacity__ = capacity
                            while capacity__ > 0:
                                for a, b in converted_PE_indices:
                                    if capacity__ <= 0:
                                        break
                                    capacity__ = capacity__ - self.scalar_PE.inverse_transform(
                                        self.main_MAP[0, a, b].reshape(-1, 1))
                                    if capacity__ <= 0:
                                        self.main_MAP[0, a, b] = self.scalar_PE.fit_transform(capacity__)
                                    else:
                                        self.main_MAP[0, a, b] = 0
                                        self.main_MAP[1, a, b] = 0
                                else:
                                    break

                else: ## it is for local-multi-policies, not update the environment in max_timesteps
                    self.time_step = 0
                    terminate = False
                    self.temp_action_record = np.array([[0, 0, 0]])

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

        for y in range(self.main_MAP.shape[1]):
            for x in range(self.main_MAP.shape[2]):
                cell = (x * self.cell_size, y * self.cell_size)
                color = (255,255,255)
                if self.main_MAP[1, y, x] == -1:
                    color = (255,0,0) # Red
                elif self.main_MAP[1, y, x] == -2:
                    color = (0,0,255) # Blue
                elif self.main_MAP[1, y, x] == - 8:
                    PE_value = int((self.main_MAP[0, y, x] / 100) * 255)
                    color = (PE_value, 0, 0)
                elif self.main_MAP[1, y, x] == -16:
                    self.window.blit(self.evcs_imgs, cell)
                pygame.draw.rect(self.window, color, pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size))
        pygame.display.flip()

    def close(self):
        pygame.quit()
















