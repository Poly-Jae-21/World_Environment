from gymnasium import Env, spaces
import numpy as np
from pyogrio import read_dataframe
from typing_extensions import Optional
import math
import gemgis as gg
import random
import pygame
import math
from os import path
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from template.env_name.envs.utils.action import Action
from template.env_name.envs.utils.data_conversion import Polygon_to_matrix, Density

WINDOW_SIZE = [3420, 4207]


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

        self.time_step = 0

        low = np.array(
            [0, # x
             0, # y
             0, # boundary map
             0, # landuse map
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

        self.observation_space = spaces.Box(low, high)
        self.action_space = spaces.Box(-1, +1, (3,), dtype=np.float32)

        self.scalar_VMT_ = MinMaxScaler(feature_range=(0, 1))
        self.scalar_PE_ = MinMaxScaler(feature_range=(0, 1))

        self.main_MAP = self.Mapping()

        self.episode = 0
        self.initial_position = 0
        self.position_record = []
        self.capacity_record = [0]

        self.temp_action_record = np.array([[0, 0, 0]])

        self.probability_list = []  ## This is for the starting point distribution probability. It can be updated over episodes after 32 self.episode
        self.max_steps = 100
        self.factor = None
        self.service_radius_list = []

        self.select_community = None

        self.Den = 0
        self.radius = 0

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
        read_landuse_data = read_dataframe(
            'C:/Users\S2HubLab\PycharmProjects\World_Environment/template\env_name\envs\data\landuse_map\Landuse2018_Dissolve_Pr_Clip.shp')
        landuse_numpy, landuse_minX, landuse_maxX, landuse_minY, landuse_maxY = PtM.transform_data_landuse(
            read_landuse_data)

        # Import the community boundary map data (matrix format) and convert it into numpy format for sub_MAP
        read_community_boundary_data = read_dataframe(
            'C:/Users\S2HubLab\PycharmProjects\World_Environment/template\env_name\envs\data\community_boundary_map\geo_export_b5a56d3a9_Project.shp')
        boundary_numpy, boundary_minX, boundary_maxX, boundary_minY, boundary_maxY = PtM.transform_data_community_boundary(
            read_community_boundary_data)

        # Import the vegetation map data (matrix format) and convert it into numpy format for sub_MAP
        read_vegetation_data = read_dataframe(
            'C:/Users\S2HubLab\PycharmProjects\World_Environment/template\env_name\envs\data/vegetation_map\SevenCensusWCommunit_Pr_Clip.shp')
        vegetation_numpy, vegetation_minX, vegetation_maxX, vegetation_minY, vegetation_maxY = PtM.transform_data_vegetation(
            read_vegetation_data)
        self.vegetation_percentage_max = np.max(vegetation_numpy) / 100

        # Import the main road map data (shape file)
        read_main_road_data = read_dataframe(
            'C:/Users\S2HubLab\PycharmProjects\World_Environment/template\env_name\envs\data/road_map\geo_export_90a38541d_Pr_Clip.shp')
        main_road_numpy, main_road_minX, main_road_maxX, main_road_minY, main_road_maxY = PtM.transform_data_mainroad(
            read_main_road_data)

        # Existing charging infrastructure locations data (shape file) -> only used in test
        existing_charging_infra = read_dataframe(
            'C:/Users\S2HubLab\PycharmProjects\World_Environment/template\env_name\envs\data\existing_infrastructure_map/alt_fuel_stationsSep_Pr_Clip1.shp')
        existing_charging_infra = gg.vector.extract_xy(existing_charging_infra)
        existing_charging_infra.X, existing_charging_infra.Y = np.trunc(existing_charging_infra.X / 10), np.trunc(
            existing_charging_infra.Y / 10)

        # Traffic AADT data -> Vehicle miles traveled (VMT) data (Point data) ##  AADT * foot * 0.000189394 = VMT
        VMT_data = read_dataframe(
            'C:/Users\S2HubLab\PycharmProjects\World_Environment/template\env_name\envs\data\VMT_point_map\Average_Annual_FeatureT_Clip.shp')
        VMT_data = gg.vector.extract_xy(VMT_data)
        VMT_data.X, VMT_data.Y = np.trunc(VMT_data.X / 10), np.trunc(VMT_data.Y / 10)
        VMT_lower, VMT_upper = np.percentile(VMT_data["VMT_mile"], 25, method='midpoint'), np.percentile(
            VMT_data["VMT_mile"], 75, method='midpoint')
        IQR = VMT_upper - VMT_lower
        VMT_upper_outlier, VMT_lower_outlier = VMT_upper + 1.5 * IQR, VMT_lower - 1.5 * IQR
        VMT_upper_array = VMT_data.index[(VMT_data["VMT_mile"] >= VMT_upper_outlier)]
        VMT_lower_array = VMT_data.index[(VMT_data["VMT_mile"] <= VMT_lower_outlier)]
        VMT_data = VMT_data.drop(VMT_upper_array, axis=0)
        VMT_data = VMT_data.drop(VMT_lower_array, axis=0)
        VMT_data["VMT_mile"] = VMT_data[
                                   "VMT_mile"] / 2  # Consider the share of EV sales in estimating charging demand through traffic count data: 50% target goal of U.S. in 2030

        # Power Grid location data (Polyline format)
        read_PowerGrid_line_data = read_dataframe(
            'C:/Users\S2HubLab\PycharmProjects\World_Environment/template\env_name\envs\data/transmission_line_map\geo_export_d59_Polyg_Pr_Clip.shp')
        PowerGrid_line_numpy, PowerGrid_line_minX, PowerGrid_line_maxX, PowerGrid_line_minY, PowerGrid_line_maxY = PtM.transform_data_transmission(
            read_PowerGrid_line_data)

        # Potential electricity data from zipped file
        file_path = "C:/Users\S2HubLab\PycharmProjects\World_Environment/template\env_name\envs\data\potential_electricity_map/rooftop_vector-20230817T071422Z-001.zip"
        shapefile_name = "rooftop_vector/buildings_Proj_FeatureToPoin.shp"
        potential_electricity = read_dataframe(f'zip://{file_path}!{shapefile_name}')
        potential_electricity = gg.vector.extract_xy(potential_electricity)
        potential_electricity.X, potential_electricity.Y = np.trunc(potential_electricity.X / 10), np.trunc(
            potential_electricity.Y / 10)
        raw_potential_electricity = potential_electricity["MEAN"]
        potential_electricity_lower, potential_electricity_upper = np.percentile(raw_potential_electricity, 25,
                                                                                 method="midpoint"), np.percentile(
            raw_potential_electricity, 75, method="midpoint")
        IQR = potential_electricity_upper - potential_electricity_lower
        potential_electricity_upper_outlier, potential_electricity_lower_outlier = potential_electricity_upper + 1.5 * IQR, potential_electricity_lower - 1.5 * IQR
        potential_electricity_upper_array = potential_electricity.index[
            (raw_potential_electricity >= potential_electricity_upper_outlier)]
        potential_electricity_lower_array = potential_electricity.index[
            (raw_potential_electricity <= potential_electricity_lower_outlier)]
        potential_electricity = potential_electricity.drop(potential_electricity_upper_array, axis=0)
        potential_electricity = potential_electricity.drop(potential_electricity_lower_array, axis=0)

        # raw (x, y) extent to grid coordinates ( 0 to max )
        self.min_x = int(min(boundary_minX, landuse_minX, PowerGrid_line_minX, main_road_minX, vegetation_minX,
                             int(np.min(np.concatenate((VMT_data.X, existing_charging_infra.X, potential_electricity.X),
                                                       axis=0)))))
        self.max_x = int(max(boundary_maxX, landuse_maxX, PowerGrid_line_maxX, main_road_maxX, vegetation_maxX,
                             int(np.max(np.concatenate((VMT_data.X, existing_charging_infra.X, potential_electricity.X),
                                                       axis=0)))))
        self.min_y = int(min(boundary_minY, landuse_minY, PowerGrid_line_minY, main_road_minY, vegetation_minY,
                             int(np.min(np.concatenate((VMT_data.Y, existing_charging_infra.Y, potential_electricity.Y),
                                                       axis=0)))))
        self.max_y = int(max(boundary_maxX, landuse_maxY, PowerGrid_line_maxY, main_road_maxY, vegetation_maxY,
                             int(np.max(np.concatenate((VMT_data.Y, existing_charging_infra.Y, potential_electricity.Y),
                                                       axis=0)))))

        read_community_boundary_data['centroid_x'], read_community_boundary_data['centroid_y'] = \
            read_community_boundary_data['centroid_x'] - self.min_x, read_community_boundary_data[
                'centroid_y'] - self.min_y
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
               
        
        Update ->
        
        main_MAP = includes all layers, [obj1, obj2, obj3, obj4, obj5, obj6, obj7, obj8, new CSs]
        1 layer: boundary (community map) = [community code] 1 ~ 76
        2 layer: landuse map = [non-available/available] non-available = -1, available = +1
        3 layer: vegetation map = the percentage of vegetation canopy
        4 layer: potential EVCSs [capacities] 
        5 layer: Power Grid line map
        6 layer: VMT
        7 layer: Potential electricity
        8 layer: main road 
        9 layer: existing charging infra <- it is for comparing with potential EVCSs in the testing. 
        """

        self.boundary_x, self.boundary_y = int(self.max_x - self.min_x), int(self.max_y - self.min_y)

        main_MAP = np.zeros(shape=(9, self.boundary_y + 1, self.boundary_x + 1))

        VMT = np.array(obj5.iloc[:, -4]).reshape(-1, 1)
        self.normalized_VMT = self.scalar_VMT_.fit_transform(VMT)

        PE = np.array(obj6.iloc[:, -4]).reshape(-1, 1)
        self.normalized_PE = self.scalar_PE_.fit_transform(PE)

        for ii in range(len(obj5)):
            x_val, y_val = int(obj5.iloc[ii, -2]), int(obj5.iloc[ii, -1])

            info = 1
            if x_val > self.boundary_x or x_val < 0 or y_val > self.boundary_y or y_val < 0:
                continue
            else:
                main_MAP[5, self.boundary_y - y_val, x_val] = self.normalized_VMT[ii]

        for ii in range(len(obj6)):
            x_val, y_val = int(obj6.iloc[ii, -2]), int(obj6.iloc[ii, -1])

            info = 1
            if x_val > self.max_x or x_val < 0 or y_val > self.max_y or y_val < 0:
                continue
            else:
                main_MAP[6, self.boundary_y - y_val, x_val] = self.normalized_PE[ii]

        main_MAP[7, :obj7.shape[0], main_MAP.shape[2] - obj7.shape[1]:] = obj7

        main_MAP[4, :obj4.shape[0], main_MAP.shape[2] - obj4.shape[1]:] = obj4


        main_MAP[0, :obj1.shape[0], main_MAP.shape[2] - obj1.shape[1]:] = obj1

        main_MAP[1, :obj2.shape[0], main_MAP.shape[2] - obj2.shape[1]:] = obj2

        main_MAP[2, :obj8.shape[0], main_MAP.shape[2] - obj8.shape[1]:] = obj8

        for ii in range(len(obj3)):
            x_val, y_val = int(obj3.iloc[ii, -2]), int(obj3.iloc[ii, -1])
            if x_val > self.max_x or x_val < 0 or y_val > self.max_y or y_val < 0:
                continue
            else:
                main_MAP[8, self.boundary_y - y_val, x_val] = 1
        return main_MAP

    def conversion_into_extent(self, action_record):
        """
        Converting grid coordinate into x,y extent to visualize the output on ArcGIS Pro
        """
        capacity_value = action_record[..., 2]
        capacity = np.reshape(capacity_value, (len(capacity_value), 1))

        x_extent = 10 * ((action_record[..., 1] - self.min_x) + self.min_x)
        x_extent = np.reshape(x_extent, (len(x_extent), 1))

        y_extent = 10 * (self.boundary_y - action_record[..., 0] + self.min_y)
        y_extent = np.reshape(y_extent, (len(y_extent), 1))

        output_action = np.hstack((x_extent, y_extent, capacity))
        return output_action

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[int] = None, ):

        """
        This is to create environment or set up an initial position and initial partial observation
        """
        self.time_step = 0
        self.initial_position = np.array([0, 0])
        self.episode = options + 1
        self.main_MAP_ = self.main_MAP.copy()

        if not hasattr(self, 'action_converter'):
            self.action_converter = Action(self.boundary_x, self.boundary_y)
        ## Simplify the training model
        if self.episode <= 30:
            self.select_community = random.randint(1, 77)
            initial_position_list = np.argwhere(self.main_MAP_[0] == self.select_community)
            if initial_position_list.size > 0:
                selected_initial_starting_point = random.choice(initial_position_list)
                selected_initial_starting_point = np.array(selected_initial_starting_point)
                self.initial_position = selected_initial_starting_point
                self.temp_action_record = np.hstack(([self.initial_position, np.array([12000])]))[np.newaxis, :]

                start_action = (self.temp_action_record.squeeze(), "None", False)

                info = {"community": self.select_community, "initial_position": self.initial_position.tolist()}

                return self.step(start_action)[0], info
            else:
                print("No positions with the value 3 found")
                info = {"error": f"No valid positions in community {self.select_community}"}
                return None, info

        else:
            Density_weight = self.Den.KernelDensity(self.radius, self.main_MAP_)
            updated_weight_list = 1 / (np.exp(Density_weight) + 77) ## 77 = The number of community areas in Chicago
            self.probability_list = updated_weight_list / np.sum(updated_weight_list)
            selected_community = random.choices(population=[i+1 for i in range(77)], weights=self.probability_list, k=1)[0]
            high_positions_list = np.argwhere(self.main_MAP_[0] == selected_community)
            selected_high_position = np.array(random.choice(high_positions_list))

            while selected_high_position.size > 0:
                if selected_high_position.size > 0:

                    self.initial_position = selected_high_position
                    self.temp_action_record = np.hstack((self.initial_position, np.array([12000])))[np.newaxis, :]
                    info = {"community": selected_community, "initial_position": self.initial_position.tolist()}

                    start_action = (self.temp_action_record.squeeze(), "None", False)

                    return self.step(start_action)[0], info
                else:
                    print("Non valid positions")
                    selected_community = random.choices(population=[i + 1 for i in range(77)], weights=self.probability_list, k=1)[0]
                    high_positions_list = np.argwhere(self.main_MAP_[0] == selected_community)
                    selected_high_position = np.array(random.choice(high_positions_list))

    def step(self, action_with_factor):

        current_position = self.temp_action_record[-1][0:2]
        print(current_position)

        action = action_with_factor[0]

        self.factor = action_with_factor[1]

        env_update = action_with_factor[2]

        converted_action = self.action_converter.local_action_converter(current_position,action) if self.time_step != 0 else action  # next position
        action_group = converted_action[0:2].astype(int)
        x, y, capacity = converted_action[0].astype(int), converted_action[1].astype(int), converted_action[2].astype(
            int)

        next_observation = generate_partial_observation(action_group, self.main_MAP_)

        observation_position = np.array((50, 50))

        VMT_indices, VMT_values, VMT = self._process_indices(next_observation, observation_position, -1)

        PE_indices, PE_values, PE = self._process_indices(next_observation, observation_position, -8)

        # potential EVCS
        indices_all = np.argwhere(self.main_MAP_[3,...] != 0)
        av_of_ops = np.mean(np.linalg.norm(observation_position - indices_all, axis=1)) / 50 if len(indices_all) > 0 else 0

        # power grid
        indices_all = np.argwhere(self.main_MAP_[4] != 0)
        mg = np.min(np.linalg.norm(observation_position - indices_all, axis=1)) / (2*math.sqrt(50))

        # main road
        indices_all = np.argwhere(self.main_MAP_[7] != 0)
        mr = np.min(np.linalg.norm(observation_position - indices_all, axis=1)) / (2*math.sqrt(50))

        converted_PE = np.sum(self.scalar_PE_.inverse_transform(PE_values)) if len(PE_values) > 0 else 0
        Alpha = 1 if converted_PE >= capacity else converted_PE / capacity

        avm = np.mean(next_observation[2][next_observation[2] != 0]) / 100
        if avm == np.nan:
            avm = 0
        VMT = 0 if len(VMT_indices) == 0 else VMT

        if mg == np.nan:
            mg = 0
        if mr == np.nan:
            mr = 0

        next_state = [
            x / self.boundary_x,
            y / self.boundary_y,
            self.main_MAP_[0, x, y]/77,
            self.main_MAP_[1, x, y],
            mg,
            mr,
            avm,
            VMT,
            PE,
            av_of_ops
        ]
        print(self.time_step, env_update, self.factor, action, converted_action, next_state)
        if len(VMT_indices) == 0 or self.main_MAP_[0,x,y] is not self.select_community:
            return self._handle_invalid_action(next_state)

        r, info = self._calculate_reward(self.factor, VMT, PE, Alpha, capacity, next_observation)
        done, terminate = self._update_environment(self.factor, action_group, VMT_indices, PE_indices, capacity, r, x, y, env_update)

        return np.array(next_state, dtype=np.float32), r, done, terminate, info

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
                color = (255, 255, 255)
                if self.main_MAP[1, y, x] == -1:
                    color = (255, 0, 0)  # Red
                elif self.main_MAP[1, y, x] == -2:
                    color = (0, 0, 255)  # Blue
                elif self.main_MAP[1, y, x] == - 8:
                    PE_value = int((self.main_MAP[0, y, x] / 100) * 255)
                    color = (PE_value, 0, 0)
                elif self.main_MAP[1, y, x] == -16:
                    self.window.blit(self.evcs_imgs, cell)
                pygame.draw.rect(self.window, color,
                                 pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size))
        pygame.display.flip()

    def _process_indices(self, observation, observation_position, target_value):
        layer = 5 if target_value == -1 else 6
        indices_all = np.argwhere(observation[layer] != 0)
        filtered_indices = indices_all[np.linalg.norm(observation_position - indices_all, axis=1) < 50]

        if target_value == -1:
            total_values = []
            for x, y in filtered_indices:
                obs_ = observation[5, x, y]
                total_values.append(obs_)

            total_indices = np.unique(filtered_indices, axis=0)

            value = 0.28 * np.sum(total_values)

        else:
            total_values = []
            for x, y in filtered_indices:
                obs_ = observation[6, x, y]

                total_values.append(obs_)

            total_indices = np.unique(filtered_indices, axis=0)

            value = np.sum(total_values)

        return total_indices, np.array(total_values).reshape(-1, 1), value

    def _handle_invalid_action(self, next_state):
        r = -1
        done = self.time_step == self.max_steps
        terminate = (self.episode + 1 == 5000) if done else False
        self._reset_or_continue_episode(done)
        return np.array(next_state, dtype=np.float32), r, done, terminate, {}

    def _reset_or_continue_episode(self, done):
        if done:
            self.temp_action_record = np.hstack(([self.initial_position, np.array([0])]))[np.newaxis, :]
            self.time_step = 0
        else:
            last_action = self.temp_action_record[-1]
            self.temp_action_record = np.append(self.temp_action_record, last_action.reshape(1, -1).astype(int), axis=0)
            self.time_step += 1

    def _calculate_reward(self, factor, VMT, PE, Alpha, capacity, observation_map):
        if factor == 'environment':
            return self._calculate_environment_reward(VMT, Alpha, observation_map)
        elif factor == 'economic':
            return self._calculate_economic_reward(VMT, Alpha, capacity)
        elif factor == 'urbanity':
            return self._calculate_urbanity_reward(VMT, Alpha, capacity, observation_map)
        else:
            return self._calculate_composite_reward(VMT, Alpha, capacity, observation_map)

    def _calculate_environment_reward(self, VMT, Alpha, observation_map):
        avm = np.mean(observation_map[2][observation_map[2] != 0]) / 100
        r_avm = np.exp(-avm)
        viss = observation_map[2, 50, 50] / 100
        r_viss = np.exp(-viss)

        r_apr = VMT * 23.7 / 21.79 - VMT * 0.72576 / 4.56
        r_eser = Alpha * VMT * 0.72576 / 4.56
        r_TER = (r_apr + r_eser) * 0.0005
        r_TER = 1 - np.exp(-r_TER)

        R_e = (r_avm + r_viss + r_TER) / 3
        if R_e >= 0.6:
            R_e = 10
        else:
            R_e = -1

        if self.time_step == self.max_steps and R_e < 0.6:
            R_e = -1

        info = {}

        return R_e, info

    def _calculate_economic_reward(self, VMT, Alpha, capacity):
        z = round(capacity / 6000) if capacity > 0 else 0
        F_z = z * 800 * 20600 * 0.2
        P_G = (Alpha * 0.00526 + (1 - Alpha) * 0.05) * VMT * 365
        P_z = F_z / P_G
        # z / P_G

        R_ec = 100 / P_z

        if R_ec >= 1:
            R_ec = 10
        else:
            R_ec = -1

        if self.time_step == self.max_steps and R_ec < 1:
            R_ec = -1

        info = {}
        return R_ec, info

    def _calculate_urbanity_reward(self, VMT, Alpha, capacity, observation_map):
        r_drn = 1 if observation_map[7] != 0 else 0
        r_dg = 1 if Alpha == 1 else (0.5 if observation_map[4] != 0 else 0)
        r_lu = 1 if observation_map[1, 50, 50] != 0 else 0
        r_sc = 1 if capacity >= (VMT / 4.56) else 0
        R_u = (r_drn + r_dg + r_lu + r_sc) / 4
        if R_u == 1:
            R_u = 10
        else:
            R_u = -1

        if self.time_step == self.max_steps and R_u < 1:
            R_u = -1

        info = {}
        return R_u, info

    def _calculate_composite_reward(self, VMT, Alpha, capacity, observation_map):
        R_e, _ = self._calculate_environment_reward(VMT, Alpha)
        R_ec, _ = self._calculate_economic_reward(VMT, Alpha, capacity)
        R_u, _ = self._calculate_urbanity_reward(VMT, Alpha, capacity, observation_map)
        R = R_e + R_ec + R_u
        info = {'environment reward': R_e, 'economic reward': R_ec, 'urbanity reward': R_u, 'overall reward': R}
        return R, info

    def _update_environment(self, factor, action_group, VMT_indices, PE_indices, capacity, reward, x, y, env_update):
        if self.episode == 1:
            self.action_record_environment = np.hstack(([self.initial_position, np.array([12000])])).reshape(1, -1)
            self.action_record_economy = np.hstack(([self.initial_position, np.array([12000])])).reshape(1, -1)
            self.action_record_urbanity = np.hstack(([self.initial_position, np.array([12000])])).reshape(1, -1)
            self.average_action_record = np.hstack(([self.initial_position, np.array([12000])])).reshape(1, -1)
            self.meta_action_record = np.hstack(([self.initial_position, np.array([12000])])).reshape(1, -1)

        if reward >= 30 or self.time_step == self.max_steps:

            if self.time_step < 20:
                done = False
                terminate = False
                converted_action = np.array([x, y, capacity]).reshape(1, -1)
                self.temp_action_record = np.append(self.temp_action_record, converted_action, axis=0)
                self.time_step += 1
            else:
                converted_action = np.array([x, y, capacity]).reshape(1, -1)
                if factor == 'environment':
                    self.action_record_environment = np.append(self.action_record_environment, converted_action, axis=0)
                elif factor == 'economic':
                    self.action_record_economy = np.append(self.action_record_economy, converted_action, axis=0)
                elif factor == 'urbanity':
                    self.action_record_urbanity = np.append(self.action_record_urbanity, converted_action, axis=0)
                else:
                    average_value = np.array([np.mean(np.array(
                        [self.action_record_environment[-1], self.action_record_economy[-1],
                         self.action_record_urbanity[-1]]), axis=0)]).astype(int)
                    self.average_action_record = np.append(self.average_action_record, average_value, axis=0)
                    self.meta_action_record = np.append(self.meta_action_record, converted_action, axis=0)
                if env_update == True:
                    self._apply_map_updates(factor, action_group, VMT_indices, PE_indices, capacity, x, y)
                done = True
                terminate = self.episode + 1 == 5000
                self.temp_action_record = np.hstack(([self.initial_position, np.array([12000])]))[np.newaxis, :]
                self.time_step = 0
        else:
            done = False
            terminate = False
            converted_action = np.array([x, y, capacity]).reshape(1, -1)
            self.temp_action_record = np.append(self.temp_action_record, converted_action, axis=0)
            self.time_step += 1
        return done, terminate

    def _apply_map_updates(self, factor, action_group, VMT_indices, PE_indices, capacity, x, y):
        if factor is None:
            self.main_MAP_[2, x, y] = 0  # vegetation destruction
            self.main_MAP_[3, x, y] += capacity / 72000
            self._update_demand(action_group, VMT_indices, PE_indices, capacity)

    def _update_demand(self, action_group, VMT_indices, PE_indices, capacity):
        self._update_VMT_demand(action_group, VMT_indices, capacity)
        if len(PE_indices) != 0:
            self._update_PE_demand(action_group, PE_indices, capacity)

    def _update_VMT_demand(self, action_group, VMT_indices, capacity):

        converted_VMT_indices = (action_group + (VMT_indices - np.array([50, 50]))).astype(int)
        remaining_capacity = capacity

        x_indices, y_indices = converted_VMT_indices.T
        vmt_raw_values = self.main_MAP_[5, x_indices, y_indices]

        converted_reductions = self.scalar_VMT_.fit_transform((0.28 * vmt_raw_values).reshape(-1, 1)).flatten()

        for i, (x, y, reduction, c_reduction) in enumerate(
                zip(x_indices, y_indices, vmt_raw_values, converted_reductions)):
            if remaining_capacity <= 0:
                break

            reduction_amount = min(reduction, remaining_capacity)
            remaining_capacity -= reduction_amount
            self.main_MAP_[5, x, y] -= c_reduction

    def _update_PE_demand(self, action_group, PE_indices, capacity):

        converted_PE_indices = (action_group + (PE_indices - np.array([50, 50]))).astype(int)
        remaining_capacity = capacity

        x_indices, y_indices = converted_PE_indices.T
        pe_raw_values = self.main_MAP_[6, x_indices, y_indices]

        for i, (x, y, reduction) in enumerate(zip(x_indices, y_indices, pe_raw_values)):
            if remaining_capacity <= 0:
                break

            remaining_capacity -= reduction

            if remaining_capacity <= 0:
                updated_value = 1 - self.scalar_PE_.fit_transform(
                    np.array([abs(remaining_capacity)]).reshape(-1, 1)).flatten()[0]

                self.main_MAP_[6, x, y] = abs(updated_value)

            else:

                self.main_MAP_[6, x, y] = 0

    def plot(self, args, rank, meta=False):
        if rank == 0:
            filename = args.reward_folder + '/test/potential_sites_by_{}_map.png'.format("environment")
            x_coords = self.action_record_environment[:, 0]
            y_coords = self.action_record_environment[:, 1]

            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()

            heatmap, x_edges, y_edges = np.histogram2d(x_coords, y_coords, bins=[x_max - x_min + 1, y_max - y_min + 1],
                                                       range=[[x_min, x_max + 1], [y_min, y_max + 1]])

            plt.figure(figsize=(6, 8))
            plt.imshow(heatmap.T, origin='lower', extent=[x_min, x_max + 1, y_min, y_max + 1], cmap='viridis')
            plt.colorbar(label='Frequency of Selected sites')
            plt.xlabel('Columns Coordinates (Extent X)')
            plt.ylabel('Rows Coordinates (Extent Y)')
            plt.title('Potential Sites Frequency Map by {} factor in {} community'.format("environment",
                                                                                          self.select_community))
            plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
            plt.savefig(filename)

        elif rank == 1:
            filename = args.reward_folder + '/test/potential_sites_by_{}_map.png'.format("economy")
            x_coords = self.action_record_economy[:, 0]
            y_coords = self.action_record_economy[:, 1]

            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()

            heatmap, x_edges, y_edges = np.histogram2d(x_coords, y_coords, bins=[x_max - x_min + 1, y_max - y_min + 1],
                                                       range=[[x_min, x_max + 1], [y_min, y_max + 1]])

            plt.figure(figsize=(6, 8))
            plt.imshow(heatmap.T, origin='lower', extent=[x_min, x_max + 1, y_min, y_max + 1], cmap='viridis')
            plt.colorbar(label='Frequency of Selected sites')
            plt.xlabel('Columns Coordinates (Extent X)')
            plt.ylabel('Rows Coordinates (Extent Y)')
            plt.title(
                'Potential Sites Frequency Map by {} factor in {} community'.format("economy", self.select_community))
            plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
            plt.savefig(filename)

        elif rank == 2:
            filename = args.reward_folder + '/test/potential_sites_by_{}_map.png'.format("urbanity")
            x_coords = self.action_record_urbanity[:, 0]
            y_coords = self.action_record_urbanity[:, 1]

            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()

            heatmap, x_edges, y_edges = np.histogram2d(x_coords, y_coords, bins=[x_max - x_min + 1, y_max - y_min + 1],
                                                       range=[[x_min, x_max + 1], [y_min, y_max + 1]])

            plt.figure(figsize=(6, 8))
            plt.imshow(heatmap.T, origin='lower', extent=[x_min, x_max + 1, y_min, y_max + 1], cmap='viridis')
            plt.colorbar(label='Frequency of Selected sites')
            plt.xlabel('Columns Coordinates (Extent X)')
            plt.ylabel('Rows Coordinates (Extent Y)')
            plt.title(
                'Potential Sites Frequency Map by {} factor in {} community'.format("urbanity", self.select_community))
            plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
            plt.savefig(filename)

        if meta == True:
            filename = args.reward_folder + '/test/potential_sites_by_{}_map.png'.format("average")
            x_coords = self.average_action_record[:, 0]
            y_coords = self.average_action_record[:, 1]

            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()

            heatmap, x_edges, y_edges = np.histogram2d(x_coords, y_coords, bins=[x_max - x_min + 1, y_max - y_min + 1],
                                                       range=[[x_min, x_max + 1], [y_min, y_max + 1]])

            plt.figure(figsize=(6, 8))
            plt.imshow(heatmap.T, origin='lower', extent=[x_min, x_max + 1, y_min, y_max + 1], cmap='viridis')
            plt.colorbar(label='Frequency of Selected sites')
            plt.xlabel('Columns Coordinates (Extent X)')
            plt.ylabel('Rows Coordinates (Extent Y)')
            plt.title(
                'Potential Sites Frequency Map by {} factor in {} community'.format("average", self.select_community))
            plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
            plt.savefig(filename)

            filename = args.reward_folder + '/test/potential_sites_by_{}_map.png'.format("meta")
            x_coords = self.meta_action_record[:, 0]
            y_coords = self.meta_action_record[:, 1]

            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()

            heatmap, x_edges, y_edges = np.histogram2d(x_coords, y_coords, bins=[x_max - x_min + 1, y_max - y_min + 1],
                                                       range=[[x_min, x_max + 1], [y_min, y_max + 1]])

            plt.figure(figsize=(6, 8))
            plt.imshow(heatmap.T, origin='lower', extent=[x_min, x_max + 1, y_min, y_max + 1], cmap='viridis')
            plt.colorbar(label='Frequency of Selected sites')
            plt.xlabel('Columns Coordinates (Extent X)')
            plt.ylabel('Rows Coordinates (Extent Y)')
            plt.title(
                'Potential Sites Frequency Map by {} factor in {} community'.format("meta", self.select_community))
            plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
            plt.savefig(filename)

    def close(self):
        pygame.quit()
















