from gym import Env
import numpy as np
from typing_extensions import Optional
import math

def generate_map():
    MAP = np.zeros(shape=(Matrix_y_size + 1, Matrix_x_size + 1, 4))
    return MAP

def generate_partial_observation():
    p_observation_map = MAP[-50 + agent_position[0]: 50 + agent_position[0], -50 + agent_position[1]: 50 + agent_position[1]]
    return p_observation_map

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
        self.desc = 0 # MAP


    def step(self, action, factor):

        action_group = action[0:2]
        x, y, capacity = action[0], action[1], action[2]
        next_observation = generate_partial_observation(action_group, self.desc)
        observation_position = np.array((50,50))
        VMT =  # charging demand from vehicle miles traveled, refers to the total number of miles traveled by vehicles in a partial observation map as daily.
        Alpha =  # alpha = the ratio of replaced electric resources to alternative sources.
        # Reward function for first policy of environment factor
        if factor == 'environment':

            avm = # average of vegetation in observation map
            viss = # loss of vegetation cover in selected site after installation EVCSs

            r_apr = VMT * 1/21.79 * 23.7 - VMT * 1/4.56 * 0.72576
            r_eser = Alpha * VMT * 1/4.56 * 0.72576
            r_TER = r_apr + r_eser # r_TER = total emission reduction, r_apr = Air pollution reduction, r_eser = electricity sources emission reduction by replacing with solar energy

            R_e = r_TER * math.exp(-avm) * math.exp(0.35 - viss) # 0.35 = max vegetation coverage in MAP

        elif factor == 'economic':

            z = int(capacity / 150,000)
            rho = 800

            pc = 20,600

            F_z = z * rho * pc # rho = the maintenance and management cost coefficient = $ 800 / charger
            P_G = (Alpha * 0.00526 + (1 - Alpha) * 0.05) * VMT    # (alternative electricity + general electricity fees) * VMT
            P_z = (F_z / VMT) - P_G # input costs per charging demand and electricity profit
            R_ec = 1 / P_z # maximization of the profit of EV charging network investor

        elif factor == 'urbanity':


            if Alpha == 1:
                r_dg = 1
            else:


            R_u = r_drn + r_dg + r_lu + r_sc
