import torch
import numpy as np
import torch.distributions as distributions
import math

class Action(object):
    def __init__(self, boundary_x, boundary_y):
        self.max_capacity = 12 ## 12 stalls; 1 stall = 250 kWh = 6 MWh/daily; 12 stalls = 72 MWh/day
        self.min_capacity = 2 ## 2 stalls; 12 MWh/day
        self.boundary_x = boundary_x
        self.boundary_y = boundary_y

    def local_action_converter(self, current_position , next_action):
        """
        In the update of local_policy_network, we do not need to record the positions and capacities.
        Thus, this function just exports converted next action for MAP (Not partial observation) to utilize it on next step as current action.
        Current action = converted action already for MAP
        Next action = not-converted action yet for partial observation
        MAP next action = converted action for MAP from next action
        """

        MAP_next_action_x = round(np.tanh(next_action[0])*50)
        MAP_next_action_y = round(np.tanh(next_action[1])*50)
        MAP_next_action = np.array([-MAP_next_action_y, MAP_next_action_x])
        MAP_next_position = (current_position + MAP_next_action).astype('int32')

        original_next_capacity = 6000 * (np.tanh(next_action[2]) * (self.max_capacity-self.min_capacity)/2 + (self.max_capacity+self.min_capacity)/2) ## next_action[0][2] = 0 ~ 1 ex) if 0.1 -> stalls =3
        MAP_next_action = np.concatenate((MAP_next_position, np.array([original_next_capacity]))) # MAP_next_action -> current_action in next step

        return MAP_next_action

    def action_converter(self, action, position_record, action_record):
        boundary_x, boundary_y = self.boundary_x, self.boundary_y
        raw_action_x = (action[0][0] + 1) * 100 / 2 - 50
        raw_action_y = (action[0][1] + 1) * 100 / 2 - 50
        raw_action = np.array([-raw_action_y, raw_action_x])
        real_action_position = (position_record[-1] + raw_action).astype('int32')

        if (real_action_position[0] >= boundary_x) or (real_action_position[0] <= 0) or (real_action_position[1] >= boundary_y) or (real_action_position[1] <= 0):
            reward = -10
            real_action_position = real_action_position

        raw_capacity = (action[0][2] + 1) * (30000 - 0) / 2 + 0 # (value + 1) * ( max capacity in action - min capacity in action) / 2  + min capacity in action

        raw_capacity, real_action_position = np.matrix(raw_capacity.astype('int32')), np.matrix(real_action_position).astype('int32')
        position_record = np.append(position_record, real_action_position, axis=0)
        Action_record = np.concatenate((position_record[-1], raw_capacity), axis=1)
        action_record_ = np.append(action_record, Action_record, axis=0)

        position_record, action_record_ = np.squeeze(np.asarray(position_record)), np.squeeze(np.asarray(action_record_))

        return position_record, action_record_


