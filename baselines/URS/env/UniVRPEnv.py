'''
Copyright (C) 2026  CIAM Group, Southern University of Science and Technology, Shenzhen, China.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

from dataclasses import dataclass

import math
import torch

from problem.ProblemDef import get_random_problems, augment_xy_data_by_8_fold
from problem.ProblemSet import ProblemSet

# For registration, these imports are necessary
from env.mask.visited_constraint_mask import mask_visited_nodes
from env.mask.tour_length_constraint_mask import tour_length_constraint_mask
from env.mask.prize_to_collect_constraints_mask import prize_to_collect_constraint_mask
from env.mask.tw_constraint_mask import tw_constraint_mask
from env.mask.route_limit_constraint_mask import route_limit_constraint_mask
from env.mask.pickup_deliver_constraint_mask import pickup_deliver_constraint_mask
from env.mask.bp_constraint_mask import bp_constraint_mask
from env.mask.capacity_constraint_mask import capacity_constraint_mask

@dataclass
class Reset_State:
    problem_name: str = None
    problems: torch.Tensor = None # x,y, demand, prize,fake_prize, penalty,early_time,late_time,service_time
    # shape: (batch, problem+1, 9)
    dist: torch.Tensor = None
    # shape: (batch, problem, problem)
    log_scale: float = None
    relation: torch.Tensor = None

@dataclass
class Step_State:
    batch_size: torch.Tensor = None
    pomo_size: torch.Tensor = None
    depot_num: int = None
    # shape: (batch, pomo)
    selected_count: int = None
    load: torch.Tensor = None
    tour_maxlength: torch.Tensor = None  #op
    length: torch.Tensor = None #l constraint
    collected_prize: torch.Tensor = None
    current_depot: torch.Tensor = None
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)
    START_NODE: torch.Tensor = None
    time: torch.Tensor = None

class UniVRPEnv:
    """
    A unified reinforcement learning environment for Vehicle Routing Problems (UniVRP).

    Designed for Deep RL models (specifically POMO style). 
    It supports a wide range of VRP variants within a single unified framework.
    Detailed supported VRP variants can be found in `problem.ProblemSet.ProblemSet`. 
    The environment dynamically adjusts its logic based on the specified `problem_name`
    
    Core Mechanisms:
        - `Reset_State`: Stores static features (e.g., coordinates, demands, distance matrix).
        - `Step_State`: Tracks dynamic features (e.g., current location, remaining capacity, time).
        - `mask_fn`: An external masking function used to dynamically constrain valid action spaces.

    Attributes:
        problem_name (str): Name of the specific VRP variant (e.g., 'cvrp', 'tsp').
        batch_size (int): Number of parallel environments.
        problem_size (int): Number of customer nodes (excluding depots).
        pomo_size (int): Number of parallel rollouts/starts for the POMO algorithm.
        depot_num (int): Number of depots (default 1, MDVRP uses 3, TSP uses 0).
        reset_state (Reset_State): Static properties initialized at the start.
        step_state (Step_State): Dynamic properties updated per step.
        device (torch.device): Compute device (CPU/GPU).

    Dynamic Attributes (State variables):
        selected_count (int): Number of decoding steps taken.
        current_node (torch.Tensor): Current node index for each instance, shape (batch, pomo).
        load (torch.Tensor): Remaining vehicle capacity, shape (batch, pomo).
        length (torch.Tensor): Accumulated travel distances.
        current_time (torch.Tensor): Current time progress (used for time windows).
        finished (torch.Tensor): Boolean mask indicating if the episode is fully completed.

    Methods:
        - load_problems: Generates random problem instances or loads from a dataset.
        - reset: Initializes the environment's dynamic state and returns `reset_state`.
        - pre_step: Retrieves the current `step_state` before an action is taken.
        - step: Executes the chosen node (`selected`) and applies `mask_fn`, updating states 
              and returning rewards if the episode is finished.
        - process_depot_specially: A method to handle depot-specific masking logic and validity checks.
        - get_local_feature: Retrieves the distance vector between the current node and all others.
    """
    def __init__(self):
        # Const @INIT
        ####################################
        self.problem_size = None
        self.pomo_size = None

        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_index = None
        self.device = None

        self.original_xy_lib = None # for lib data
        self.edge_weight_type = None # for lib data

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)
        self.dist = None # shape: (batch, problem+1, problem+1)
        self.depot_num = 0
        self.last_selected_xy = None

        #for variants of vrp
        self.backhaul_ratio = 0.2
        self.speed = 1.0
        self.depot_start, self.depot_end = 0., None  # tw for depot, depot_end depends on problems

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        self.tour_maxlength = None # for op
        self.max_length = float('inf') # constant, for op,we set max_length=4.0
        self.collected_prize = None # for pctsp
        self.dynamic_demand = None #for sdvrp
        self.open_route = False
        self.START_NODE = None
        # shape: (batch, pomo)
        self.length = None
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.current_time = None
        self.finished = None
        # shape: (batch, pomo)
        self.round_error_epsilon = 0.00001 # for precision stability


        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

        self.loc_scaler = None

    def load_problems(self, batch_size, problem_name, problem_size,pomo_size=None,
                      lib_data=None, validation_data=None, aug_factor=1, 
                      start=0,device=None,capacity=None):
        self.batch_size = batch_size
        self.problem_size = problem_size
        if device is not None:
            self.device = device
        self.problem_name = problem_name
        
        # Get depot_num
        if self.problem_name in ['tsp','atsp']:
            self.depot_num = 0
        elif "md" in self.problem_name:
            self.depot_num = 3  #follow routefinder
        else:
            self.depot_num = 1

        # Get pomo_size
        if pomo_size is None:
            self.pomo_size = problem_size
        else:
            self.pomo_size = pomo_size
        if "pd" in self.problem_name:
            if self.pomo_size > problem_size//2:
                self.pomo_size = problem_size//2
        elif "md" in self.problem_name:
            self.pomo_size = problem_size * self.depot_num


        if lib_data is not None:
            xy, demand, dist, prize, penalty, fake_prize, \
            service_time, tw_start, tw_end, route_limit = self._get_lib_data(lib_data)
        elif validation_data is not None:
            xy, demand, dist, prize, penalty, fake_prize,\
            service_time, tw_start, tw_end, route_limit = self._get_validation_data(validation_data,start)
        else:
            # for atsp
            problem_gen_params = {
                'int_min': 0,
                'int_max': 1000 * 1000,
                'scaler': 1000 * 1000
            }
            if 'vrp' in self.problem_name:
                assert capacity is not None, "capacity must be given when randomly generating cvrp instances."

            data = get_random_problems(batch_size, self.problem_size, capacity, 
                                       self.problem_name,
                                       problem_gen_params=problem_gen_params,
                                       depot_num=self.depot_num)
            xy = data['xy']
            demand = data['demand']
            dist = data['dist']
            prize = data['prize']
            penalty = data['penalty']
            fake_prize = data['fake_prize']
            service_time = data['service_time']
            tw_start = data['tw_start']
            tw_end = data['tw_end']
            route_limit = data['route_limit']

        route_limit = route_limit[:, None] if route_limit.dim() == 1 else route_limit

        if aug_factor > 1:
            self.batch_size = self.batch_size * aug_factor
            if aug_factor == 8 and "a" not in self.problem_name: # 8-fold augmentation is only for symmetric problems, as it relies on the symmetry property.
                xy = augment_xy_data_by_8_fold(xy)
            elif "a" in self.problem_name:
                xy = xy.repeat(aug_factor, 1, 1)
            else:
                raise NotImplementedError(f'The augmentation factor {aug_factor} is not implemented.')
            repeat_1d = [demand, prize, penalty, fake_prize, route_limit,
                         service_time, tw_start, tw_end]
            repeat_2d = [dist]

            for i, field in enumerate(repeat_1d):
                repeat_1d[i] = field.repeat(aug_factor, 1)

            for i, field in enumerate(repeat_2d):
                repeat_2d[i] = field.repeat(aug_factor, 1, 1)

            demand, prize, penalty, fake_prize, route_limit, service_time, tw_start, tw_end = repeat_1d
            dist, = repeat_2d


        if lib_data == None:
            if 'tw' in self.problem_name:
                if "a" in self.problem_name:
                    self.depot_end = 1.0
                else:
                    self.depot_end = 3.0
                tw_end[:,:self.depot_num] = self.depot_end


        problems = torch.cat((xy, 
                              demand.unsqueeze(-1),
                              prize.unsqueeze(-1),
                              fake_prize.unsqueeze(-1),
                              penalty.unsqueeze(-1),
                              tw_start.unsqueeze(-1),
                              tw_end.unsqueeze(-1),
                              service_time.unsqueeze(-1)),
                              dim=-1)
        if self.problem_name == "op":
            self.max_length = 4.0
        elif self.problem_name == 'aop':
            self.max_length = 1.0
        elif 'o' in self.problem_name:
            self.open_route = True

        if self.problem_name in ProblemSet.get(included="b", excluded="bp"):
            self.pomo_size = min(int(self.problem_size * (1 - self.backhaul_ratio) * self.depot_num), self.pomo_size)
            self.START_NODE = torch.arange(start=self.depot_num, end=self.problem_size + self.depot_num)[None,
                              :].expand(self.batch_size, -1).to(self.device)
            self.START_NODE = self.START_NODE[demand[:, self.depot_num:] > 0].reshape(self.batch_size, -1)[:,
                              :self.pomo_size].repeat(1, self.depot_num)


        self.reset_state.problems = problems
        self.reset_state.problem_name = self.problem_name
        self.dist = dist
        self.depot_node_demand = demand
        self.prize = prize
        self.route_limit = route_limit
        self.reset_state.dist = self.dist
        self.step_state.START_NODE = self.START_NODE
        self.step_state.depot_num = self.depot_num
        self.tw_start = tw_start
        self.tw_end = tw_end
        self.service_time = service_time


    def _get_lib_data(self,lib_data):
        # ========= Utility: append depot value before node values =========
        def _cat_depot(field, default_value=0):
            """
            Get field from lib_data and prepend a depot value.
            If the field does not exist, return a full tensor with default values.
            """
            if field in lib_data:
                v = lib_data[field].to(self.device)
                depot_v = torch.full((self.batch_size, 1), default_value, device=self.device)
                return torch.cat((depot_v, v), dim=1)
            return torch.full(
                (self.batch_size, self.problem_size + self.depot_num),
                float('inf') if default_value == float('inf') else 0,
                device=self.device,
            )

        # ========= problem-specific =========
        if self.problem_name in ["tsp", "cvrp", "cvrptw"]:
            xy = lib_data["normalized_xy"].to(self.device)
            demand = lib_data["normalized_demand"].to(self.device)
            self.original_xy_lib = lib_data["original_xy_lib"].to(self.device)
            self.edge_weight_type = lib_data.get("edge_weight_type", "EUC_2D")
        else:
            raise ValueError(f"Unknown problem type {self.problem_name} for benchmark evaluation.")

        # ========= Time-related features (shared logic) =========
        service_time = _cat_depot("service_time", default_value=0)
        tw_start = _cat_depot("tw_start", default_value=0)
        tw_end = _cat_depot("tw_end", default_value=self.depot_end)

        dist = torch.cdist(xy, xy, p=2, compute_mode='donot_use_mm_for_euclid_dist')

        # ========= Extra fields (not provided in lib_data) =========
        zero_tensor = lambda: torch.zeros(self.batch_size, self.problem_size + self.depot_num, device=self.device)
        prize = zero_tensor()
        penalty = zero_tensor()
        fake_prize = zero_tensor()

        route_limit = torch.full((self.batch_size,), float('inf'), device=self.device)

        return xy, demand, dist, prize, penalty, fake_prize, service_time, tw_start, tw_end, route_limit

    def _get_validation_data(self, data, start):
        device = self.device

        def _get_field(key, default):
            if key in data:
                return data[key][start:start + self.batch_size].to(device)
            if isinstance(default, torch.Tensor):
                return default.to(device)
            return torch.full(default["shape"], default["value"], device=device)

        xy = _get_field("xy",{"shape": (self.batch_size, self.problem_size + self.depot_num, 2), "value": 0.0})
        demand = _get_field("demand",{"shape": (self.batch_size, self.problem_size + self.depot_num), "value": 0.0})

        if "dist" in data:
            dist = data["dist"][start:start + self.batch_size].to(device)
        else:
            dist = torch.cdist(xy, xy, p=2, compute_mode="donot_use_mm_for_euclid_dist")

        vector_fields = {
            "prize": 0.0,
            "penalty": 0.0,
            "fake_prize": 0.0,
            "service_time": 0.0,
            "tw_start": 0.0,
            "tw_end": float("inf"),
        }

        outputs = {}
        for key, default_value in vector_fields.items():
            outputs[key] = _get_field(key,{"shape": (self.batch_size, self.problem_size + self.depot_num), "value": default_value})

        route_limit = _get_field("route_limit",{"shape": (self.batch_size,), "value": float("inf")})

        return xy, demand, dist, outputs["prize"], outputs["penalty"], outputs["fake_prize"], outputs["service_time"], outputs["tw_start"], outputs["tw_end"], route_limit


    def reset(self):
        self.selected_count = 0
        self.current_node = None
        self.current_depot = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.long)

        self.current_route_num = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.long)  #第几个环路
        self.current_pickup = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.long) #有多少pickup没有送出去

        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size))

        # tour_maxlength is for op
        self.tour_maxlength = torch.ones(size=(self.batch_size, self.pomo_size)) * self.max_length  # max_length<=4.0
        # prize need to  be collected for pctsp
        self.collected_prize = torch.zeros(size=(self.batch_size, self.pomo_size))
        # for l constraint
        self.length = torch.zeros(size=(self.batch_size, self.pomo_size)).to(self.device)
        # for tw constraint
        self.current_time = torch.zeros(size=(self.batch_size, self.pomo_size)).to(self.device)

        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+self.depot_num))
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+self.depot_num))
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)

        # self.last_selected_xy = self.reset_state.problems[:, [0], :2].expand(-1,self.pomo_size,-1)
        self.any_to_depot_dist = self.reset_state.dist[:, :, 0][:, None, :].expand(-1, self.pomo_size, -1)

        # shape:(batch,pomo,problem+1)
        self.dynamic_demand = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)

        self.reset_state.log_scale = math.log2(self.problem_size)

        self.step_state.batch_size = self.batch_size
        self.step_state.pomo_size = self.pomo_size

        #for pdtsp/ pdcvrp
        # The first half (including depot) as True, the latter as False, [1,1...1, 0...0] where 1 is Pick and 0 is Deliver
        self.to_deliver = torch.cat(
            [
                torch.ones(
                    self.batch_size,
                    self.pomo_size,
                    self.problem_size // 2 + 1,
                    dtype=torch.bool,
                ).to(self.device),
                torch.zeros(
                    self.batch_size,
                    self.pomo_size,
                    self.problem_size // 2,
                    dtype=torch.bool,
                ).to(self.device),
            ],
            dim=-1,
        )  # batch,problem_size+1  前面一半包括depot为True，后面为False，  [1,1...1, 0...0]  1是Pick，0是deliver

        if 'pd' in self.problem_name:
            pd_matrix = torch.ones(self.batch_size, self.problem_size + 1, self.problem_size + 1)
            half = self.problem_size // 2
            pairs = torch.arange(1, half + 1)
            pickup = pairs
            deliver = pairs + half
            row = torch.cat([pickup, deliver])
            col = torch.cat([deliver, pickup])
            pd_matrix[:, row, col] = 0  # Associated is 0, not associated is 1
            self.reset_state.relation = pd_matrix

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size).to(
            self.device)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size).to(
            self.device)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.tour_maxlength = self.tour_maxlength
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.length = self.length
        self.step_state.time = self.current_time

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected,mask_fn):
        # selected.shape: (batch, pomo)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        last_node = self.current_node
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node.unsqueeze(-1)), dim=2)
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected < self.depot_num)
        self.current_depot[self.at_the_depot] = selected[self.at_the_depot]
        self.current_route_num[self.at_the_depot] +=1  #第几个环路

        # For PDCVRP
        self.current_pickup[self.at_the_depot] = 0
        self.current_pickup[(selected < self.problem_size//2 + self.depot_num) & (selected >= self.depot_num) ] +=1
        self.current_pickup[(selected >= self.problem_size // 2 + self.depot_num)] -= 1

        # capacity_calculate
        ####################################
        # For SDCVRP, partially fulfill demand if capacity is insufficient; for standard VRP, dynamic_demand is constant
        demand_list = self.dynamic_demand
        gathering_index = selected.unsqueeze(-1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)

        # For CVRPBP, if starting from the depot and directly selecting point B, load should start from 0
        if last_node is not None and "bp" in self.problem_name:
            last_at_depot = last_node < self.depot_num
            begin_with_backhaul = selected_demand < 0
            self.load[last_at_depot & begin_with_backhaul] = 0

        if self.problem_name == "sdcvrp":
            actual_selected_demand = torch.min(selected_demand, self.load)
            self.load -= actual_selected_demand
            self.dynamic_demand = self.dynamic_demand.scatter_add(-1, gathering_index,-actual_selected_demand.unsqueeze(-1))
        else:
            self.load -= selected_demand
        assert (self.load >= -self.round_error_epsilon).all(), "load cannot be negative!"

        self.load[self.at_the_depot] = 1 # refill loaded at the depot

        # length_calculate
        #####################################
        #-----------------op_length ------------------
        new_length = torch.zeros_like(self.tour_maxlength)
        if last_node is not None:
            batch_idx = torch.arange(self.batch_size)[:, None]
            new_length = self.dist[batch_idx, last_node, selected]
        self.tour_maxlength -= new_length
        #-------------- vrps_length -------------------
        self.length = self.length + new_length
        self.length[self.at_the_depot] = 0

        # collect prize for pctsp
        ########################################
        self.collected_prize += self.prize[:, None, :].expand(-1, self.pomo_size, -1).gather(-1, selected[:, :, None]).squeeze()

        #backhaul
        ########################################
        if 'b' in self.problem_name:
            visited_mask = mask_visited_nodes(self)
            visited_mask[~self.at_the_depot] = 0
            unvisited_demand = demand_list + visited_mask
            # shape: (batch, pomo, problem+1)
            linehauls_unserved = torch.where(unvisited_demand > 0., True, False)
            reset_index = self.at_the_depot & (~linehauls_unserved.any(dim=-1))
            # shape: (batch, pomo)
            self.load[reset_index] = 0.


        # calculate time
        self.current_time = torch.max(self.current_time + new_length / self.speed,
                                      self.tw_start[torch.arange(self.batch_size)[:, None], selected]) + \
                            self.service_time[torch.arange(self.batch_size)[:, None], selected]
        self.current_time[self.at_the_depot] = 0

        # mask register
        self.ninf_mask = mask_fn(self)
        self.ninf_mask = self.process_depot_specially(self.ninf_mask)

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.tour_maxlength = self.tour_maxlength
        self.step_state.collected_prize = self.collected_prize
        self.step_state.time = self.current_time
        self.step_state.length = self.length
        self.step_state.current_node = self.current_node
        self.step_state.current_depot = self.current_depot
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        done = self.finished.all()
        if done:
            if self.problem_name in ['pctsp','spctsp','apctsp','aspctsp']:
                reward = -self._get_penalty_travel_distance()   # note the minus sign!
            elif self.problem_name in ['op','aop']:
                reward = self._get_prize()
            elif self.problem_name in ProblemSet.get(included="md"):
                reward = -self._get_md_reward()
            else:
                if self.original_xy_lib is not None:
                    reward = -self._get_travel_distance_lib()
                else:
                    reward = -self._get_total_distance() # note the minus sign!
        else:
            reward = None
        return self.step_state, reward, done


    def _get_md_reward(self):
        num_depots = self.depot_num
        actions = self.selected_node_list.reshape(self.batch_size * self.pomo_size, -1)
        dist_matrix = self.dist.repeat_interleave(self.pomo_size, dim=0)
        b, seq_len = actions.size()

        # go_from / go_to 节点索引
        go_from = actions  # [b, seq_len]
        go_to = torch.roll(go_from, -1, dims=1)  # [b, seq_len]

        # 每个 batch 的索引，用于从矩阵里取数
        batch_idx = torch.arange(b, device=dist_matrix.device).unsqueeze(1).expand_as(go_from)

        # 起点 depot 的修正
        starting_points = self.get_starting_points(actions, num_depots)  # [b, seq_len]
        actual_depot = torch.roll(starting_points, 1, dims=1)  # [b, seq_len]

        # 从矩阵中取距离
        distances = dist_matrix[batch_idx, go_from, go_to]  # [b, seq_len]
        if self.open_route:
            distances_to_depot = 0
        else:
            distances_to_depot = dist_matrix[batch_idx, go_from, actual_depot]  # [b, seq_len]

        is_depot = go_to < num_depots
        distances = torch.where(is_depot, distances_to_depot, distances)

        # depot → depot 的情况，距离设为 0
        is_depot_to_depot = (go_from < num_depots) & (go_to < num_depots)
        distances = torch.where(is_depot_to_depot, torch.zeros_like(distances), distances)

        # 路径长度
        tour_length = distances.sum(-1)  # [b]

        return tour_length.reshape(self.batch_size, self.pomo_size)  # reward 是负的 cost

    def _get_travel_distance_lib(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        assert self.original_xy_lib.size(0) == 1, 'The original_xy_lib should be a single instance.'
        all_xy = self.original_xy_lib.unsqueeze(1).expand(self.batch_size,self.pomo_size, -1, -1) 
        # shape: (aug_factor, pomo, problem+1, 2)
        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        
        segment_lengths_raw = ((ordered_seq - rolled_seq)**2).sum(3).sqrt()
        if self.edge_weight_type == 'CEIL_2D':
            segment_lengths = torch.ceil(segment_lengths_raw)
        elif self.edge_weight_type == 'EUC_2D':
            segment_lengths = torch.floor(segment_lengths_raw + 0.5)
        else:
            segment_lengths = segment_lengths_raw
        # shape: (batch, pomo, selected_list_length)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances


    #calculate through dist
    def _get_total_distance(self):
        node_from = self.selected_node_list
        seq_len = node_from.size(-1)
        # shape: (batch, pomo, node)
        node_to = self.selected_node_list.roll(dims=2, shifts=-1)
        # shape: (batch, pomo, node)
        BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        batch_index = BATCH_IDX[:, :, None].expand(self.batch_size, self.pomo_size, seq_len)
        # shape: (batch, pomo, node)

        selected_cost = self.dist[batch_index, node_from, node_to]
        #shape: (batch, pomo, node)
        if self.loc_scaler:
            selected_cost = torch.round(selected_cost * self.loc_scaler) / self.loc_scaler

        if self.open_route:
            not_to_depot = self.selected_node_list.roll(dims=2, shifts=-1) != 0
            total_distance = (selected_cost * not_to_depot).sum(2)
        else:
            total_distance = selected_cost.sum(2)
        #shape: (batch, pomo)

        return total_distance

    # for op reward
    def _get_prize(self):
        solution = self.selected_node_list.clone()
        visited = torch.zeros((solution.size(0), solution.size(1), self.reset_state.problems.size(-2)))
        visited = visited.scatter(-1, solution, 1)
        prize = (visited * self.reset_state.problems[:, :, 3][:, None, :].expand(-1, solution.size(1), -1)).sum(-1)

        return prize

    def get_local_feature(self):
        # dist.shape: (batch, problem+1, problem+1)
        # current_node.shape: (batch, pomo)
        if self.current_node is None:
            return None

        current_node = self.current_node.unsqueeze(-1).expand(-1, -1, self.problem_size + self.depot_num)
        # shape: (batch, pomo, problem+1)
        cur_dist = self.dist.gather(dim=1,index=current_node)
        # shape: (batch, pomo, problem+1)

        return cur_dist

    #for pctsp/spctsp
    def _get_penalty_travel_distance(self):
        node_from = self.selected_node_list
        seq_len = node_from.size(-1)
        # shape: (batch, pomo, node)
        node_to = self.selected_node_list.roll(dims=2, shifts=-1)
        # shape: (batch, pomo, node)
        BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        batch_index = BATCH_IDX[:, :, None].expand(self.batch_size, self.pomo_size, seq_len)
        # shape: (batch, pomo, node)

        selected_cost = self.dist[batch_index, node_from, node_to]
        travel_distances = selected_cost.sum(2)

        self.problems =self.reset_state.problems
        solution = self.selected_node_list.clone()
        visited = torch.ones((solution.size(0), solution.size(1), self.problems.size(-2)))
        visited = visited.scatter(-1, solution, 0)
        penalty = (visited * self.problems[:, :, 5][:, None, :].expand(-1, solution.size(1), -1)).sum(-1)
        
        return travel_distances + penalty

    def process_depot_specially(self, mask):
        if 'vrp' in self.problem_name or self.problem_name in ['op','aop'] :
            mask[:, :, :self.depot_num][self.at_the_depot.unsqueeze(-1).expand(-1, -1, self.depot_num)] = float('-inf')
            if 'pd' in self.problem_name:
                depot_mask = ~self.at_the_depot.unsqueeze(-1) & (self.current_pickup == 0).unsqueeze(-1)
            else:
                depot_mask = ~self.at_the_depot.unsqueeze(-1)  # [batch, pomo, 1]
            depot_mask = depot_mask.expand(-1, -1, self.depot_num)
            mask[:, :, :self.depot_num][depot_mask] = 0  # depot is considered unvisited, unless you are AT the depot
            if self.problem_name in ['op','aop']:
                finished = self.at_the_depot & (self.selected_count > 1)
                finished_extend = finished.unsqueeze(-1).expand(-1, -1, self.problem_size)
                mask[:, :, 1:][finished_extend] = float('-inf')

        if self.problem_name in ['op', 'pctsp', 'spctsp', 'aop', 'apctsp', 'aspctsp']:
            newly_finished = self.at_the_depot & (self.selected_count > 1)
        elif self.problem_name in ['tsp', 'atsp', 'pdtsp', 'apdtsp']:
            visited_all = (self.selected_count == (self.problem_size + self.depot_num))
            if visited_all:
                
                # 步数够了的同时, 保证全部节点也同时masked才可以保证是一个合法解
                visited_mask = mask_visited_nodes(self)
                double_check_all_visited = (visited_mask == float('-inf')).all(-1)
                if (~double_check_all_visited).any():
                    raise ValueError("infeasible solution")
                newly_finished = double_check_all_visited
            else:
                newly_finished = visited_all
        else:
            visited_mask = mask_visited_nodes(self)
            depot_mask = ~self.at_the_depot.unsqueeze(-1)
            depot_mask = depot_mask.expand(-1, -1, self.depot_num)
            visited_mask[:, :, :self.depot_num][depot_mask] = 0
            
            # 全部用户节点都被mask,但depot只需有一个即可,证明其回到depot
            customers_all_inf = (visited_mask[:, :, self.depot_num:] == float('-inf')).all(dim=2)
            depot_any_inf = (visited_mask[:, :, :self.depot_num] == float('-inf')).any(dim=2)
            newly_finished = customers_all_inf & depot_any_inf

        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)
        if self.problem_name in ['pctsp','spctsp', 'apctsp', 'aspctsp']:
            num_nodes = self.problem_size + self.depot_num
            allow = (self.collected_prize >= 1.) | (self.selected_count == num_nodes)
            # 将finished的其他action设置为-inf
            finished = self.at_the_depot & (self.selected_count > 1)
            finished_extend = finished.unsqueeze(-1).expand(-1, -1, num_nodes)
            mask[finished_extend] = float('-inf')
            mask[:, :, 0][allow] = 0

        # do not mask depot for finished episode.
        mask[:, :, 0][self.finished] = 0

        return mask

    def get_starting_points(self,actions, num_depots):
        """
        Get which depot (starting point) each action in the sequence starts from
        Example:
        >>> actions = torch.tensor([[1, 10, 2, 0, 3, 30, 21, 2], [2, 15, 20, 1, 25, 30, 0, 1]])
        >>> get_starting_points(actions, 3) -> torch.tensor([[1, 1, 2, 0, 0, 0, 0, 2], [2, 2, 2, 1, 1, 1, 0, 1]])
        """
        # Create mask for numbers < num_depots
        mask = actions < num_depots  # shape: (batch_size, seq_len)
        batch_size, seq_len = actions.shape

        # Compute the cumulative sum of the mask to get segment IDs
        segment_ids = torch.cumsum(mask.long(), dim=1)  # shape: (batch_size, seq_len)

        # Adjust segment IDs for indexing (shift by -1)
        segment_indices = segment_ids - 1

        # Create a mask for valid segment positions
        valid_positions = segment_ids > 0

        # Compute the number of masked elements per batch
        num_values_per_batch = mask.sum(dim=1)  # shape: (batch_size,)
        max_num_values = num_values_per_batch.max().item()

        # Generate batch indices
        batch_indices = (
            torch.arange(batch_size, device=actions.device)
            .unsqueeze(1)
            .expand(batch_size, seq_len)
        )

        # Get indices where mask is True
        masked_indices = torch.where(
            mask,
            torch.cumsum(mask.long(), dim=1) - 1,
            torch.tensor(-1, device=actions.device),
        )
        valid_masked_positions = masked_indices >= 0

        # Gather valid batch and masked indices
        valid_batch_indices = batch_indices[valid_masked_positions]
        valid_masked_indices = masked_indices[valid_masked_positions]
        valid_actions = actions[valid_masked_positions]

        # Initialize padded values tensor
        values_padded = torch.zeros(
            batch_size, max_num_values, dtype=actions.dtype, device=actions.device
        )

        # Fill in the padded values tensor
        values_padded[valid_batch_indices, valid_masked_indices] = valid_actions

        # Initialize the starting_points tensor
        starting_points = torch.zeros_like(actions)

        # Fill in the starting_points tensor using advanced indexing
        starting_points[valid_positions] = values_padded[
            batch_indices[valid_positions], segment_indices[valid_positions]
        ]

        return starting_points


