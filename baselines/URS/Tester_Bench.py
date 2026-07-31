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

import time

import numpy as np
import torch
from tqdm import tqdm
import os
from logging import getLogger

from model.Model import Model

from env.UniVRPEnv import UniVRPEnv
from env.mask.mask_registry import mask_registry

from problem.ProblemRep import get_problem_representations
from data.LibReader import TSPLIBReader, CVRPLIBReader, tsplib_cost

from utils.utils import *

class Tester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='tester')
        self.result_folder = get_result_folder()
        
        self.test_problem_list = env_params['test_problem_list']
        assert len(self.test_problem_list) == 1, \
            "For benchmark evaluation, only one specific problem is allowed for testing, but got: {}".format(self.test_problem_list)
        self.logger.info("{0} problems for testing: {1}".format(len(self.test_problem_list), self.test_problem_list))

        self.problem_representation_set = get_problem_representations()
        unknown_problems = [p for p in self.test_problem_list if p not in self.problem_representation_set]
        if unknown_problems:
            raise ValueError(f"Unknown problem names: {unknown_problems}")

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.problem_name = self.test_problem_list[0]
        self.env = UniVRPEnv()
        self.mask_fn = mask_registry.build_combined_mask(self.problem_name)
        self.model = Model(**self.model_params).to(self.device)

        # Restore
        checkpoint_fullname = tester_params['model_load']
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info("Model loaded successfully!!!")
        self.logger.info("Model loaded from: {}".format(checkpoint_fullname))
        
        total = sum([param.nelement() for param in self.model.parameters()])
        self.logger.info("Number of parameters: %.2fM" % (total / 1e6))

        # convert problem representations to tensors, which will be used for conditioning the model during training and validation
        self.problem_representation_tensor = {
            name: torch.tensor(rep, dtype=torch.float32, device=self.device)
            for name, rep in self.problem_representation_set.items()
        }
        
    def get_sorted_instances(self, data_dir):
        scale_range = self.env_params['scale_range_lib']
        self.logger.info("Reading instances from data_dir: {}, with scale_range: {}".format(data_dir, scale_range))
        self.data = {}
        num_sample = 0
        if self.test_problem_list[0] == "tsp":
            for root, dirs, files in os.walk(data_dir):
                for f in files:
                    if f.endswith(".tsp"):
                        name, problem_size, locs, edge_weight_type = TSPLIBReader(os.path.join(root, f))
                        if name is None:
                            continue
                        if scale_range is not None and not (scale_range[0] <= problem_size < scale_range[1]):
                            continue
                        optimal = tsplib_cost.get(name, None)
                        if optimal is None:
                            raise ValueError(f"Optimal value for TSP instance {name} not found in tsplib_cost dict.")
                        self.data[name] = {
                                           "problem_size": problem_size, 
                                           "locations": torch.as_tensor(locs, dtype=torch.float32),
                                           "edge_weight_type": edge_weight_type,
                                           "demand": torch.zeros(problem_size, dtype=torch.float32), # dummy demand for TSP
                                           "capacity": 1.0, # dummy capacity for TSP, avoid potential division by zero when normalizing demand
                                           "optimal": optimal
                                           }
                        num_sample += 1
        elif self.test_problem_list[0] == "cvrp":
            for root, dirs, files in os.walk(data_dir):
                for f in files:
                    if f.endswith(".vrp"):
                        name, problem_size, locs, demand, capacity, optimal, edge_weight_type = CVRPLIBReader(
                            os.path.join(root, f)
                        )
                        if name is None:
                            continue
                        if scale_range is not None and not (scale_range[0] <= problem_size < scale_range[1]):
                            continue
                        self.data[name] = {
                            "problem_size": problem_size,
                            "locations": torch.as_tensor(locs, dtype=torch.float32),
                            "edge_weight_type": edge_weight_type,
                            "demand": torch.as_tensor(demand, dtype=torch.float32),
                            "capacity": capacity,
                            "optimal": optimal
                        }
                        num_sample += 1
        else:
            raise ValueError(f"Unsupported problem type: {self.test_problem_list[0]}")
        
        if num_sample == 0:
                raise ValueError(f"No TSP files found in {data_dir} within the specified scale range {scale_range}")
        else:
            self.data = dict(sorted(self.data.items(), key=lambda item: item[1]["problem_size"]))

        self.logger.info("The instances are sorted according to their problem size, and the total number of instances is {}".format(len(self.data)))
        
    def run(self):
        self.get_sorted_instances(self.env_params['data_dir'])
        result_dict = {}
        result_dict["instances"] = []
        result_dict['optimal'] = []
        result_dict['problem_size'] = []
        result_dict['no_aug_score'] = []
        result_dict['aug_score'] = []
        result_dict['no_aug_gap'] = []
        result_dict['aug_gap'] = []
        result_dict['time'] = []

        start_time = time.time()
        solved_count = 0
        for name, instance_info in self.data.items():
            
            instance_start_time = time.time()
            optimal = instance_info["optimal"]  # optimal value of the instance
            problem_size = instance_info["problem_size"]  # node number of the instance,not including the depot
            edge_weight_type = instance_info["edge_weight_type"]  # edge weight type of the instance
            capacity = instance_info["capacity"]  # capacity,shape:(1,)
            node_coord = instance_info["locations"].unsqueeze(0) # node coordinates, including the depot
            # shape:(1,problem_size+1,2)

            instance_info['original_xy_lib'] = node_coord
            instance_info['normalized_demand'] = instance_info["demand"].unsqueeze(0) / float(capacity) # shape:(problem_size+1,)
            # shape:(1,problem_size)
            instance_info['optimal'] = optimal
            instance_info['pomo_size'] = problem_size


            # normalize coordinates to [0,1] 
            ################################################################
            xy_max = torch.max(node_coord, dim=1, keepdim=True).values
            xy_min = torch.min(node_coord, dim=1, keepdim=True).values
            # shape: (1, 1, 2)
            ratio = torch.max((xy_max - xy_min), dim=-1, keepdim=True).values
            ratio[ratio == 0] = 1
            # shape: (1, 1, 1)
            normalized_xy = (node_coord - xy_min) / ratio.expand(-1, 1, 2)
            # shape: (1, problem_size+1,2)
            instance_info["normalized_xy"] = normalized_xy

            self.logger.info("===============================================================")
            self.logger.info("Instance name: {0}, problem_size: {1}, edge_weight_type: {2}".format(name, problem_size, edge_weight_type))
            try:
                score, aug_score = self._test_one_instance(batch_size=1,instance_info=instance_info)
                solved_count += 1
                no_aug_gap = (score - optimal) * 100 / optimal
                aug_gap = (aug_score - optimal) * 100 / optimal
                instance_end_time = time.time()
                during_instance_time = instance_end_time - instance_start_time

                self.logger.info("Instance name: {}, optimal score: {:.4f}".format(name, optimal))
                self.logger.info("No aug score:{:.3f}, No aug gap:{:.3f}%".format(score, no_aug_gap))
                self.logger.info("Aug score:{:.3f}, Aug gap:{:.3f}%".format(aug_score, aug_gap))
                self.logger.info("Time: {:.2f}s, {:.2f}m".format(during_instance_time, during_instance_time / 60))
                self.logger.info("Solved {}/{} instances.".format(solved_count, len(self.data)))
            except Exception as e:
                self.logger.info("Error occurred in instance {0}, dimension: {1}, skip it!".format(name, problem_size))
                self.logger.info("Error message: {0}".format(e))
                continue
            
            ############################
            # Logs
            ############################
            result_dict["instances"].append(name)
            result_dict['optimal'].append(optimal)
            result_dict['problem_size'].append(problem_size)
            result_dict['no_aug_score'].append(score)
            result_dict['aug_score'].append(aug_score)
            result_dict['no_aug_gap'].append(no_aug_gap)
            result_dict['aug_gap'].append(aug_gap)
            result_dict['time'].append(during_instance_time)

        end_time = time.time()
        assert solved_count > 0, "No instance is solved successfully."
        self.logger.info(" *** Test Done *** ")
        self.logger.info("===============================================================")
        if self.tester_params["detailed_log"]:
            self.logger.info("instance: {0}".format(result_dict['instances']))
            self.logger.info("optimal: {0}".format(result_dict['optimal']))
            self.logger.info("problem_size: {0}".format(result_dict['problem_size']))
            self.logger.info("no_aug_score: {0}".format(result_dict['no_aug_score']))
            self.logger.info("aug_score: {0}".format(result_dict['aug_score']))
            self.logger.info("no_aug_gap: {0}".format(result_dict['no_aug_gap']))
            self.logger.info("aug_gap: {0}".format(result_dict['aug_gap']))
            self.logger.info("===============================================================")

        avg_all_no_aug_gap = np.mean(result_dict['no_aug_gap'])  # avg of all instances gap (no aug)
        avg_all_aug_gap = np.mean(result_dict['aug_gap'])  # avg of all instances gap (aug)
        assert solved_count == len(result_dict['instances'])
        max_dimension = max(result_dict['problem_size'])
        min_dimension = min(result_dict['problem_size'])
        self.logger.info("Solved {0}/{1} instances, with dimension range: [{2}, {3}] ==> avg gap(no aug): {4:.3f}%, avg gap(aug): {5:.3f}%".format(
            solved_count, len(self.data), min_dimension, max_dimension, avg_all_no_aug_gap, avg_all_aug_gap))
        self.logger.info("Avg time per instance: {0:.2f}s".format((end_time - start_time) / solved_count))

    def _test_one_instance(self, batch_size, instance_info):

        # Augmentation
        ###############################################
        problem_size = instance_info['problem_size']
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor'][0] if "a" not in self.problem_name else self.tester_params['aug_factor'][1]
            pomo_size = min(problem_size, 5000) # due to memory issue, we limit the pomo size to 5000 when adopting augmentation
        else:
            aug_factor = 1
            pomo_size = problem_size
        
        problem_representation = self.problem_representation_tensor[self.problem_name]

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():

            self.env.load_problems(batch_size=batch_size, 
                                   problem_name=self.problem_name,
                                   problem_size=problem_size, 
                                   pomo_size=pomo_size,
                                   lib_data=instance_info,
                                   aug_factor=aug_factor, 
                                   device=self.device)
            
            reset_state, _, _ = self.env.reset()

            self.model.decoder.assign(problem_representation)
            self.model.pre_forward(reset_state, self.problem_name, problem_representation)

            # POMO Rollout
            ###############################################
            state, reward, done = self.env.pre_step()
            with tqdm(total=0) as pbar:
                while not done:
                    cur_dist = self.env.get_local_feature()
                    selected, _ = self.model(state, cur_dist)
                    # shape: (batch, pomo)
                    state, reward, done = self.env.step(selected, mask_fn=self.mask_fn)
                    pbar.total += 1
                    pbar.update(1)

        # Return
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

        return no_aug_score.item(), aug_score.item()
