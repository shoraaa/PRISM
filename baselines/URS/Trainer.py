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

import os
import random

import numpy as np
import torch
from logging import getLogger
from env.UniVRPEnv import UniVRPEnv

from model.Model import Model

from problem.ProblemRep import get_problem_representations

from data.DataFinder import DataFinder
from data.DataReader import get_saved_data

from torch.optim import AdamW, Adam

from utils.utils import *
from env.mask.mask_registry import mask_registry


class Trainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        
        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()
        
        self.train_problem_list = trainer_params['train_problem_list']
        if trainer_params['validation_problem_list'] is None:
            self.validation_problem_list = trainer_params['train_problem_list']
        else:
            self.validation_problem_list = trainer_params['validation_problem_list']

        self.validation_saved_folder = os.path.join(self.result_folder, "validation_curves")
        os.makedirs(self.validation_saved_folder, exist_ok=True)

        self.logger.info("{} problems for training: {}".format(len(self.train_problem_list), self.train_problem_list))
        self.logger.info("{} problems for validation: {}".format(len(self.validation_problem_list), self.validation_problem_list))

        self.problem_representation_set = get_problem_representations()
        unknown_problems = [p for p in self.validation_problem_list if p not in self.problem_representation_set]
        if unknown_problems:
            raise ValueError(f"Unknown problem names: {unknown_problems}")

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        self.device = device

        # Main Components
        self.model = Model(**self.model_params)
        if self.optimizer_params['optimizer_type'] == 'AdamW':
            self.optimizer = AdamW(self.model.parameters(), **self.optimizer_params['optimizer'])
        elif self.optimizer_params['optimizer_type'] == 'Adam':
            self.optimizer = Adam(self.model.parameters(), **self.optimizer_params['optimizer'])
        else:
            raise NotImplementedError(f"optimizer_type: {self.optimizer_params['optimizer_type']} is not implemented!")

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info('Saved Model Loaded !!')
            self.logger.info("Model loaded from: {0}".format(checkpoint_fullname))

        # utility
        self.time_estimator = TimeEstimator()
        
        # convert problem representations to tensors, which will be used for conditioning the model during training and validation
        self.problem_representation_tensor = {
            name: torch.tensor(rep, dtype=torch.float32, device=self.device)
            for name, rep in self.problem_representation_set.items()
        }

        # load the data for validation, including saved_data and oracle_score, for each problem type and scale
        ##############################################################################################################
        data_finder = DataFinder(data_dir=env_params['data_dir'])

        validation_scale = trainer_params['validation_scale'] if trainer_params['validation_scale'] is not None else env_params['problem_size']
        validation_episodes = trainer_params['validation_episodes'] if trainer_params['validation_episodes'] is not None else 1000
        self.validation_batch_size = trainer_params['validation_batch_size'] if trainer_params['validation_batch_size'] is not None else validation_episodes
        
        # load the data for validation, including saved_data and oracle_score, for each problem type and scale
        self.saved_data = {}
        for problem_name in self.validation_problem_list:
            data_info = data_finder.get(problem_name, validation_scale)
            if data_info is None:
                raise ValueError(f"No validation data found for problem: {problem_name} at scale: {validation_scale}.")
            saved_data, oracle_score = get_saved_data(
                filename=data_info["data_path"],
                problem_name=problem_name,
                total_episodes=validation_episodes,
                device=self.device, 
                solution_name=data_info["solution_path"],
            )

            self.saved_data[problem_name] = {
                "saved_data": saved_data,
                "oracle_score": oracle_score,
                "scale": validation_scale,
                "episodes": validation_episodes
            }

            self.logger.info('Successfully load {0:4d} {1:16s} ==> oracle score: {2:7.4f}'.format(
                validation_episodes, problem_name.upper()+str(validation_scale), oracle_score))
        
        self.best_gap = float("inf") # set best gap to inf at the beginning, which will be updated during training

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        self.lr_decay_epoch = self.optimizer_params['lr_decay_epoch']
        

        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('========================================================================')

            #########################################################################
            # Train
            #########################################################################
            if epoch in self.lr_decay_epoch:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = param_group["lr"] * 0.1 # 1e-5

            self.logger.info('Epoch {:4d}: Current learning rate: {}'.format(epoch, self.optimizer.param_groups[0]['lr']))

            train_loss = self._train_one_epoch(epoch)

            self.result_log.append('train_loss', epoch, train_loss)

            #########################################################################
            # Validation
            #########################################################################
            gap_sum = 0.0
            for problem_name in self.validation_problem_list:
                cur_saved_data = self.saved_data[problem_name]["saved_data"]
                cur_oracle_score = self.saved_data[problem_name]["oracle_score"]
                gap = self._validation_one_problem(problem_name, cur_saved_data, cur_oracle_score, epoch,
                                            episodes=self.saved_data[problem_name]["episodes"],
                                            problem_size=self.saved_data[problem_name]["scale"],
                                            batch_size=self.validation_batch_size
                                            )
                gap_sum = gap_sum + gap
            avg_gap = gap_sum / len(self.validation_problem_list)
            self.logger.info("Epoch {:4d}: Average gap: {:.4f}%".format(epoch, avg_gap))
            if avg_gap < self.best_gap:
                self.best_gap = avg_gap
                self.logger.info("Epoch {:4d}: New best average gap: {:.4f}%, saving best checkpoint".format(epoch, self.best_gap))
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{0}/best_checkpoint.pt'.format(self.result_folder))

            #########################################################################
            # Logs & Checkpoint
            #########################################################################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:4d}/{:4d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']

            if epoch > 1:  # save latest images, every epoch
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{0}/checkpoint_{1}.pt'.format(self.result_folder, epoch))

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        loss = AverageMeter()

        batches_per_epoch =  self.trainer_params['batches_per_epoch']
        loop_cnt = 0 # used for counting the number of batches

        while loop_cnt < batches_per_epoch:
            problem_name = random.choice(self.train_problem_list)

            true_problem_size = self.env_params['problem_size']
            true_batch_size = self.trainer_params['batch_size']

            if 'vrp' in problem_name:
                capacity = self.env_params.get('capacity', 50.0)
            else:
                capacity = None
            kwargs = {
                "capacity": capacity,
            }
            avg_score, avg_loss = self._train_one_batch(problem_name,true_problem_size,true_batch_size,kwargs)

            loss.update(avg_loss, true_batch_size)

            loop_cnt += 1
            if loop_cnt <= 5 or loop_cnt % 200 == 0:
                self.logger.info('Epoch {:4d}: Trained batches {:4d}/{:4d}({:5.1f}%): Accumulated loss: {:7.4f}. Current batch: problem: {:8s}, score: {:7.4f}, single-step loss: {:7.4f}'
                                 .format(epoch, loop_cnt, batches_per_epoch, 100. * loop_cnt / batches_per_epoch, loss.avg, 
                                         problem_name, avg_score, avg_loss))

        # Log Once, for each epoch
        self.logger.info('Epoch {:4d}: Train ({:3.0f}%)  Loss: {:.4f}'
                         .format(epoch, 100. * loop_cnt / batches_per_epoch, loss.avg))

        return loss.avg

    def _train_one_batch(self, problem_name,problem_size,batch_size,kwargs):

        # Prep
        ###############################################
        self.model.train()
        self.model.set_decoder_type("sampling")
        # re-instantiate environment for each batch to avoid potential influence of different problem types
        self.env = UniVRPEnv() 

        # Build combined mask function for the specific problem
        mask_fn = mask_registry.build_combined_mask(problem_name)
        problem_representation = self.problem_representation_tensor[problem_name]

        pomo_size = problem_size
        self.env.load_problems(batch_size=batch_size,
                               problem_name=problem_name,
                               problem_size=problem_size,
                               pomo_size=pomo_size,
                               device=self.device,
                               **kwargs)

        pomo_size = self.env.pomo_size

        reset_state, _, _ = self.env.reset()
        self.model.decoder.assign(problem_representation)
        self.model.pre_forward(reset_state, problem_name, problem_representation)
        prob_list = torch.zeros(size=(batch_size, pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            cur_dist = self.env.get_local_feature()
            selected, prob = self.model(state,cur_dist)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected,mask_fn=mask_fn)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        ###############################################
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)

        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        if problem_name in ["op"]:
            score_mean = max_pomo_reward.float().mean()
        else:
            score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.optimizer.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        return score_mean.item(), loss_mean.item()

    def _validation_one_problem(self, problem_name, dataset, optimal_score, epoch,episodes,problem_size,batch_size=None):

        self.model.eval()
        self.model.set_decoder_type("greedy")
        self.env = UniVRPEnv()

        # Build combined mask function for the specific problem
        mask_fn = mask_registry.build_combined_mask(problem_name)
        problem_representation = self.problem_representation_tensor[problem_name]

        tested_episodes = 0
        if batch_size is None:
            batch_size = episodes

        results = torch.zeros(size=(episodes,), dtype=torch.float32, device=self.device)

        with torch.inference_mode():
            while tested_episodes < episodes:
                remaining_episodes = episodes - tested_episodes
                batch_size = min(batch_size, remaining_episodes)
                self.env.load_problems(batch_size=batch_size,
                                       problem_name=problem_name,
                                       problem_size=problem_size,
                                       validation_data=dataset,
                                       start=tested_episodes,
                                       device=self.device)
                reset_state, _, _ = self.env.reset()

                self.model.decoder.assign(problem_representation)
                self.model.pre_forward(reset_state, problem_name, problem_representation)

                # POMO Rollout
                ###############################################
                state, reward, done = self.env.pre_step()
                while not done:
                    cur_dist = self.env.get_local_feature()
                    selected, _ = self.model(state, cur_dist)
                    # shape: (batch, pomo)
                    state, reward, done = self.env.step(selected,mask_fn=mask_fn)

                # Return
                ###############################################
                max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
                # shape: (batch,)
                results[tested_episodes:tested_episodes + batch_size] = max_pomo_reward
                tested_episodes += batch_size
        if problem_name in ["op"]:
            avg_score = results.mean()
            gap = ((optimal_score - avg_score) * 100 / optimal_score).item()
        else:
            avg_score = -results.mean() # negative sign to make positive value
            gap = ((avg_score - optimal_score) * 100 / optimal_score).item()
        avg_score_item = avg_score.item()
        # Logs
        ##################################################
        self.result_log.append(f'{problem_name}_eval_{problem_size}', epoch, avg_score_item)
        self.result_log.append(f'{problem_name}_gap_{problem_size}', epoch, gap)
        
        self.logger.info('Epoch {0:4d}: {1:4d} {2:16s} ==> Score: {3:7.4f}, Gap: {4:7.4f}%'.format(
                epoch, tested_episodes, problem_name.upper()+str(problem_size), avg_score_item, gap))

        if epoch > 1:
            image_prefix = '{}/latest'.format(self.validation_saved_folder)
            util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                           self.result_log, labels=[f'{problem_name}_eval_{problem_size}'])
            util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                           self.result_log, labels=[f'{problem_name}_gap_{problem_size}'])

        return gap

