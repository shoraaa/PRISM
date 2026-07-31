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
 
import argparse
import torch

import pytz
from datetime import datetime

import logging
from utils.utils import *
from problem.ProblemSet import ProblemSet

from args import obtain_all_hyperparameters

def _print_config(logger, CUDA_DEVICE_NUM=0):
    USE_CUDA = True if CUDA_DEVICE_NUM >=0 and torch.cuda.is_available() else False
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a unified model for multiple VRP variants.")
    obtain_all_hyperparameters(parser)
    args = parser.parse_args()

    args.train_problem_list = ProblemSet.get(name="train_problem_list")
    if args.add_training_problems is not None:
        combined_list = args.train_problem_list + args.add_training_problems
        args.train_problem_list = sorted(list(dict.fromkeys(combined_list)), key=len)

    validation_problems_combined = args.train_problem_list + ProblemSet.get(name=args.validation_problem_set)
    args.validation_problem_list = sorted(list(dict.fromkeys(validation_problems_combined)), key=len)
    
    # args.train_problem_list = ['cvrp'] # for debugging
    # args.validation_problem_list = args.train_problem_list[:]

    env_params = {
        'problem_size': args.problem_size,
        'capacity': args.capacity,
        'data_dir': args.data_dir,
    }

    model_params = {
        'embedding_dim': args.embedding_dim,
        'encoder_layer_num': args.encoder_layer_num,
        'ff_hidden_dim': args.ff_hidden_dim,
        'logit_clipping': args.logit_clipping,
        'demand_max1': not args.no_demand_max1,
        'eval_type': args.eval_type,
    }

    optimizer_params = {
        'optimizer_type': args.optimizer_type, 
        'optimizer': {
            'lr': args.optimizer_lr,
            'weight_decay': args.weight_decay,
        },
        'lr_decay_epoch': args.lr_decay_epoch,
    }

    trainer_params = {
        'use_cuda': True if args.cuda >= 0 and torch.cuda.is_available() else False,
        'cuda_device_num': args.cuda,
        'epochs': args.training_epochs,
        'batches_per_epoch': args.batches_per_epoch,
        'batch_size': args.batch_size,
        'validation_scale': args.validation_scale,
        'validation_episodes': args.validation_episodes,
        'validation_batch_size': args.validation_batch_size,
        'train_problem_list': args.train_problem_list,
        'validation_problem_list': args.validation_problem_list,
        'logging': {
            'model_save_interval': args.model_save_interval,
            'log_image_params_1': {
                'json_foldername': 'log_image_style',
                'filename': 'style_score.json'
            },
            'log_image_params_2': {
                'json_foldername': 'log_image_style',
                'filename': 'style_loss.json'
            },
        },
        'model_load': {
            'enable': False,  # enable loading pre-trained model
            #'path': '',  # directory path of pre-trained model and log files saved.
            #'epoch': ,  # epoch version of pre-trained model to load.
        },
    }

    process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))

    lr_str = str(args.optimizer_lr).replace('0.', '')
    demand_max_str = 'demand_max1' if model_params['demand_max1'] else 'no_demand_max1'
    train_tasks_len = len(args.train_problem_list)
    validation_tasks_len = len(args.validation_problem_list)
    file_str = f"{demand_max_str}_T{train_tasks_len}V{validation_tasks_len}_epoch{args.training_epochs}_bs{args.batch_size}_batch{args.batches_per_epoch}"

    logger_params = {
        'log_file': {
            'desc': file_str,
            'filename': 'run.log',
            'filepath': f'./result_train/{process_start_time.strftime("%Y-%m-%d")}/' + 
                        process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'
        }
    }
    
    # main
    ##########################################################################################
    seed_everything(args.seed)
    create_logger(**logger_params)
    
    logger = logging.getLogger('root')
    print_startup(args=args, 
                  logger=logger, 
                  result_folder=get_result_folder(), 
                  log_filename=logger_params['log_file']['filename'],
                  phase="training")
    
    from Trainer import Trainer 
    
    _print_config(logger, CUDA_DEVICE_NUM=args.cuda)
    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)
    copy_all_src(trainer.result_folder)
    trainer.run()
