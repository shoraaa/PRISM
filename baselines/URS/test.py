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

def _print_config(CUDA_DEVICE_NUM=0):
    USE_CUDA = True if CUDA_DEVICE_NUM >=0 and torch.cuda.is_available() else False
    logger = logging.getLogger('root')
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

def set_problem_list(args):
    all_supported_problems = ProblemSet.get_all_problem_names()
    problem_set = args.problem_set.strip().lower() if args.problem_set is not None else ""

    if not problem_set:
        raise ValueError(
            "--problem_set cannot be empty. Supported formats: a ProblemSet name ending "
            "with '_list', comma-separated problem names like 'tsp,cvrp,pdcvrp', "
            "or benchmark aliases 'tsplib'/'cvrplib'."
        )
        
    is_benchmark = False
    if problem_set in ["tsplib", "cvrplib"]:
        test_problem_list = [problem_set.replace("lib", "")]  # convert to 'tsplib' -> 'tsp', 'cvrplib' -> 'cvrp'
        log_flag = problem_set
        is_benchmark = True
    elif problem_set.endswith('_list'):
        test_problem_list = ProblemSet.get(name=problem_set)
        log_flag = problem_set
    else:
        test_problem_list = [problem.strip() for problem in problem_set.split(",") if problem.strip()]
        if not test_problem_list:
            raise ValueError(f"--problem_set: {problem_set} does not contain any valid problem names.")
        log_flag = "_".join(test_problem_list)

        invalid_problems = [
            problem for problem in test_problem_list
            if problem not in all_supported_problems
        ]
        if invalid_problems:
            raise ValueError(f"Invalid problem name(s) in --problem_set: {invalid_problems}.")
    
    return test_problem_list, log_flag, is_benchmark

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Solving multiple VRP variants using our unified model.")
    obtain_all_hyperparameters(parser)
    args = parser.parse_args()
    
    if len(args.test_episodes) != len(args.test_scale_list):
        raise ValueError(f"--test_episodes length must match --test_scale_list length: {len(args.test_episodes)} != {len(args.test_scale_list)}")
    test_problem_list, log_flag, is_benchmark = set_problem_list(args)

    env_params = {
        'test_problem_list': test_problem_list,
        'test_scale_list': args.test_scale_list,
        'data_dir': args.data_dir,
        'scale_range_lib': args.scale_range_lib,
    }

    model_params = {
        'embedding_dim': args.embedding_dim,
        'encoder_layer_num': args.encoder_layer_num,
        'ff_hidden_dim': args.ff_hidden_dim,
        'logit_clipping': args.logit_clipping,
        'demand_max1': not args.no_demand_max1,
        'eval_type': args.eval_type,
    }

    tester_params = {
        'use_cuda': True if args.cuda >= 0 and torch.cuda.is_available() else False,
        'cuda_device_num': args.cuda,
        'model_load': args.model_load,
        'test_episodes': args.test_episodes,
        'augmentation_enable':  not args.disable_aug,
        'aug_factor': args.aug_factor,
        'test_batch_size_small': args.test_batch_size_small, 
        'test_batch_size_large': args.test_batch_size_large,
        'summary_problems_per_row': args.summary_problems_per_row,
        'detailed_log': args.detailed_log,
        }

    process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))
    is_aug = "aug" if tester_params['augmentation_enable'] else "no_aug"
    model = tester_params['model_load'].split('/')[-1].replace('.pt', '')
    test_problem_len = len(test_problem_list)
    if is_benchmark:
        scale_range_str = f"range{args.scale_range_lib[0]}_{args.scale_range_lib[1]}"
    else:
        scale_range_str = f"range{min(args.test_scale_list)}_{max(args.test_scale_list)}"
    
    file_str = f"{is_aug}_{log_flag}_{model}_{test_problem_len}tasks_{scale_range_str}"

    logger_params = {
        'log_file': {
            'desc': file_str,
            'filename': 'run.log',
            'filepath': f'./result_test/{log_flag}_{test_problem_len}tasks/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'
        }
    }

    ##########################################################################################
    # main
    seed_everything(args.seed)
    create_logger(**logger_params)
    logger = logging.getLogger('root')
    
    print_startup(args=args, 
                  logger=logger, 
                  result_folder=get_result_folder(), 
                  log_filename=logger_params['log_file']['filename'],
                  phase="testing")
    
    if is_benchmark:
        from Tester_Bench import Tester
    else:
        from Tester import Tester
        
    _print_config(CUDA_DEVICE_NUM=args.cuda)
    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)
    copy_all_src(tester.result_folder)
    tester.run()
