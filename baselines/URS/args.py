'''
This module defines common argument parsing functionality for the project.

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

def add_common_arguments(parser):
    """common parameters"""
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device number to use")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for reproducibility")
    # simply set data_dir to './data' and the project will recursively search for all sub-folders under './data'.
    parser.add_argument("--data_dir", type=str, default="./dataset", 
                        help="Directory where the data is stored, used for both validation and testing.")
    # generalist model: unified_checkpoint_500.pt
    # specialist models: {problem_name}_checkpoint_300.pt, e.g., tsp_checkpoint_300.pt
    parser.add_argument("--model_load", type=str, default="./pretrained/unified_checkpoint_500.pt", 
                        help="Path to the pre-trained model for testing.")

def add_inference_arguments(parser):
    """inference/test dataset parameters"""
    parser.add_argument("--problem_set", type=str, default="all_evaluated_list",
                        help="Problem set for testing. Supports inputs include: " + 
                        "(1) prepared ProblemSet names like 'vrpmix_list', 'zero_shot_list', 'all_evaluated_list', " +
                        "(2) comma-separated specific problems like 'tsp,cvrp,pdcvrp', "+
                        "(3) benchmark aliases 'tsplib' or 'cvrplib'.")
    parser.add_argument("--test_scale_list", type=int, nargs='+', default=[100,1000,2000,3000,4000,5000], help="Problem scales for testing.") 
    parser.add_argument("--test_episodes", type=int, nargs='+', default=[1000, 16, 16, 16, 16, 16], 
                        help="Number of test episodes for each scale. Must match --test_scale_list length.")
    parser.add_argument("--scale_range_lib", type=int, nargs=2, default=[3000, 7001], help="Scale range [min, max) for benchmark evaluation.")
    parser.add_argument("--test_batch_size_small", type=int, nargs=4, default=[1000, 125, 50, 8], 
                        metavar=("SYM", "SYM_MD", "ASYM", "ASYM_MD"), 
                        help="Batch sizes for scale 100: symmetric, symmetric_md, asymmetric, asymmetric_md.")
    parser.add_argument("--test_batch_size_large", type=int, nargs='+', default=[16, 8, 2, 1, 1],
                        help="Batch sizes for large scales: 1000, 2000, 3000, 4000, 5000.")    
    parser.add_argument("--disable_aug", action="store_true", help="Disable instance augmentation during testing")
    parser.add_argument("--aug_factor", type=int, nargs='+', default=[8, 128], metavar=("SYM", "ASYM"),
                        help="Augmentation factor for testing: symmetric and asymmetric problems.")
    parser.add_argument("--summary_problems_per_row", type=int, default=6,
                        help="Number of problems shown per row in the test summary markdown table.")
    parser.add_argument("--detailed_log", action="store_true", help="Whether to log detailed results for each instance in benchmark testing.")

def add_training_arguments(parser):
    """training dataset parameters"""
    parser.add_argument("--training_epochs", type=int, default=500, help="Total epochs for training")
    parser.add_argument("--lr_decay_epoch", type=int, nargs='+', default=[451],
                        help="Epochs at which to decay the learning rate")
    parser.add_argument("--batches_per_epoch", type=int, default=2000, help="Steps per epoch")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training in stage 1")
    parser.add_argument("--model_save_interval", type=int, default=1, help="Interval (in epochs) for saving the model during training")
    parser.add_argument("--add_training_problems", type=str, nargs='+', default=None,
                        help="Additional training problems to include, if None, only the predefined training problems are used.")

def add_validation_arguments(parser):
    """validation dataset parameters"""
    parser.add_argument("--validation_scale", type=int, default=100,
                        help="Validation problem scale. None means use --problem_size")
    parser.add_argument("--validation_episodes", type=int, default=1000,
                        help="Number of validation episodes per problem")
    parser.add_argument("--validation_batch_size", type=int, default=1000,
                        help="Validation batch size. None means validate all episodes in one batch")
    parser.add_argument("--validation_problem_set", type=str, default="vrpmix_list",
                        help="Problem set for validation. Supports prepared ProblemSet names like 'vrpmix_list', 'train_problem_list', etc.")

def add_environment_arguments(parser):
    """environment parameters"""
    parser.add_argument("--problem_size", type=int, default=100, help="problem size for training")
    parser.add_argument("--capacity", type=int, default=50, help="Vehicle capacity for training")
    
def add_model_arguments(parser):
    """model parameters"""
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension for the model")
    parser.add_argument("--encoder_layer_num", type=int, default=12, help="Number of encoder layers in the model")
    parser.add_argument("--ff_hidden_dim", type=int, default=512,
                        help="Hidden dimension for feed-forward layer in the model")
    parser.add_argument("--logit_clipping", type=float, default=50, help="Logit clipping value for the model")
    parser.add_argument("--eval_type", type=str, default="greedy", choices=['sampling', 'greedy'], 
                        help="Evaluation type for the model. During training, we force it is sampling.")
    parser.add_argument("--no_demand_max1", action="store_true", help="Do not normalize demand to a maximum value of 1")

def add_optimizer_arguments(parser):
    """optimizer parameters"""
    parser.add_argument("--optimizer_type", type=str, default="AdamW", help="Optimizer type for the model",
                        choices=['AdamW', 'Adam'])
    parser.add_argument("--optimizer_lr", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay for the optimizer")


def obtain_all_hyperparameters(parser):
    """
    Integrate all sub-functions and configure all hyperparameters for the project.
    """
    add_common_arguments(parser)
    add_environment_arguments(parser)
    add_model_arguments(parser)
    add_optimizer_arguments(parser)
    add_training_arguments(parser)
    add_validation_arguments(parser)
    add_inference_arguments(parser)
    return parser
    
