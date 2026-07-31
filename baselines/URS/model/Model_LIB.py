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

########################################
# NN SUB CLASS / FUNCTIONS
########################################
from typing import Tuple

import math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from problem.ProblemSet import ProblemSet

def get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes



def select_next_node(probs: Tensor, decoding_strategy: str="sampling")-> Tuple[Tensor, Tensor]:
    """
    Design a novel algorithm to select the next node in each step.
    Args:
    probs: Probability distribution over nodes, shape: (batch_size, m).
    decoding_strategy: Decoding strategy to use. Available strategies: ['sampling', 'greedy'], default: 'sampling'.

    Return:
    ID of the next node to visit.
    prob of the selected node.
    """
    assert not torch.isnan(probs).any(), "probs has nan, but it should not have any nans."
    batch_size, pomo_size, problem_size = probs.size()
    if decoding_strategy == "sampling":
        # Check if sampling went OK, can go wrong due to bug on GPU
        # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
        # to fix pytorch.multinomial bug on selecting 0 probability elements
        while True:
            selected = (probs.reshape(batch_size * pomo_size, -1).multinomial(1)
                        .squeeze(dim=1).reshape(batch_size, pomo_size))
            # shape: (batch, pomo)
            prob = torch.gather(probs, dim=-1, index=selected.unsqueeze(-1)).squeeze(dim=-1)
            # shape: (batch, pomo)
            if (prob != 0).all():
                break
        assert prob.size() == (batch_size,pomo_size), f"prob.size(): {prob.size()}. Expected: {(batch_size,pomo_size)}"
        # shape: (batch, n_start)
    elif decoding_strategy == "greedy":
        selected = torch.argmax(probs, dim=-1)
        # (batch_size, pomo)
        prob = None # prob is not needed for greedy decoding
        # shape: (batch, n_start)
    else:
        raise NotImplementedError(f"eval_type: {decoding_strategy} is not implemented!")

    return selected, prob

def distance_normalization(distance_matrix, dist_norm_style):
    # distance_matrix.shape: (batch, n, m)
    batch_size = distance_matrix.size(0)
    if dist_norm_style == 'sep_min_max':
        dist_max = distance_matrix.amax(dim=2, keepdim=True)  # <B, N, 1>
        dist_min = distance_matrix.amin(dim=2, keepdim=True)  # <B, N, 1>
        dist_normed = ((distance_matrix - dist_min) / (dist_max - dist_min + 1e-8))
    elif dist_norm_style == 'sep_max':
        dist_max = distance_matrix.amax(dim=2, keepdim=True)  # <B, N, 1>
        dist_normed = distance_matrix / (dist_max + 1e-8)
    elif dist_norm_style == 'all_min_max':
        dist_max = distance_matrix.amax(dim=(1, 2), keepdim=True)
        dist_min = distance_matrix.amin(dim=(1, 2), keepdim=True)
        # <B, 1, 1>
        assert dist_max.shape == (batch_size, 1, 1)
        assert dist_min.shape == (batch_size, 1, 1)
        dist_normed = ((distance_matrix - dist_min) / (dist_max - dist_min + 1e-8))
    elif dist_norm_style == 'all_max':
        dist_max = distance_matrix.amax(dim=(1, 2), keepdim=True)
        assert dist_max.shape == (batch_size, 1, 1)
        dist_normed = distance_matrix / (dist_max + 1e-8)  # normalize edge features per node
    else:
        assert dist_norm_style == 'nonorm', "Unknown bias style: {}".format(dist_norm_style)
        dist_normed = distance_matrix

    return dist_normed

def adaptation_attention_free_module(q, k, v, adaptation_bias, ninf_mask=None):
    """
    The core code of Adaptation Attention Free Module (https://arxiv.org/pdf/2405.01906).

    Args:
        q: query, shape: (batch, n, embedding_dim)
        k: key, shape: (batch, m, embedding_dim)
        v: value, shape: (batch, m, embedding_dim)
        adaptation_bias: - alpha * log_scale * dist, shape: (batch, n, m)
        ninf_mask: shape: (batch, n, m)

    Return:
        out: shape: (batch, n, embedding_dim)

    Note:
    AAFM may have potential value overflow issues due to the exponential operation. 
    To improve the numerical stability of AAFM in training, we use two operations to prevent overflow:
    1. **log-sum-exp trick:** We compute the maximum value of K matrix and subtract it from the K before applying the exponential function. 
    This can help to prevent overflow while still maintaining the relative differences between the values.
    2. **torch.nan_to_num:** We will check again if there are any infinite or NaN values. If there are, we use "torch.nan_to_num" to replace them with finite numbers.
    For more details, please refer to the official document: https://pytorch.org/docs/1.10/generated/torch.nan_to_num.html.
    """

    sigmoid_q = torch.sigmoid(q)
    # shape: (batch, n, embedding_dim)

    if ninf_mask is not None:
        adaptation_bias = adaptation_bias + ninf_mask

    # stable exp(k) ---
    # logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
    k_max = torch.amax(k, dim=-2, keepdim=True)
    # (batch, 1, embedding_dim)
    exp_k = torch.exp(k - k_max)  # maximum value is exp(0) = 1, avoid overflow

    exp_A = torch.exp(adaptation_bias)

    bias = exp_A @ torch.mul(exp_k, v)
    # shape: (batch, n, embedding_dim)
    a_k = exp_A @ exp_k

    if torch.isinf(bias).any() or torch.isnan(bias).any():
        torch.nan_to_num_(bias)
    if torch.isinf(a_k).any() or torch.isnan(a_k).any():
        torch.nan_to_num_(a_k)

    weighted = bias / (a_k + 1e-8)
    # shape: (batch, n, embedding_dim)

    if torch.isnan(weighted).any() or torch.isnan(weighted).any():
        torch.nan_to_num_(weighted)
    # shape: (batch, n, embedding_dim)

    out = torch.mul(sigmoid_q, weighted)
    # shape: (batch, n, embedding_dim)

    return out

class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))

def unified_node_position_construction(problems, problem_name):
    batch_size = problems.size(0)
    problem_size = problems.size(1)
    position_features = torch.zeros((batch_size, problem_size, 3), device=problems.device)
    if problem_name in ProblemSet.get(name="asymmetric_list"):
        # node projections
        node_random_emb = torch.rand((batch_size, problem_size, 1), device=problems.device)
        position_features[:, :, 0] = node_random_emb[:, :, 0]  # random feature
    else:
        position_features[:, :, 1] = problems[:, :, 0]  # x
        position_features[:, :, 2] = problems[:, :, 1]  # y

    return position_features

def unified_node_attribute_construction(problems, problem_name, demand_max1=True):
    batch_size = problems.size(0)
    problem_size = problems.size(1)
    # Order: [demand0, prize1, penalty2, early_time3, late_time4, service_time5, depot6, pickup7, delivery8, multi route9, open route10]
    attribute_features = torch.zeros((batch_size, problem_size, 11), device=problems.device)
    if problem_name in ['atsp','tsp']:
        pass
    elif problem_name in ['op','aop']:
        attribute_features[:, 0, 6] = 1  # depot node
        attribute_features[:, :, 1] = problems[:, :, 3]  # prize
    elif problem_name in ['pctsp','apctsp']:
        attribute_features[:, 0, 6] = 1  # depot node
        attribute_features[:, :, 1] = problems[:, :, 3]  # prize
        attribute_features[:, :, 2] = problems[:, :, 5]  # penalty
    elif problem_name in ['spctsp','aspctsp']:
        # x,y,real_prize(sto_prize),fake_prize,penalty
        attribute_features[:, 0, 6] = 1  # depot node
        attribute_features[:, :, 1] = problems[:, :, 4]  # fake_prize
        attribute_features[:, :, 2] = problems[:, :, 5]  # penalty
    elif problem_name in ProblemSet.get(included=["pd","tsp"]):
        # follow rl4co, 1~n/2 is pickup,n/2+1~n is delivery
        # https://github.com/ai4co/rl4co/blob/main/rl4co/envs/routing/pdp/env.py
        attribute_features[:, 0, 6] = 1  # depot node
        attribute_features[:, 1:problem_size // 2 + 1, 7] = 1  # pickup node
        attribute_features[:, problem_size // 2 + 1:,  8] = 1  # delivery node
    elif problem_name in ProblemSet.get(included="cvrp", excluded="pd"):
        if demand_max1:
            demand = problems[:, :, 2]  # shape: (batch, problem)
            demand_normed = demand / (demand.amax(dim=-1, keepdim=True) + 1e-8)
        else:
            demand_normed = problems[:, :, 2]  # shape: (batch, problem)
        attribute_features[:, :, 0] = demand_normed  # demand
        if problem_name in ProblemSet.get(included=["md", "cvrp"]):
            attribute_features[:, :3, 6] = 1  # depot node
        else:
            attribute_features[:, 0, 6] = 1  # single depot node
        attribute_features[:, :, 9] = 1  # multi route
        if 'tw' in problem_name:
            tw_norm_factor = problems[0,0,-2]  # depot_tw_end
            service_time = problems[0,-1,-1]   # service_time
            attribute_features[:, :, 3] = problems[:, :, -3] / tw_norm_factor  # early time window
            attribute_features[:, :, 4] = problems[:, :, -2] / tw_norm_factor  # late time window
            attribute_features[:, :, 5] = service_time  # service_time
        if 'b' in problem_name:
            attribute_features[:, :, 7][problems[:, :, 2] < 0] = 1  # pickup node
            attribute_features[:, :, 8][problems[:, :, 2] > 0] = 1  # delivery node
        else:
            attribute_features[:, 1:, 8] = 1  # delivery node
        if 'o' in problem_name:
            attribute_features[:, :, 10] = 1  # open route
    elif problem_name in ProblemSet.get(included=["pd", "cvrp"]):
        if demand_max1:
            demand = problems[:, :, 2] # shape: (batch, problem)
            demand_normed = demand / (demand.amax(dim=-1, keepdim=True) + 1e-8)
        else:
            demand_normed = problems[:, :, 2]  # shape: (batch, problem)
        attribute_features[:, :, 0] = -demand_normed  # demand
        attribute_features[:, 0, 6] = 1  # depot node
        attribute_features[:, 1:problem_size // 2 + 1, 7] = 1  # pickup node
        attribute_features[:, problem_size // 2 + 1:, 8] = 1  # delivery node
        attribute_features[:, :, 9] = 1  # multi route
        if 'o' in problem_name:
            attribute_features[:, :, 10] = 1  # open route

    else:
        raise ValueError(f"Unsupported problem name: {problem_name}")

    return attribute_features
    # shape: (batch, problem, 11)

class Adaptation_Bias_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.W1 = nn.Linear(13, embedding_dim)
        self.W2 = nn.Linear(embedding_dim, 1)

    def forward(self, problem_representation):
        # input.shape: (8,)
        # bias >= 1, because we find it is helpful for obtaining better coverage behavior in our experiments.
        return F.relu(self.W2(self.W1(problem_representation))) + 1 





