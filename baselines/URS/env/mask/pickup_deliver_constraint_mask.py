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

import torch

from .mask_registry import register_mask

@register_mask("pd")
def pickup_deliver_constraint_mask(env) -> torch.Tensor:

    #to_deliver 指出前一半包括depot是p,后一半是d
    batch_size, pomo_size, seq_len = env.selected_node_list.shape
    problem_size = env.problem_size + env.depot_num

    mask = torch.zeros((batch_size, pomo_size, problem_size))

    #deliver最开始是不能被访问的
    mask[~env.to_deliver] = float('-inf')
    # Step1: 找出 visited_nodes <= 50 的位置
    cond = (env.selected_node_list <= problem_size//2) & (env.selected_node_list > 0)# shape: (batch, pomo, seq_length)

    # Step2: 计算需要在 mask 中置零的索引 (visited_nodes + 50)
    target_idx = env.selected_node_list + problem_size//2  # shape: (batch, pomo, seq_length)

    # Step3: 展开 batch,pomo,seq_length，做索引赋值
    b_idx, p_idx, s_idx = torch.where(cond)  # 取出符合条件的索引
    m_idx = target_idx[b_idx, p_idx, s_idx]  # 计算对应 mask 的索引

    # 在 mask 中对应位置置 0
    mask[b_idx, p_idx, m_idx] = 0


    return mask

