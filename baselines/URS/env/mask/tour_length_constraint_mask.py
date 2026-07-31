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
@register_mask("op")
def tour_length_constraint_mask(env) -> torch.Tensor:

    num_nodes = env.problem_size + env.depot_num
    current_node_to_all_dist = env.dist.gather(1, env.current_node[:, :, None].expand(-1, -1, num_nodes))
    # 其他点的距离回到depot的距离
    current_all_to_depot_dist = env.dist.gather(2, env.current_depot[:, None, :].expand(-1, num_nodes, -1))
    current_all_to_depot_dist = current_all_to_depot_dist.permute(0, 2, 1)

    dist_via_node_to_depot = current_node_to_all_dist + current_all_to_depot_dist  # [batch, pomo, num_nodes]

    # 长度约束 mask: 超过 length_limit 的被屏蔽
    length_mask = dist_via_node_to_depot  > env.tour_maxlength.unsqueeze(-1)   # [batch, pomo, num_nodes]


    is_illegal = length_mask
    neg_inf = torch.tensor(float('-inf'), dtype=current_node_to_all_dist.dtype)
    mask = torch.where(is_illegal, neg_inf, torch.zeros_like(current_node_to_all_dist))

    return mask
