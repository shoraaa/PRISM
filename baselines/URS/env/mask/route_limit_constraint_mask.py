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
@register_mask("l")
def route_limit_constraint_mask(env) -> torch.Tensor:

    num_nodes = env.problem_size + env.depot_num
    round_error_epsilon = 0.00001
    batch_size = env.batch_size
    pomo_size = env.pomo_size

    route_limit = env.route_limit[:, :, None].expand(batch_size, pomo_size, num_nodes)
    current_node_to_all_dist = env.dist.gather(
        1, env.current_node[:, :, None].expand(-1, -1, num_nodes)
    )
    current_all_to_depot_dist = env.dist.gather(
        2, env.current_depot[:, None, :].expand(-1, num_nodes, -1)
    )
    current_all_to_depot_dist = current_all_to_depot_dist.permute(0, 2, 1)

    mask = torch.zeros((batch_size, pomo_size, num_nodes))


    if env.open_route:
        # check route limit constraint: length + cur->next <= route_limit
        route_too_large = env.length[:, :,None] + current_node_to_all_dist > route_limit + round_error_epsilon
        route_too_large[:, :, 0] = False
        # shape: (batch, pomo, problem+1)
        mask[route_too_large] = float('-inf')
    else:
        # check route limit constraint: length + cur->next->depot <= route_limit
        route_too_large = env.length[:, :,None] + current_node_to_all_dist + current_all_to_depot_dist > route_limit + round_error_epsilon
        # shape: (batch, pomo, problem+1)
        mask[route_too_large] = float('-inf')

    return mask
