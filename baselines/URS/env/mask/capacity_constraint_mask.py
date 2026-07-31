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

@register_mask("vrp")
def capacity_constraint_mask(env) -> torch.Tensor:
    demands = env.depot_node_demand[:, None, :].expand(env.batch_size, env.pomo_size, -1)
    current_capacity = env.load.clone()
    problem_size = env.problem_size + env.depot_num


    demand_too_large = demands > current_capacity.unsqueeze(-1) + 0.00001
    exceed_capacity = current_capacity.unsqueeze(-1) - demands > 1.0 + 0.00001

    # Combine the masks
    illegal_mask = demand_too_large| exceed_capacity

    mask = torch.zeros((env.batch_size, env.pomo_size, problem_size))
    mask[illegal_mask] = float('-inf')

    return mask