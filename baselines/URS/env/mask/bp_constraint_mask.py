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

@register_mask("bp")
def bp_constraint_mask(env) -> torch.Tensor:
    current_node = env.current_node.clone()
    demands = env.depot_node_demand[:, None, :].expand(env.batch_size, env.pomo_size, -1)
    num_nodes = env.problem_size + env.depot_num

    mask = torch.zeros((current_node.size(0), current_node.size(1), num_nodes))
    gathering_index = current_node.unsqueeze(-1)
    selected_demand = demands.gather(dim=2, index=gathering_index).squeeze(dim=2)
    load_with_backhaul = (selected_demand < 0).unsqueeze(-1)
    # If a backhaul node (demand < 0) is accessed, the linehaul node (demand > 0) becomes inaccessible.
    mask[load_with_backhaul&(demands>0)] = float('-inf')

    return mask

