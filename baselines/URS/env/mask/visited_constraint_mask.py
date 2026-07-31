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
@register_mask("visited")
def mask_visited_nodes(env) -> torch.Tensor:
    num_nodes = env.problem_size + env.depot_num
    batch_size = env.batch_size
    pomo_size = env.pomo_size


    mask = torch.zeros((batch_size, pomo_size, num_nodes))
    mask.scatter_(-1, env.selected_node_list, float('-inf'))
    mask.scatter_(-1, env.current_node.unsqueeze(-1), float('-inf'))
    return mask