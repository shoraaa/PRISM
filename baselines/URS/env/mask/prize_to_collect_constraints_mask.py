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
@register_mask("pc")
def prize_to_collect_constraint_mask(env) -> torch.Tensor:

    num_nodes = env.problem_size + env.depot_num


    mask = torch.zeros((env.batch_size, env.pomo_size, num_nodes))
    mask[:,:,0] = float('-inf')
    allow = (env.collected_prize >= 1.) | (env.selected_count == num_nodes)

    # 将finished的其他action设置为-inf
    finished = env.at_the_depot & (env.selected_count > 1)
    finished_extend = finished.unsqueeze(-1).expand(-1, -1, num_nodes)
    mask[finished_extend] = float('-inf')
    mask[:, :, 0][allow] = 0

    return mask