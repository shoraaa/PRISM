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

from typing import Dict, Callable, List, Optional
from functools import wraps
from logging import getLogger

logger = getLogger(__name__)

class MaskRegistry:
    def __init__(self):
        self._constraint_masks: Dict[str, Callable] = {}
        self._basic_constraint = 'visited'
        logger.info("First initializing MaskRegistry...")
        logger.info("Registered basic mask: {}".format(self._basic_constraint))

    def register(self, constraint: str, mask_fn: Callable):
        self._constraint_masks[constraint] = mask_fn
        logger.info("Registered mask for constraint: {}".format(constraint))

    def build_combined_mask(self, problem_name: str) -> Optional[Callable]:
        needed_masks = []
        if self._basic_constraint in self._constraint_masks:
            needed_masks.append(self._constraint_masks[self._basic_constraint])

        target_constraints = ['op', 'pc', 'pd', 'vrp', 'tw', 'l', 'bp']
        for constraint in target_constraints:
            if constraint in problem_name and constraint in self._constraint_masks:
                needed_masks.append(self._constraint_masks[constraint])

        if not needed_masks:
            logger.info("No valid masks found for problem: {}".format(problem_name))
            return None

        @wraps(needed_masks[0])
        def combined_mask(env):
            final_mask = needed_masks[0](env)
            for mask_fn in needed_masks[1:]:
                current_mask = mask_fn(env)
                final_mask +=current_mask
            return final_mask

        return combined_mask

    def list_registered_problems(self):
        """List all registered problem types"""
        return list(self._constraint_masks.keys())

    def __contains__(self, problem_name: str) -> bool:
        """Check if problem has registered mask"""
        return problem_name in self._constraint_masks


# Global mask registry instance
mask_registry = MaskRegistry()


def register_mask(constraint: str):
    """
    Decorator to register a mask function

    Usage:
        @register_mask('tw')
        def tw_constraint_mask(env, selected):
            # ... mask logic ...
            return mask
    """

    def decorator(mask_fn: Callable):
        mask_registry.register(constraint, mask_fn)
        return mask_fn

    return decorator










