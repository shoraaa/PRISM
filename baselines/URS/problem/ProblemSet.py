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

from itertools import product
from typing import ClassVar, List, Optional, Union


BEFORE_CVRP_PREFIXES = ("", "o") # 放在 cvrp 前面的前缀
EXCLUSIVE_AFTER_CVRP_SUFFIXES = ("", "b", "bp") # 互斥选项：每个问题只能选一个
OPTIONAL_AFTER_CVRP_SUFFIXES = ("l", "tw") # 可选选项：可以组合使用, 并且放在 cvrp 后面


# ascending order by name length for better readability
def sort_by_name_length(problem_list):
    return sorted(problem_list, key=len)


def build_vrpmix_list():
    problem_list = []

    for before_cvrp_prefix in BEFORE_CVRP_PREFIXES:
        for exclusive_after_cvrp_suffix in EXCLUSIVE_AFTER_CVRP_SUFFIXES:
            for enabled_flags in product([False, True], repeat=len(OPTIONAL_AFTER_CVRP_SUFFIXES)):
                optional_suffix = "".join(
                    constraint
                    for constraint, enabled in zip(OPTIONAL_AFTER_CVRP_SUFFIXES, enabled_flags)
                    if enabled
                )
                problem_list.append(
                    f"{before_cvrp_prefix}cvrp{exclusive_after_cvrp_suffix}{optional_suffix}"
                )

    return sort_by_name_length(problem_list)


def with_problem_prefix(prefix, problem_list):
    return [f"{prefix}{problem}" for problem in problem_list]


# build zero_shot_list, asymmetric_list and symmetric_list based on all_evaluated_list and train_problem_list
def build_derived_lists(all_evaluated_list, train_problem_list):
    train_problem_set_ = set(train_problem_list)
    zero_shot_list = []
    asymmetric_list = []
    symmetric_list = []

    for problem in all_evaluated_list:
        if problem not in train_problem_set_:
            zero_shot_list.append(problem)
        if problem.startswith("a"):
            asymmetric_list.append(problem)
        else:
            symmetric_list.append(problem)

    return (
        sort_by_name_length(zero_shot_list),
        sort_by_name_length(asymmetric_list),
        sort_by_name_length(symmetric_list),
    )


class ProblemSet:
    """
    The total number of problems evaluated in our experiments is 110.

    `ProblemSet` provides predefined lists of problems based on different
    classification methods. Use `name` to retrieve a named problem set,
    or use `included` / `excluded` to filter problems whose names should contain
    or avoid specific substrings, such as 'l', 'o', 'tw', 'b', 'bp', 'md', 'pd'
    and 'a'. It can be a list of multiple strings, such as included=['l', 'tw'].

    If both parameters are None, the full list of 110 problems is returned.
    If `name` is provided, it has priority and `included` / `excluded` are ignored.

    Commonly adopted problem classification methods in our experiments:

    1. Based on constraints:
        * Single depot, symmetric CVRP variants (24) -> `vrpmix_list`
        * Single depot, asymmetric CVRP variants (24) -> `avrpmix_list`
        * Multi depots, symmetric CVRP variants (24) -> `mdvrpmix_list`
        * Multi depots, asymmetric CVRP variants (24) -> `amdvrpmix_list`
        * CVRP variants with single depot (48) -> `all_sdvrpmix_list`
        * CVRP variants with multiple depots (48) -> `all_mdvrpmix_list`
        * CVRP variants with B(P), L, TW, O constraint (96) -> `all_vrpmix_list`
        * PDCVRP variants (4) -> `pdcvrp_list`
        * Remaining problems with one depot (10) -> `remaining_list`
        * Total evaluated problems (110) -> `all_evaluated_list`

    2. Based on training:
        * Training problems (11) -> `train_problem_list`
        * Zero-shot testing problems (99) -> `zero_shot_list`

    3. Based on symmetry:
        * Symmetric problems (55) -> `symmetric_list`
        * Asymmetric problems (55) -> `asymmetric_list`
        
    4. Based on problem scale:
        * Large scale problems (3) -> `large_scale_list`
    """

    # CVRP and its variants, combining l, o, tw, b, bp constraints, totaling 24 problems.
    vrpmix_list: ClassVar[List[str]] = build_vrpmix_list()

    # ACVRP and its variants, combining l, o, tw, b, bp constraints, totaling 24 problems.
    avrpmix_list: ClassVar[List[str]] = with_problem_prefix("a", vrpmix_list)

    # MDCVRP and its variants, combining l, o, tw, b, bp constraints, totaling 24 problems.
    mdvrpmix_list: ClassVar[List[str]] = with_problem_prefix("md", vrpmix_list)

    # AMDCVRP and its variants, combining l, o, tw, b, bp constraints, totaling 24 problems.
    amdvrpmix_list: ClassVar[List[str]] = with_problem_prefix("amd", vrpmix_list)

    # PDCVRP variants, which do not include pdtsp and apdtsp, totaling 4 problems.
    pdcvrp_list: ClassVar[List[str]] = sort_by_name_length(["pdcvrp", "apdcvrp", "opdcvrp", "aopdcvrp"])

    # Remaining problems with one depot, totaling 10 problems.
    remaining_list: ClassVar[List[str]] = sort_by_name_length([
        "tsp", "pctsp", "spctsp", "op", "pdtsp", "atsp", "apctsp", "aspctsp", "aop", "apdtsp"
        ])

    '''
    11 training tasks are selected by adhering to the principle of minimal redundancy and strictly align with existing work on CVRP variant selection (e.g., MVMoE and ReLD): 
    Most features in UDR appear in at least two training problems, helping prevent URS from overfitting to a single problem. 
    The only exception is the Penalty attribute, as there is structural similarity between the seen PCTSP and unseen SPCTSP.
    '''
    train_problem_list: ClassVar[List[str]] = sort_by_name_length([
        'atsp', 'acvrp', 'tsp', 'op', 'pctsp', 'cvrp', 'cvrpb', 'cvrptw', 'ocvrp', 'ocvrptw', 'pdtsp'
        ])
    
    # all evaluated problems, totaling 110 problems.
    all_evaluated_list: ClassVar[List[str]] = sort_by_name_length(
        vrpmix_list
        + avrpmix_list
        + mdvrpmix_list
        + amdvrpmix_list
        + pdcvrp_list
        + remaining_list
    )
    assert len(all_evaluated_list) == 110, f"Expected 110 problems, but got {len(all_evaluated_list)}"

    all_sdvrpmix_list: ClassVar[List[str]] = sort_by_name_length(vrpmix_list + avrpmix_list)
    all_mdvrpmix_list: ClassVar[List[str]] = sort_by_name_length(mdvrpmix_list + amdvrpmix_list)
    all_vrpmix_list: ClassVar[List[str]] = sort_by_name_length(all_sdvrpmix_list + all_mdvrpmix_list)
    
    zero_shot_list, asymmetric_list, symmetric_list = build_derived_lists(
        all_evaluated_list,
        train_problem_list,
    )

    large_scale_list: ClassVar[List[str]] = ["cvrptw", "cvrpb", "ocvrptw"]
    
    # construct a dictionary to store all collections for easy retrieval
    @classmethod
    def _collections(cls):
        return {
            name: value
            for name, value in vars(cls).items()
            if name.endswith("_list") and not name.startswith("_") and isinstance(value, list)
        }
        
    @classmethod
    def get_all_problem_names(cls):
        return cls.all_evaluated_list

    # get problems based on name or included/excluded characters(can be a list of multiple characters)
    @classmethod
    def get(
        cls,
        name: Optional[str] = None,
        included: Optional[Union[str, List[str]]] = None,
        excluded: Optional[Union[str, List[str]]] = None,
    ) -> List[str]:
        if name is None and included is None and excluded is None:
            return cls.all_evaluated_list

        if name is not None:
            collections = cls._collections()
            if name in collections:
                return collections[name]
            raise ValueError(
                f"Invalid name: {name}. Valid names are: "
                f"{', '.join(collections.keys())}"
            )

        if isinstance(included, str):
            included = [included]
        if included is None:
            included = []

        if isinstance(excluded, str):
            excluded = [excluded]
        if excluded is None:
            excluded = []

        return sort_by_name_length([
            problem
            for problem in cls.all_evaluated_list
            if all(sub_str in problem for sub_str in included)
            and not any(sub_str in problem for sub_str in excluded)
        ])
