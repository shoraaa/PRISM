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

from .ProblemSet import ProblemSet

def get_problem_representations():
    """
    Generates and returns a dictionary of feature representations for various VRP variants.

    This function defines attribute vectors for 110 different routing problems (e.g., TSP, 
    CVRP, VRPTW). Each vector is a list of 13 binary integers (0 or 1) indicating 
    the presence of specific constraints or characteristics.

    Feature Vector Index Mapping (0-12):
        0: identifier   - 1 if using node IDs, 0 if using coordinates.
        1: coordinates  - 1 if coordinate data is available.
        2: demand       - 1 if there is a capacity constraint or demand attribute.
        3: prize        - 1 if the problem includes prizes.
        4: penalty      - 1 if the problem includes penalties.
        5: early_time   - 1 if there is an early arrival time window.
        6: late_time    - 1 if there is a late arrival time window.
        7: service_time - 1 if there is a duration for service at nodes.
        8: depot        - 1 if a starting/ending depot is required.
        9: pickup       - 1 if the task includes picking up goods or backhauls.
        10: delivery    - 1 if the task includes delivering goods.
        11: multi_route - 1 if multiple routes/vehicles are allowed.
        12: open_route  - 1 if it is an open route (no return to depot).

    Returns:
        dict: A dictionary where keys are problem type strings (e.g., 'atsp') 
              and values are the 13-bit attribute lists.

    Raises:
        AssertionError: If the resulting dictionary does not contain exactly 110 problems.
    """
    # [identifier0, coordinates1, demand2, prize3, penalty4, early_time5, late_time6, service_time7, depot8, pickup9, delivery10, multi_route11, open_route12]
    problem_representation_dict = {
        'atsp':    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # identifier
        'tsp':     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # coordinates
        'op':      [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # coordinates, prize,depot
        'aop':     [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # identifier, prize,depot
        'pctsp':   [0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # coordinates, prize,penalty,depot
        'apctsp':  [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # identifier, prize,penalty,depot
        'spctsp':  [0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # coordinates, prize,penalty,depot
        'aspctsp': [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # identifier, prize,penalty,depot
        'pdtsp':   [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],  # coordinates,depot, pickup,delivery
        'apdtsp':  [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],  # identifier,depot, pickup,delivery
        'pdcvrp':  [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],  # coordinates,demand, depot,pickup,delivery,multi_route
        'apdcvrp': [1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],  # identifier,demand, depot,pickup,delivery,multi_route
        'opdcvrp': [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # coordinates,demand, depot,pickup,delivery,multi_route,open_route
        'aopdcvrp':[1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # identifier,demand, depot,pickup,delivery,multi_route,open_route
    }
    
    variant_configs = {
        "vrpmix_list":  [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0], # coordinates, demand, depot, delivery, multi_route
        "avrpmix_list": [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0], # identifier , demand, depot, delivery, multi route
    }
    
    vrpmix_variants = ProblemSet.get(name="vrpmix_list")
    for k in vrpmix_variants:
        attributes = variant_configs["vrpmix_list"].copy() 
        if 'tw' in k:
            attributes[5] = 1  #early_time
            attributes[6] = 1  #late_time
            attributes[7] = 1  #service_time
        if 'b' in k:
            attributes[9] = 1 #pickup
        if 'o' in k:
            attributes[-1] = 1 #open route
        problem_representation_dict[k] = attributes

    avrpmix_variants = ProblemSet.get(name="avrpmix_list")
    for k in avrpmix_variants:
        attributes = variant_configs["avrpmix_list"].copy()
        if 'tw' in k:
            attributes[5] = 1  #early_time
            attributes[6] = 1  #late_time
            attributes[7] = 1  #service_time
        if 'b' in k:
            attributes[9] = 1 #pickup
        if 'o' in k:
            attributes[-1] = 1 #open route
        problem_representation_dict[k] = attributes

    mdvrpmix_variants = ProblemSet.get(name="mdvrpmix_list")
    for k in mdvrpmix_variants:
        attributes = variant_configs["vrpmix_list"].copy()
        if 'tw' in k:
            attributes[5] = 1  #early_time
            attributes[6] = 1  #late_time
            attributes[7] = 1  #service_time
        if 'b' in k:
            attributes[9] = 1 #pickup
        if 'o' in k:
            attributes[-1] = 1 #open route
        problem_representation_dict[k] = attributes

    amdvrpmix_variants = ProblemSet.get(name="amdvrpmix_list")
    for k in amdvrpmix_variants:
        attributes = variant_configs["avrpmix_list"].copy()
        if 'tw' in k:
            attributes[5] = 1  #early_time
            attributes[6] = 1  #late_time
            attributes[7] = 1  #service_time
        if 'b' in k:
            attributes[9] = 1 #pickup
        if 'o' in k:
            attributes[-1] = 1 #open route
        problem_representation_dict[k] = attributes
    
    assert len(problem_representation_dict) == 110, f"problem_representation_dict should contain 110 problems, but got {len(problem_representation_dict)}"

    return problem_representation_dict



