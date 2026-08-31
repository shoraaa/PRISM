"""Compiled kernels are fast paths for declared rows, not a private semantics.

Every compiled constraint kernel publishes the declarative resource row it
implements (``Decoder.resource_declarations``). These tests rebuild each such
constraint from its published row alone -- dropping the compiled kernel from the
schema -- and require the two decoders to produce the identical solution. What
the kernel does fast, the algebra therefore also says exactly; the generality
claim rests on the published row, and the kernel is only an optimization.

Kernels whose semantics fall outside the current language report
``declared=False`` and are asserted to be exactly the known set, so the gap
stays visible instead of silently widening.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import prism_decoder  # noqa: E402
from problem_data import generated_problem, problem_schema  # noqa: E402
from train import _neutral_guidance  # noqa: E402


# Kernels the resource language cannot express yet, and why: backhaul order and
# pickup-delivery are precedence relations over nodes rather than resource
# extension functions at all, so no operator in an extension-function language
# reaches them.
UNDECLARABLE: set[str] = set()

# Relation roles no longer have node-feature columns of their own: a pairwise
# row publishes its relation delta as an ordinary node term, so the roles arrive
# in node_resource_features under the shared per-resource weights. Slots 0/1 are
# the signed increment (a chain's middle node nets to zero, correctly) and slot 5
# is the node's rank in the declared order, which is what separates a middle node
# from one in no relation at all.
# Row property carrying declared-relation density. Named rather than inlined:
# the property vector is compressed whenever a dimension stops being separately
# observable, so a bare integer here goes stale silently.
RELATION_DENSITY_PROPERTY = 13
RELATION_OPENS_SLOT = 0
RELATION_REQUIRES_SLOT = 1
RELATION_RANK_SLOT = 5
CHAIN_HEAD_RANK = 1.0 / 3.0
CHAIN_MIDDLE_RANK = 2.0 / 3.0
CHAIN_TAIL_RANK = 1.0


def _relation_rows(decoder):
    """[node, slot] of the single active precedence row's node attributes."""
    rows = [
        index
        for index, row in enumerate(decoder.resource_declarations)
        if row["active"] and row.get("operator") == "precedence"
    ]
    assert len(rows) == 1, rows
    return np.asarray(decoder.node_resource_features)[:, rows[0]]

# Constraint schema name -> registry kernel name.
KERNEL_OF_CONSTRAINT = {
    "capacity": "capacity",
    "route_limit": "route_limit",
    "tour_limit": "tour_limit",
    "prize_quota": "prize_quota",
    "time_windows": "time_window",
    "backhaul_order": "backhaul_order",
    "pickup_delivery": "pickup_delivery",
}


def _registry(solver) -> list[str]:
    """Registry order for THIS problem.

    A row's position depends on which constraints the instance declares, so it
    cannot be read off prism_decoder.FIELD_CHANNEL_NAMES -- that enumerates the
    compiled fast paths, not any one problem's rows.
    """
    return list(solver.metadata["resource_names"])


def _row(solver, name: str) -> int:
    return _registry(solver).index(name)


def _program_row(solver, index: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(solver.resource_term_counts, dtype=np.int64)
    start = int(counts[:index].sum())
    stop = start + int(counts[index])
    return (
        np.asarray(solver.resource_row_properties)[index],
        np.asarray(solver.resource_term_properties)[start:stop],
    )


def _decoder(problem: dict, **kwargs) -> prism_decoder.Decoder:
    explicit = problem_schema(str(problem.get("name", "schema")))
    explicit.update(problem)
    return prism_decoder.Decoder(explicit, **kwargs)


def _row_input(declaration: dict, attributes: dict) -> dict:
    """Turn a published declaration back into a solver input row."""
    if declaration["operator"] == "precedence":
        row = {
            "name": f"declared_{declaration['name']}",
            "operator": "precedence",
            "relation": declaration["relation"],
            "scope": declaration["scope"],
        }
        if declaration["relation"] == "pairwise":
            key = f"{declaration['name']}_predecessor"
            attributes[key] = np.asarray(
                declaration["predecessor"], dtype=np.float32
            )
            row["predecessor"] = {"node_attribute": key}
        else:
            key = f"{declaration['name']}_class"
            attributes[key] = np.asarray(declaration["class"], dtype=np.float32)
            row["class"] = {"node_attribute": key}
        return row

    # Inactive compiled rows keep their registry names, so the twin row needs a
    # distinct one; the name is metadata and never reaches the solver semantics.
    row: dict = {
        "name": f"declared_{declaration['name']}",
        "state_dim": declaration["state_dim"],
        "direction": declaration["direction"],
        "scope": declaration["scope"],
        "initial": declaration["initial"],
        "scale": declaration["scale"],
    }
    row["operator"] = declaration["operator"]
    row["semiring"] = declaration["semiring"]
    # Rebuild the extension from the published terms rather than from the named
    # shorthand. The shorthand is lossy by construction -- it can only name the
    # combinations that happen to have fields -- so a new behavior added later would
    # silently vanish from the twin. Terms carry their own coordinates, so this
    # loop needs no edit when one is added.
    terms = []
    for index, term in enumerate(declaration["terms"]):
        entry: dict = {
            "op": term["op"],
            "phase": term["phase"],
            "when": term["when"],
            "at": term["at"],
            "coefficient": term["coefficient"],
            "at_depot": term["at_depot"],
        }
        if len(term["trigger_nodes"]):
            key = f"{declaration['name']}_trigger{index}"
            attributes[key] = np.asarray(term["trigger_nodes"], dtype=np.float32)
            entry["at_nodes"] = {"node_attribute": key}
        if term["gate"] != "always":
            key = f"{declaration['name']}_gate{index}"
            attributes[key] = np.asarray(term["gate_values"], dtype=np.float32)
            entry["gate"] = {
                "node_attribute": key,
                "sign": "positive" if term["gate_sign"] >= 0 else "negative",
                "branch": (
                    "alternative"
                    if term["gate"] == "remainder_alternative"
                    else "default"
                ),
            }
        if term["source"] == "distance":
            entry["edge_attribute"] = "distance"
        elif term["source"] == "value":
            entry["value"] = term["value"]
        else:
            key = f"{declaration['name']}_term{index}"
            attributes[key] = np.asarray(term["values"], dtype=np.float32)
            entry[
                "node_attribute"
                if term["source"] == "node_attribute"
                else "edge_attribute"
            ] = key
        terms.append(entry)
    if terms:
        row["terms"] = terms
    bound = dict(declaration["bounds"][0])
    for side in ("lower", "upper"):
        values = bound.pop(f"{side}_values")
        if len(values):
            key = f"{declaration['name']}_{side}"
            attributes[key] = np.asarray(values, dtype=np.float32)
            bound[side] = {"node_attribute": key}
        elif not np.isfinite(bound[side]):
            del bound[side]
    row["bounds"] = [bound]
    return row


def _declared_twin(problem: dict, constraint: str) -> dict:
    """Replace one compiled constraint with its own published resource row."""
    compiled = _decoder(problem)
    kernel = KERNEL_OF_CONSTRAINT[constraint]
    declarations = [
        row
        for row in compiled.resource_declarations
        if row["active"] and row["kernel"] == kernel
    ]
    assert len(declarations) == 1, f"{constraint} must publish exactly one row"
    declaration = declarations[0]
    assert declaration["declared"], f"{kernel} is expected to be declarable"

    twin = problem_schema(str(problem["name"]))
    twin.update(problem)
    twin["constraints"] = [
        name for name in twin["constraints"] if name != constraint
    ]
    attributes = dict(twin.get("node_attributes", {}))
    twin["resources"] = [_row_input(declaration, attributes)]
    if attributes:
        twin["node_attributes"] = attributes
    return twin


def _solve(problem: dict, seed: int) -> tuple[float, list[int]]:
    solver = (
        prism_decoder.Decoder(problem)
        if "constraints" in problem
        else _decoder(problem)
    )
    solver.seed(seed)
    solution = solver.solve(24)
    return float(solution["objective"]), [int(node) for node in solution["route"]]


@pytest.mark.parametrize(
    ("variant", "constraint", "size"),
    [
        ("cvrp", "capacity", 30),
        ("cvrpl", "capacity", 30),
        ("cvrpl", "route_limit", 30),
        ("op", "tour_limit", 30),
        ("pctsp", "prize_quota", 40),
        ("vrptw", "time_windows", 30),
        ("cvrptw", "time_windows", 30),
        ("cvrpbp", "backhaul_order", 30),
        ("ocvrpbp", "backhaul_order", 30),
        ("pdcvrp", "pickup_delivery", 30),
        ("pdtsp", "pickup_delivery", 30),
    ],
)
def test_declared_row_reproduces_the_compiled_kernel(
    variant: str, constraint: str, size: int
) -> None:
    for seed in (11, 12, 13):
        # generated_problem draws from the global torch RNG, so pin it here to
        # keep the instance independent of whatever ran before in the session.
        torch.manual_seed(seed)
        problem = generated_problem(variant, size, seed)
        twin = _declared_twin(problem, constraint)
        assert _solve(problem, seed) == _solve(twin, seed)


def test_a_kernel_abstains_only_inside_its_documented_domain() -> None:
    """No kernel may quietly drop out of the declarative language.

    Every compiled kernel is expressible; two abstain on instances that leave
    the domain their declaration covers, and nothing else may abstain at all.
    Encoding the rule rather than a fixed set means a new abstention has to be
    justified here before the test will pass.
    """
    for variant in ("cvrp", "cvrptw", "cvrpb", "cvrpbp", "pdcvrp", "pdtsp",
                    "op", "pctsp", "cvrpl", "mdcvrptw"):
        problem = generated_problem(variant, 24, 5)
        signed_demand = float(np.asarray(problem.get("demand", [0.0])).min()) < 0.0
        del signed_demand
        for row in _decoder(problem).resource_declarations:
            if not row["active"]:
                continue
            # No kernel abstains anywhere in the benchmark family: capacity's
            # last gap closed once the opening load became a term. The only
            # remaining abstention needs a declared class order, which
            # test_capacity_abstains_only_under_a_declared_class_order covers.
            assert row["declared"] is True, (
                f"{variant}: {row['kernel']} declared={row['declared']}"
            )


def test_declared_rows_are_published_for_every_registry_entry() -> None:
    problem = generated_problem("cvrptw", 24, 5)
    solver = _decoder(problem)
    assert len(solver.resource_declarations) == len(solver.resource_scales)


def _driver_break_problem(variant: str, size: int, seed: int) -> dict:
    from prism_eval import instances

    torch.manual_seed(seed)
    data = instances.GENERATORS[variant](size, 1, seed=seed)
    return instances.solver_problem(variant, instances._instance_data(data, 0))


def test_time_windows_abstain_when_another_row_charges_wall_time() -> None:
    """A break costs wall time, which the declared clock cannot account for.

    The compiled time-window kernel folds the driving-hours row's optional-reset
    duration into its own clock. No operator in the language expresses one row's
    increment depending on another row's reset, so the kernel must abstain
    wherever such a row is active rather than publish a declaration that is exact
    only while no break happens to be taken.
    """
    problem = _driver_break_problem("vrpdbtw", 24, 3)
    declarations = prism_decoder.Decoder(problem).resource_declarations
    time_window = [
        row for row in declarations if row["active"] and row["kernel"] == "time_window"
    ]
    assert len(time_window) == 1
    assert time_window[0]["declared"] is False

    # Without a break row present the same kernel declares itself exactly.
    plain = generated_problem("cvrptw", 24, 3)
    rows = [
        row
        for row in _decoder(plain).resource_declarations
        if row["active"] and row["kernel"] == "time_window"
    ]
    assert len(rows) == 1 and rows[0]["declared"] is True


def test_the_driver_break_resource_is_load_bearing_in_vrpdb() -> None:
    """Guard against the probe silently degenerating to a trained channel.

    A driver-break row whose breaks never fire is just a per-route length cap --
    algebraically the trained route-limit channel -- so it would stop testing an
    unseen resource at all. Deleting every break-eligible node must therefore
    change the solution.
    """
    changed = 0
    for seed in (3, 4, 5):
        problem = _driver_break_problem("vrpdb", 30, seed)
        without = dict(problem)
        attributes = dict(problem["node_attributes"])
        attributes["break_allowed"] = np.zeros_like(
            np.asarray(attributes["break_allowed"])
        )
        without["node_attributes"] = attributes
        changed += _solve(problem, seed) != _solve(without, seed)
    assert changed == 3, "vrpdb breaks never fire; the resource has degenerated"


def _accumulator_probe(size: int, cap_at_depot: float) -> dict:
    """Solution-scoped row whose terminal bound only binds at the depot."""
    rng = np.random.default_rng(5)
    gain = np.r_[0.0, np.full(size - 1, 0.2)].astype(np.float32)
    cap = np.full(size, 1e6, np.float32)
    cap[0] = cap_at_depot
    return {
        "name": "probe",
        "constraints": ["visit_all"],
        "objective": "distance",
        "depot_count": 1,
        "multi_route": True,
        "open_route": False,
        "coordinates": rng.random((size, 2), dtype=np.float32),
        "capacity": 1.0,
        "demand": np.zeros(size, np.float32),
        "node_attributes": {"gain": gain, "cap": cap},
        "resources": [
            {
                "name": "total",
                "operator": "affine_accumulator",
                "state_dim": 1,
                "direction": "forward",
                "scope": "solution",
                "initial": 0.0,
                "scale": 1.0,
                "increment": {"node_attribute": "gain", "coefficient": 1.0},
                "bounds": [
                    {"upper": {"node_attribute": "cap"}, "check": "solution_end"}
                ],
            }
        ],
    }


def test_a_per_node_terminal_bound_binds_in_every_validator() -> None:
    """Construction, evaluate(), and the resource report must agree.

    A solution-scoped row with a per-node terminal bound was once rejected by
    construction (which resolved the bound at the terminal node) and accepted by
    evaluate() (which read the scalar fallback, here infinity).
    """
    size = 14
    total = 0.2 * (size - 1)
    route = list(range(size)) + [0]

    solver = prism_decoder.Decoder(_accumulator_probe(size, cap_at_depot=0.5))
    assert solver.evaluate(route)["feasible"] is False
    assert solver.evaluate_resources(route)["violation"][-1] == pytest.approx(
        total - 0.5, abs=1e-4
    )

    relaxed = prism_decoder.Decoder(_accumulator_probe(size, cap_at_depot=100.0))
    assert relaxed.evaluate(route)["feasible"] is True
    assert relaxed.evaluate_resources(route)["violation"][-1] == 0.0


def test_violation_reporting_uses_the_bound_feasibility_used() -> None:
    """A route evaluate() rejects may not be reported as violation-free."""
    torch.manual_seed(11)
    problem = generated_problem("cvrptw", 24, 11)
    twin = _declared_twin(problem, "time_windows")
    solver = prism_decoder.Decoder(twin)
    deadlines = np.asarray(problem["tw_end"])
    # Visiting in reverse-deadline order violates the windows badly.
    route = [0] + sorted(range(1, len(deadlines)), key=lambda i: -deadlines[i]) + [0]

    assert solver.evaluate(route)["feasible"] is False
    assert solver.evaluate_resources(route)["violation"][-1] > 0.0


def _declared_class_order_problem(size: int = 24, capacity: int = 12) -> dict:
    """Signed demand plus a *declared* class order, with no backhaul flag.

    This is the one configuration capacity still cannot declare. The opening
    term that covers the class-ordered opening load is emitted by publish(),
    which runs during the registry build -- before precedence rows are indexed
    -- so it keys on the problem flag. A class order arriving as a declared row
    turns class_ordered() true only afterwards, leaving opening_load applying a
    rule the published row does not carry.
    """
    problem = generated_problem("cvrpb", size, capacity)
    explicit = problem_schema("cvrpb")
    explicit.update(problem)
    demand = np.asarray(explicit["demand"], dtype=np.float32)
    explicit["node_attributes"] = {"zone_class": (demand < 0).astype(np.float32)}
    explicit["resources"] = [
        {
            "name": "zones",
            "operator": "precedence",
            "relation": "class_order",
            "class": {"node_attribute": "zone_class"},
            "scope": "route",
        }
    ]
    return explicit


def _capacity_row(problem: dict) -> dict:
    rows = [
        row
        for row in _decoder(problem).resource_declarations
        if row["active"] and row["kernel"] == "capacity"
    ]
    assert len(rows) == 1
    return rows[0]


def test_capacity_abstains_only_under_a_declared_class_order() -> None:
    """Every part of the kernel is stated now except one build-order corner.

    Signed demand is a guarded reset; the class-ordered opening load is an
    opening term. What is left is a class order that arrives as a declared row
    rather than as the problem flag, which publish() cannot see.
    """
    torch.manual_seed(11)
    plain = generated_problem("cvrp", 30, 11)
    assert np.asarray(plain["demand"]).min() >= 0.0
    assert _capacity_row(plain)["declared"] is True

    backhaul = generated_problem("cvrpb", 30, 11)
    assert np.asarray(backhaul["demand"]).min() < 0.0
    assert _capacity_row(backhaul)["declared"] is True

    ordered = generated_problem("cvrpbp", 30, 11)
    assert np.asarray(ordered["demand"]).min() < 0.0
    assert _capacity_row(ordered)["declared"] is True

    torch.manual_seed(11)
    assert _capacity_row(_declared_class_order_problem(30, 11))["declared"] is False


def test_unexecuted_resource_directions_are_rejected() -> None:
    """Execution is forward-only, so a backward label must not be accepted."""
    for direction in ("backward", "bidirectional"):
        problem = _accumulator_probe(10, cap_at_depot=100.0)
        problem["resources"][0]["direction"] = direction
        with pytest.raises(ValueError, match="forward-only"):
            prism_decoder.Decoder(problem)


def test_a_precedence_chain_runs_although_no_kernel_expresses_one() -> None:
    """The point of the family: a third instantiation neither kernel covers.

    Pickup-delivery is a depth-1 pair and backhaul a two-class order. A longer
    chain a -> b -> c is a structurally new instantiation of the same declared
    relation, reachable with no new kernel, output head, or variant name.
    """
    size = 25
    rng = np.random.default_rng(9)
    chains = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)]
    predecessor = np.full(size, -1.0, dtype=np.float32)
    for first, second, third in chains:
        predecessor[second] = first
        predecessor[third] = second

    problem = {
        "name": "chain",
        "constraints": ["visit_all"],
        "objective": "distance",
        "depot_count": 1,
        "multi_route": True,
        "open_route": False,
        "coordinates": rng.random((size, 2), dtype=np.float32),
        "capacity": 1.0,
        "demand": np.zeros(size, np.float32),
        "node_attributes": {"chain_predecessor": predecessor},
        "resources": [
            {
                "name": "assembly_order",
                "operator": "precedence",
                "relation": "pairwise",
                "scope": "route",
                "predecessor": {"node_attribute": "chain_predecessor"},
            }
        ],
    }

    solver = prism_decoder.Decoder(problem)
    solver.seed(9)
    solution = solver.solve(24)
    route = [int(node) for node in solution["route"]]
    assert solution["feasible"] is True
    assert sorted(node for node in route if node > 0) == list(range(1, size))

    position = {node: index for index, node in enumerate(route) if node > 0}
    for first, second, third in chains:
        assert position[first] < position[second] < position[third]
        # A route may not close between two linked nodes.
        assert 0 not in route[position[first] : position[third]]

    # The declared relation is what enforces it: dropping the row admits orders
    # the chain forbids, so the constraint is not an artifact of the geometry.
    unconstrained = dict(problem)
    unconstrained.pop("resources")
    unconstrained.pop("node_attributes")
    free = prism_decoder.Decoder(unconstrained)
    free.seed(9)
    free_route = [int(node) for node in free.solve(24)["route"]]
    free_position = {node: i for i, node in enumerate(free_route) if node > 0}
    assert any(
        free_position[a] > free_position[b]
        for chain in chains
        for a, b in zip(chain, chain[1:])
    )


def _pairwise_problem(scope: str, size: int = 14) -> dict:
    rng = np.random.default_rng(1)
    predecessor = np.full(size, -1.0, dtype=np.float32)
    predecessor[7] = 1
    return {
        "name": "pairwise",
        "constraints": ["visit_all"],
        "objective": "distance",
        "depot_count": 1,
        "multi_route": True,
        "open_route": False,
        "coordinates": rng.random((size, 2), dtype=np.float32),
        "capacity": 1.0,
        "demand": np.zeros(size, np.float32),
        "node_attributes": {"before": predecessor},
        "resources": [
            {
                "name": "order",
                "operator": "precedence",
                "relation": "pairwise",
                "scope": scope,
                "predecessor": {"node_attribute": "before"},
            }
        ],
    }


def test_precedence_scope_is_executed_not_just_declared() -> None:
    """Route scope confines a pair to one route; solution scope does not."""
    split = [0, 1, 2, 3, 0, 7, 4, 5, 6, 8, 9, 10, 11, 12, 13, 0]
    wrong_order = [0, 7, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 0]

    route_scoped = prism_decoder.Decoder(_pairwise_problem("route"))
    solution_scoped = prism_decoder.Decoder(_pairwise_problem("solution"))

    assert route_scoped.evaluate(split)["feasible"] is False
    assert solution_scoped.evaluate(split)["feasible"] is True
    # Neither scope permits the successor before its predecessor.
    assert route_scoped.evaluate(wrong_order)["feasible"] is False
    assert solution_scoped.evaluate(wrong_order)["feasible"] is False


def test_precedence_violations_are_reported_not_silently_zero() -> None:
    """A route evaluate() rejects may not be reported as violation-free."""
    problem = _pairwise_problem("route")
    solver = prism_decoder.Decoder(problem)
    wrong_order = [0, 7, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 0]

    assert solver.evaluate(wrong_order)["feasible"] is False
    report = solver.evaluate_resources(wrong_order)
    assert report["violation"][-1] > 0.0
    assert report["binding"][-1] == 1.0


def test_two_rows_may_not_claim_the_same_node() -> None:
    """Generic operators assume one partner per node."""
    problem = _pairwise_problem("route")
    attributes = dict(problem["node_attributes"])
    attributes["also_before"] = np.array(attributes["before"], dtype=np.float32)
    problem["node_attributes"] = attributes
    problem["resources"] = problem["resources"] + [
        {
            "name": "second",
            "operator": "precedence",
            "relation": "pairwise",
            "scope": "route",
            "predecessor": {"node_attribute": "also_before"},
        }
    ]
    with pytest.raises(ValueError, match="one pairwise precedence relation"):
        prism_decoder.Decoder(problem)


def test_declarations_a_row_cannot_act_on_are_rejected() -> None:
    """A silently ignored key is how a row ends up meaning something else."""
    precedence = _pairwise_problem("route")
    precedence["resources"][0]["bounds"] = [{"upper": 1.0}]
    with pytest.raises(ValueError, match="precedence row has no"):
        prism_decoder.Decoder(precedence)

    accumulator = _accumulator_probe(10, cap_at_depot=100.0)
    accumulator["resources"][0]["relation"] = "pairwise"
    with pytest.raises(ValueError, match="only a precedence row declares"):
        prism_decoder.Decoder(accumulator)

    tour = _accumulator_probe(10, cap_at_depot=100.0)
    tour["resources"][0]["scope"] = "tour"
    with pytest.raises(ValueError, match="not executed"):
        prism_decoder.Decoder(tour)


def test_a_declared_relation_reaches_the_model_inputs() -> None:
    """The node features must describe a declared row, not only a kernel.

    Feature 10/11 used to read `delivery_of_pickup`/`pickup_of_delivery`
    directly, so a declared pairwise row was invisible to the model even though
    it constrained the search. They are now roles -- opens an obligation,
    requires one -- which a chain's middle node holds both of.
    """
    torch.manual_seed(11)
    problem = generated_problem("pdcvrp", 30, capacity=12)
    twin = _declared_twin(problem, "pickup_delivery")
    compiled_features = np.asarray(_decoder(problem).node_features)
    declared_features = np.asarray(prism_decoder.Decoder(twin).node_features)
    assert np.allclose(compiled_features, declared_features)

    size = 25
    rng = np.random.default_rng(9)
    chains = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)]
    predecessor = np.full(size, -1.0, dtype=np.float32)
    for first, second, third in chains:
        predecessor[second] = first
        predecessor[third] = second
    chain = {
        "name": "chain",
        "constraints": ["visit_all"],
        "objective": "distance",
        "depot_count": 1,
        "multi_route": True,
        "open_route": False,
        "coordinates": rng.random((size, 2), dtype=np.float32),
        "capacity": 1.0,
        "demand": np.zeros(size, np.float32),
        "node_attributes": {"before": predecessor},
        "resources": [
            {
                "name": "chain",
                "operator": "precedence",
                "relation": "pairwise",
                "scope": "route",
                "predecessor": {"node_attribute": "before"},
            }
        ],
    }
    attributes = _relation_rows(prism_decoder.Decoder(chain))
    # 4 heads + 4 middles open an obligation; 4 middles + 4 tails require one.
    # A middle does both, so its NET increment is zero and it shows up in
    # neither sign slot -- the rank slot is what keeps it distinguishable.
    assert attributes[:, RELATION_OPENS_SLOT].sum() == 4.0
    assert attributes[:, RELATION_REQUIRES_SLOT].sum() == 4.0
    ranks = attributes[:, RELATION_RANK_SLOT]
    for head, middle, tail in chains:
        assert attributes[head, RELATION_OPENS_SLOT] == 1.0
        assert attributes[tail, RELATION_REQUIRES_SLOT] == 1.0
        assert attributes[middle, RELATION_OPENS_SLOT] == 0.0
        assert attributes[middle, RELATION_REQUIRES_SLOT] == 0.0
        assert ranks[head] == pytest.approx(CHAIN_HEAD_RANK)
        assert ranks[middle] == pytest.approx(CHAIN_MIDDLE_RANK)
        assert ranks[tail] == pytest.approx(CHAIN_TAIL_RANK)
    # Every node outside a chain takes part in no relation at all.
    inside = {node for chain_nodes in chains for node in chain_nodes}
    outside = [node for node in range(size) if node not in inside]
    assert np.all(ranks[outside] == 0.0)


def test_a_node_may_not_be_claimed_by_a_kernel_and_a_declared_row() -> None:
    """relation_partner returns one partner, so overlap must be rejected."""
    torch.manual_seed(11)
    problem = generated_problem("pdcvrp", 30, capacity=12)
    explicit = problem_schema("pdcvrp")
    explicit.update(problem)
    declarations = [
        row
        for row in _decoder(problem).resource_declarations
        if row["active"] and row["kernel"] == "pickup_delivery"
    ]
    attributes = dict(explicit.get("node_attributes", {}))
    row = _row_input(declarations[0], attributes)
    explicit["node_attributes"] = attributes
    explicit["resources"] = [row]
    with pytest.raises(ValueError, match="one pairwise precedence relation"):
        prism_decoder.Decoder(explicit)


# The channel ordinal used to be a third parameter here. It was the row's
# registry position only while every problem carried one row per channel in
# channel order; both sides now resolve their row by name.
@pytest.mark.parametrize(
    ("variant", "constraint"),
    [
        ("cvrp", "capacity"),
        ("cvrptw", "time_windows"),
        ("cvrpl", "route_limit"),
        ("op", "tour_limit"),
        ("cvrpbp", "backhaul_order"),
        ("pdcvrp", "pickup_delivery"),
        ("pctsp", "prize_quota"),
    ],
)
def test_pressure_comes_from_the_declaration_not_the_kernel(
    variant: str, constraint: str
) -> None:
    """The model must not price a constraint by which path executes it.

    Resource pressure used to be a hand-written formula per field channel. A
    declared row priced generically therefore disagreed with the kernel it was
    otherwise equivalent to -- route_limit by a factor of two, because the
    generic formula ignored the declared return horizon.
    """
    torch.manual_seed(11)
    problem = generated_problem(variant, 30, capacity=12)
    twin = _declared_twin(problem, constraint)
    kernel = KERNEL_OF_CONSTRAINT[constraint]
    compiled_solver = _decoder(problem)
    declared_solver = prism_decoder.Decoder(twin)
    compiled = np.asarray(compiled_solver.resource_pressure)
    declared = np.asarray(declared_solver.resource_pressure)
    assert np.allclose(
        compiled[:, _row(compiled_solver, kernel)],
        declared[:, _row(declared_solver, f"declared_{kernel}")],
        atol=1e-5,
    )


def test_backhaul_abstains_when_a_customer_neither_latches_nor_blocks() -> None:
    """The compiled kernel latches on demand < 0 but only blocks demand > 0.

    A zero-demand customer is exempt from both, which one non-decreasing class
    order cannot express: the class would have to sit below the latch and above
    it at once.
    """
    torch.manual_seed(11)
    problem = generated_problem("cvrpbp", 30, capacity=12)
    explicit = problem_schema("cvrpbp")
    explicit.update(problem)
    rows = [
        row
        for row in prism_decoder.Decoder(explicit).resource_declarations
        if row["active"] and row["kernel"] == "backhaul_order"
    ]
    assert len(rows) == 1 and rows[0]["declared"] is True

    exempt = dict(explicit)
    demand = np.array(explicit["demand"], dtype=np.float32)
    demand[3] = 0.0
    exempt["demand"] = demand
    rows = [
        row
        for row in prism_decoder.Decoder(exempt).resource_declarations
        if row["active"] and row["kernel"] == "backhaul_order"
    ]
    assert len(rows) == 1 and rows[0]["declared"] is False


@pytest.mark.parametrize("variant", ["pdtsp", "pdcvrp"])
def test_relation_columns_come_from_the_declaration_not_the_kernel(
    variant: str,
) -> None:
    """The relation node term reads the published PRECEDENCE row.

    The compiled pickup-delivery kernel used to supply the roles through its own
    branch alongside the declared path. It publishes exactly the PRECEDENCE spec
    that branch supplied, so both must produce bit-identical node attributes.
    """
    torch.manual_seed(5)
    problem = generated_problem(variant, 20)
    compiled = _decoder(problem)
    declared = prism_decoder.Decoder(_declared_twin(problem, "pickup_delivery"))

    compiled_flags = _relation_rows(compiled)
    declared_flags = _relation_rows(declared)

    assert np.array_equal(compiled_flags, declared_flags)
    # Every pair contributes exactly one opener and one requirer.
    pairs = (problem["coordinates"].shape[0] - problem["depot_count"]) // 2
    assert compiled_flags[:, RELATION_OPENS_SLOT].sum() == pairs
    assert compiled_flags[:, RELATION_REQUIRES_SLOT].sum() == pairs
    assert compiled_flags[:, 1].sum() == pairs


def test_incumbent_suffix_state_accumulates_and_resets_per_route() -> None:
    """An accumulating row reports what its own route still spends.

    Ordering rows (backhaul_order) legitimately have no accumulated quantity and
    stay zero; capacity is the row with a node term to check.
    """
    torch.manual_seed(21)
    problem = generated_problem("cvrp", 30, capacity=12)
    solver = _decoder(problem)
    solver.seed(21)
    route = [int(node) for node in solver.solve(16)["route"]]
    solver.set_incumbent(route)

    suffix = np.asarray(solver.incumbent_suffix_state)[:, 0]
    assert suffix.std() > 1e-6
    assert np.all(suffix >= 0.0) and np.all(suffix <= 1.0)
    # Within ONE route the remainder is non-increasing as the route advances.
    # Across routes it is not: each leg restarts from its own total.
    depots = problem["depot_count"]
    leg: list[int] = []
    for node in route[1:]:
        if node < depots:
            if len(leg) > 1:
                break
            leg = []
            continue
        leg.append(node)
    assert len(leg) > 1
    values = [float(suffix[node]) for node in leg]
    assert values == sorted(values, reverse=True)


def test_suffix_state_includes_departure_terms_and_publishes_them_separately() -> None:
    """Time-window suffixes include all remaining service, not travel alone."""
    coordinates = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        dtype=np.float32,
    )
    distance = np.abs(
        np.arange(4, dtype=np.float32)[:, None]
        - np.arange(4, dtype=np.float32)[None, :]
    )
    solver = _decoder(
        {
            "name": "cvrptw",
            "coordinates": coordinates,
            "distance": distance,
            "demand": np.zeros(4, dtype=np.float32),
            "capacity": 20.0,
            "tw_start": np.zeros(4, dtype=np.float32),
            # Node 1 is reachable on the incumbent but not when revisited from
            # its late prefix at node 3. This exercises the sign of the exact
            # behavioral margin without making the incumbent infeasible.
            "tw_end": np.array([10.0, 2.0, 10.0, 10.0], dtype=np.float32),
            "service_time": np.array([0.0, 0.5, 0.75, 1.0], dtype=np.float32),
        }
    )
    solver.set_incumbent(np.array([0, 1, 2, 3, 0], dtype=np.int32))
    row = _row(solver, "time_window")
    suffix = np.asarray(solver.incumbent_suffix_state)[:, row]
    features = np.asarray(solver.incumbent_suffix_features)[:, row]

    # time scale is ten. Remaining totals are service(node) + outgoing travel
    # plus the later nodes' same quantities.
    assert suffix[3] == pytest.approx(4.0 / 10.0)
    assert suffix[2] == pytest.approx(5.75 / 10.0)
    assert suffix[1] == pytest.approx(7.25 / 10.0)
    np.testing.assert_allclose(features[[1, 2, 3], 0], suffix[[1, 2, 3]])
    np.testing.assert_allclose(
        features[[1, 2, 3], 2], np.array([2.25, 1.75, 1.0]) / 10.0
    )
    assert np.count_nonzero(features[:, 1]) == 0
    assert np.count_nonzero(features[:, 3]) == 0

    edge_index = np.asarray(solver.edge_index)
    edge = np.flatnonzero(
        (edge_index[0] == 1) & (edge_index[1] == 2)
    ).item()
    features = np.asarray(solver.incumbent_transition_features)
    mask = np.asarray(solver.incumbent_transition_feature_mask)
    assert mask[edge, row]
    # Prefix at node 1 is 1.5; travel to node 2 gives bounded arrival 2.5,
    # then service gives post-transition state 3.25. The binding admissibility
    # margin is the 4.75 slack after projecting the required depot return.
    np.testing.assert_allclose(
        features[edge, row], np.array([0.325, 0.475]), atol=1e-6
    )
    violating_edge = np.flatnonzero(
        (edge_index[0] == 3) & (edge_index[1] == 1)
    ).item()
    assert mask[violating_edge, row]
    assert features[violating_edge, row, 1] < 0.0


def test_suffix_features_preserve_opposing_resource_increments() -> None:
    """A signed suffix scalar may cancel, but the generic split must not."""
    problem = generated_problem("cvrpb", 2)
    problem["demand"] = np.array([0.0, 0.2, -0.2], dtype=np.float32)
    solver = _decoder(problem)
    solver.set_incumbent(np.array([0, 1, 2, 0], dtype=np.int32))
    row = _row(solver, "capacity")

    scalar = np.asarray(solver.incumbent_suffix_state)[:, row]
    features = np.asarray(solver.incumbent_suffix_features)[:, row]
    # Capacity declares increment=-demand. At node 1 the +2 and -2 cancel in
    # the legacy signed total, while the split retains both 20%-of-capacity
    # workloads for a descriptor-conditioned encoder to use.
    assert scalar[1] == pytest.approx(0.0, abs=1e-7)
    assert features[1, 0] == pytest.approx(0.2)
    assert features[1, 1] == pytest.approx(0.2)


# The channel ordinal used to be a third parameter here. It was the row's
# registry position only while every problem carried one row per channel in
# channel order; both sides now resolve their row by name.
@pytest.mark.parametrize(
    ("variant", "constraint"),
    [
        ("cvrp", "capacity"),
        ("cvrptw", "time_windows"),
        ("cvrpl", "route_limit"),
        ("op", "tour_limit"),
        ("cvrpbp", "backhaul_order"),
        ("pdcvrp", "pickup_delivery"),
        ("pctsp", "prize_quota"),
    ],
)
def test_live_state_comes_from_the_declaration_not_the_kernel(
    variant: str, constraint: str
) -> None:
    """Per-node live state must not depend on which path executes a constraint.

    A compiled kernel keeps its own named scalars (`load`, `current_time`, ...),
    so its live state used to be a hand-written formula per field channel. Its
    declared state is now mirrored into the same generic vector a declared row
    uses, and the feature is read from there.
    """
    torch.manual_seed(11)
    problem = generated_problem(variant, 30, capacity=12)
    twin = _declared_twin(problem, constraint)

    compiled = _decoder(problem)
    compiled.seed(11)
    route = np.array(
        [int(node) for node in compiled.solve(16)["route"]], dtype=np.int32
    )
    compiled.set_incumbent(route)
    declared = prism_decoder.Decoder(twin)
    declared.set_incumbent(route)

    kernel = KERNEL_OF_CONSTRAINT[constraint]
    compiled_row = _row(compiled, kernel)
    declared_row = _row(declared, f"declared_{kernel}")
    compiled_state = np.asarray(compiled.incumbent_live_state)[:, compiled_row]
    declared_state = np.asarray(declared.incumbent_live_state)[:, declared_row]
    assert np.allclose(compiled_state, declared_state, atol=1e-5)

    # The reverse pass is declaration-driven the same way, so a declared row
    # gets the suffix quantity the removed backward_* node columns only ever
    # gave three compiled kernels.
    compiled_suffix = np.asarray(compiled.incumbent_suffix_state)[:, compiled_row]
    declared_suffix = np.asarray(declared.incumbent_suffix_state)[:, declared_row]
    assert np.allclose(compiled_suffix, declared_suffix, atol=1e-5)


@pytest.mark.parametrize(
    ("variant", "constraint"),
    [
        ("cvrp", "capacity"),
        ("cvrptw", "time_windows"),
        ("pctsp", "prize_quota"),
    ],
)
def test_node_attributes_come_from_the_declaration_not_a_named_slot(
    variant: str, constraint: str
) -> None:
    """Demand, window bounds, and service time no longer have node-feature slots.

    They are per-node attributes of a resource, so they live in
    node_resource_features indexed by row. A declared twin must produce the same
    block the kernel it replaces does.
    """
    torch.manual_seed(11)
    problem = generated_problem(variant, 30, capacity=12)
    twin = _declared_twin(problem, constraint)
    kernel = KERNEL_OF_CONSTRAINT[constraint]
    compiled_solver = _decoder(problem)
    declared_solver = prism_decoder.Decoder(twin)
    compiled = np.asarray(compiled_solver.node_resource_features)
    declared = np.asarray(declared_solver.node_resource_features)
    compiled_row = _row(compiled_solver, kernel)
    declared_row = _row(declared_solver, f"declared_{kernel}")
    assert np.allclose(
        compiled[:, compiled_row], declared[:, declared_row], atol=1e-5
    )
    assert np.abs(compiled[:, compiled_row]).sum() > 0.0

    names = list(prism_decoder.NODE_FEATURE_NAMES)
    for retired in ("linehaul_demand", "window_start", "window_end", "service_time"):
        assert retired not in names


def test_node_attributes_survive_an_abstaining_kernel() -> None:
    """Abstention is about the extension rule, never the per-node quantities.

    Capacity abstains under a declared class order, but the demands themselves
    are still exactly what the kernel uses, so the node attributes must stay
    populated.
    """
    torch.manual_seed(11)
    solver = prism_decoder.Decoder(_declared_class_order_problem(30, 12))
    rows = [
        row
        for row in solver.resource_declarations
        if row["active"] and row["kernel"] == "capacity"
    ]
    assert len(rows) == 1 and rows[0]["declared"] is False

    attributes = np.asarray(solver.node_resource_features)[:, 0]
    assert np.abs(attributes[:, :2]).sum() > 0.0


def test_an_unseen_resource_reaches_the_node_inputs() -> None:
    """A battery row's per-node attributes are visible like any other row's."""
    from prism_eval import instances

    torch.manual_seed(3)
    data = instances.GENERATORS["evrp"](30, 1, seed=3)
    problem = instances.solver_problem("evrp", instances._instance_data(data, 0))
    solver = prism_decoder.Decoder(problem)
    attributes = np.asarray(solver.node_resource_features)
    assert attributes.shape[1] == len(np.asarray(solver.resource_scales))
    # The battery row is the appended one; its bound channel is its own.
    assert attributes.shape[2] == prism_decoder.NODE_RESOURCE_FEATURE_COUNT
    assert np.isfinite(attributes).all()
    assert (attributes >= 0.0).all() and (attributes <= 1.0).all()


def _backhaul_instance(constraints: list[str]) -> dict:
    # Two linehauls (+0.6) and two backhauls (-0.4) against a capacity of 1.0.
    demand = np.array([0.0, 0.6, 0.6, -0.4, -0.4], dtype=np.float32)
    coordinates = np.array(
        [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32
    )
    size = demand.shape[0]
    return {
        "name": "backhaul",
        "constraints": constraints,
        "objective": "distance",
        "depot_count": 1,
        "multi_route": True,
        "open_route": False,
        "coordinates": coordinates,
        "capacity": 1.0,
        "demand": demand,
        "prize": np.zeros(size, np.float32),
        "penalty": np.zeros(size, np.float32),
        "tw_start": np.zeros(size, np.float32),
        "tw_end": np.full(size, 1e9, np.float32),
        "service_time": np.zeros(size, np.float32),
    }


def test_backhaul_opening_load_matches_the_benchmark_definition() -> None:
    """Two distinct rules, and only one of them applies to plain backhaul.

    URS (`baselines/URS/env/UniVRPEnv.py`) resets the load to zero in two
    places, and PRISM has to agree with both or it is solving a different
    problem from the references it is compared against:

      * line 550, for `b` and `bp` alike: at the depot with no linehaul left
        anywhere, start empty. PRISM's `depot_reload`.
      * line 517, for `bp` only: a route that *opens* on a backhaul starts
        empty, whether or not linehauls remain elsewhere. PRISM's
        `opening_load`.

    Without the second rule PRISM rejected solutions the benchmark admits, so a
    reported gap on the `*bp` family would have mixed a modelling difference in
    with solver quality.
    """
    # A backhaul-only route scheduled first, while linehauls are still unserved.
    opens_on_backhaul = [0, 3, 4, 0, 1, 0, 2, 0]

    plain = prism_decoder.Decoder(_backhaul_instance(["visit_all", "capacity"]))
    priority = prism_decoder.Decoder(
        _backhaul_instance(["visit_all", "capacity", "backhaul_order"])
    )
    assert plain.evaluate(opens_on_backhaul)["feasible"] is False
    assert priority.evaluate(opens_on_backhaul)["feasible"] is True

    # The global rule still applies to both: once every linehaul is served, a
    # trailing backhaul-only route starts empty.
    linehauls_first = [0, 1, 0, 2, 0, 3, 4, 0]
    assert plain.evaluate(linehauls_first)["feasible"] is True
    assert priority.evaluate(linehauls_first)["feasible"] is True


def _capacity_live_state(variant: str) -> tuple[list[float], list[float]]:
    """Live capacity state along a solution, and the remaining load it should be."""
    torch.manual_seed(0)
    problem = generated_problem(variant, 20, capacity=12)
    solver = _decoder(problem)
    solver.seed(0)
    route = [int(node) for node in solver.solve(16)["route"]]
    solver.set_incumbent(np.array(route, dtype=np.int32))
    live = np.asarray(solver.incumbent_live_state)[:, 0]

    demand = np.asarray(problem["demand"])
    served: dict[int, float] = {}
    total = 0.0
    for node in route:
        total = 0.0 if node == 0 else total + float(demand[node])
        served[node] = total
    customers = [node for node in route if node > 0][:6]
    return (
        [float(live[node]) for node in customers],
        [1.0 - served[node] for node in customers],
    )


def test_abstaining_rows_still_describe_themselves_to_the_model() -> None:
    """`declared: false` must not change what a channel *means*.

    Abstention says the published row does not reproduce the kernel exactly. It
    is not licence to feed the model something different: the descriptor encodes
    the reset *form* rather than its value, and the live-state feature is a
    position within the declared bounds, so neither depends on the part that
    escapes the language. Gating them on exactness made the capacity channel read
    `remaining/capacity` on cvrp and `served/capacity` on cvrpb -- the same slot
    meaning opposite things on 68 of the 110 variants.
    """
    for variant in ("cvrp", "cvrpb"):
        live, remaining = _capacity_live_state(variant)
        assert live == pytest.approx(remaining, abs=1e-3), variant

    # A declared class order still abstains, and still publishes its algebra to
    # every consumer.
    torch.manual_seed(0)
    solver = prism_decoder.Decoder(_declared_class_order_problem(20, 12))
    rows = [
        row
        for row in solver.resource_declarations
        if row["active"] and row["kernel"] == "capacity"
    ]
    assert len(rows) == 1 and rows[0]["declared"] is False

    torch.manual_seed(0)
    plain = generated_problem("cvrp", 20, capacity=12)
    plain_program = _program_row(_decoder(plain), 0)
    backhaul_program = _program_row(solver, 0)
    # Identical: |demand| is unchanged by negating a fifth of the customers, so
    # even the quantitative slots agree. The abstaining row is described to the
    # model exactly as the declared one is.
    assert np.allclose(plain_program[0], backhaul_program[0], atol=1e-6)
    assert np.allclose(plain_program[1], backhaul_program[1], atol=1e-6)


# Solve-level equivalence cannot see a pricing divergence: both sides of that
# comparison read the same published algebra, so a row that prices wrongly
# prices wrongly twice and the two solvers still agree. The pressure a row
# publishes is a model input in its own right, and it needs its own reference.
_PRESSURE_VARIANTS = (
    "cvrp", "cvrptw", "cvrpl", "ocvrp", "mdcvrp", "acvrp",
    "pdtsp", "pdcvrp", "pctsp", "op", "cvrpbp", "ocvrpbp", "mdcvrpbp",
)


@pytest.mark.parametrize("variant", _PRESSURE_VARIANTS)
def test_published_pressure_is_never_dead_where_the_kernel_is_live(
    variant: str,
) -> None:
    """An active row may not price every edge to zero.

    This is the failure the declarative rewrite introduced: pricing from the
    published row rather than the kernel sent `capacity` (0 <= load <= cap,
    consumed downward) down an upper-bound-only branch, so every linehaul
    customer exerted zero pressure, and sent `prize_quota` (accumulating upward
    toward a floor) down a rule that is identically zero for it. Both channels
    reached the model as constants while still reporting themselves active.
    """
    torch.manual_seed(4)
    solver = _decoder(generated_problem(variant, 40, 4))
    published = np.asarray(solver.resource_pressure)
    compiled = np.asarray(solver.compiled_resource_pressure)
    # The registry holds exactly the rows this problem declares, so there is no
    # inactive row to skip; this used to walk all seven channels and filter by
    # the active mask.
    assert np.all(np.asarray(solver.metadata["field_channel_mask"]) == 1)
    for index, name in enumerate(_registry(solver)):
        if compiled[:, index].max() <= 0.0:
            continue
        assert published[:, index].max() > 0.0, (
            f"{variant}: active row {name} prices every edge to zero while its "
            "compiled kernel is live"
        )


@pytest.mark.parametrize("variant", _PRESSURE_VARIANTS)
def test_published_pressure_matches_the_compiled_kernel(variant: str) -> None:
    """Every published row prices exactly what its kernel prices.

    Two things used to break this for `backhaul_order`, and both were real. The
    depot's class is zero -- the lowest -- because published_algebra classifies
    only customers, so every backhaul->depot arc read as a class descent and
    carried pressure on the ordinary way a backhaul route closes. And the share
    was taken over the nonzero classes alone, which made one misordered arc read
    twenty times larger at a five percent backhaul share than at sixty, scaling
    a violation by the instance's composition rather than by its severity.
    """
    torch.manual_seed(4)
    solver = _decoder(generated_problem(variant, 40, 4))
    published = np.asarray(solver.resource_pressure)
    compiled = np.asarray(solver.compiled_resource_pressure)
    for index, name in enumerate(_registry(solver)):
        assert np.allclose(
            published[:, index], compiled[:, index], atol=1e-6
        ), f"{variant}: published {name} pressure diverges from its kernel"


def test_class_order_pressure_does_not_scale_with_instance_composition() -> None:
    """The same violation must price the same at any backhaul share.

    The constrained fraction is real information, but it belongs in the
    descriptor, which still reports it. Pressure is the severity of one edge's
    misordering and has to be independent of how much of the instance happens to
    be backhaul.
    """
    rng = np.random.default_rng(7)
    size = 40
    coordinates = rng.random((size + 1, 2), dtype=np.float32)
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    pressures = []
    fractions = []
    for share in (0.05, 0.2, 0.6):
        demand = rng.uniform(0.01, 0.09, size + 1).astype(np.float32)
        demand[0] = 0.0
        demand[1 : 1 + max(1, int(size * share))] *= -1
        solver = _decoder(
            problem_schema("cvrpbp")
            | {
                "coordinates": coordinates,
                "distance": distance,
                "demand": demand,
                "capacity": 1.0,
            }
        )
        # Resolved per solver: the row's position is a registry position, and
        # the registry now holds only what this instance declares.
        backhaul = _row(solver, "backhaul_order")
        pressures.append(
            float(np.asarray(solver.resource_pressure)[:, backhaul].max())
        )
        fractions.append(
            float(
                np.asarray(solver.resource_row_properties)[
                    backhaul, RELATION_DENSITY_PROPERTY
                ]
            )
        )

    assert pressures[0] == pytest.approx(1.0 / size, rel=1e-5)
    assert all(value == pytest.approx(pressures[0], rel=1e-5) for value in pressures)
    # The composition itself stays visible, just not through the pressure.
    assert fractions[0] < fractions[1] < fractions[2]


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("tsp", []),
        ("cvrp", ["capacity"]),
        ("ocvrp", ["capacity"]),
        ("cvrptw", ["capacity", "time_window"]),
        ("cvrpl", ["capacity", "route_limit"]),
        ("op", ["tour_limit"]),
        ("pctsp", ["prize_quota"]),
        ("cvrpbp", ["capacity", "backhaul_order"]),
        ("pdcvrp", ["capacity", "pickup_delivery"]),
        ("mdcvrptw", ["capacity", "time_window"]),
    ],
)
def test_the_registry_holds_exactly_the_declared_constraints(
    variant: str, expected: list[str]
) -> None:
    """A row exists because the instance declares it, not because it could.

    The registry used to open with one row per compiled field channel on every
    problem, active or not, so a TSP carried seven rows it never used and an
    appended row always landed at position seven. Position is now a property of
    the instance, which is why nothing may resolve a row by a global ordinal.
    """
    torch.manual_seed(3)
    solver = _decoder(generated_problem(variant, 24))
    assert _registry(solver) == expected
    assert solver.metadata["resource_count"] == len(expected)
    assert solver.metadata["multiplier_count"] == len(expected) + 1
    # Every row present is a row the instance enforces.
    assert np.all(np.asarray(solver.metadata["field_channel_mask"]) == 1)


def test_a_declared_row_is_supervised_like_a_compiled_one() -> None:
    """Screening labels must be as wide as the field head they supervise.

    They were emitted at FIELD_CHANNEL_COUNT while the field head is one column
    per registry row, so an appended row got no resource supervision at all and
    the auxiliary resource loss could not even be formed: the label and the
    prediction disagreed on width. A row that binds must draw label mass.
    """
    torch.manual_seed(3)
    problem = generated_problem("cvrp", 30)
    explicit = problem_schema("cvrp")
    explicit.update(problem)
    explicit["node_attributes"] = {
        "extra": np.asarray(problem["demand"], dtype=np.float32)
    }
    explicit["resources"] = [
        {
            "name": "extra_budget",
            "operator": "affine_accumulator",
            "scope": "route",
            "direction": "forward",
            "initial": 0.0,
            "scale": 1.0,
            "increment": {"node_attribute": "extra", "coefficient": 1.0},
            "bounds": [{"upper": 1.0}],
            "reset": {"at_depot": True, "value": 0.0},
        }
    ]
    solver = prism_decoder.Decoder(explicit)
    solver.seed(3)
    registry = solver.metadata["resource_count"]
    assert _registry(solver) == ["capacity", "extra_budget"]

    guidance = _neutral_guidance(solver)
    feasible = [s for s in solver.sample(**guidance) if s["feasible"]]
    assert feasible
    solver.set_incumbent(min(feasible, key=lambda s: s["objective"])["route"])
    trace = solver.sample_traced(**guidance)["trace"]

    delta = np.asarray(trace["screened_resource_delta"])
    assert delta.shape == (np.asarray(trace["screened_edges"]).shape[0], registry)
    declared = _row(solver, "extra_budget")
    assert delta[:, declared].sum() > 0.0
    assert np.asarray(
        trace["screening_verification_failures_by_channel"]
    ).shape == (registry,)


def test_a_compiled_row_and_its_declared_twin_describe_themselves_alike() -> None:
    """The descriptor must not say which path executes a row.

    A row now carries its published algebra directly, so a compiled kernel and a
    declared row stating the same algebra produce the same descriptor. The last
    slot keyed on a constraint (`op == PICKUP_DELIVERY`) is gone rather than
    merely unreachable.
    """
    for variant, constraint in (
        ("cvrp", "capacity"),
        ("cvrptw", "time_windows"),
        ("pdcvrp", "pickup_delivery"),
        ("cvrpbp", "backhaul_order"),
    ):
        torch.manual_seed(11)
        problem = generated_problem(variant, 30, capacity=12)
        twin = _declared_twin(problem, constraint)
        compiled = _decoder(problem)
        declared = prism_decoder.Decoder(twin)
        kernel = KERNEL_OF_CONSTRAINT[constraint]
        compiled_row = _program_row(compiled, _row(compiled, kernel))
        declared_row = _program_row(
            declared, _row(declared, f"declared_{kernel}")
        )
        assert np.allclose(compiled_row[0], declared_row[0], atol=1e-6), variant
        assert np.allclose(compiled_row[1], declared_row[1], atol=1e-6), variant


def test_a_row_reports_the_kernel_that_executes_it_not_its_algebra() -> None:
    """`kernel` and `operator` answer different questions.

    They used to be one field: a row's operator named a constraint, which is
    what made the operator vocabulary grow with the constraint vocabulary. The
    operator is now always a language operator, and the kernel -- possibly none
    -- is reported separately.
    """
    torch.manual_seed(3)
    solver = _decoder(generated_problem("cvrptw", 24))
    rows = {row["name"]: row for row in solver.resource_declarations}
    assert rows["capacity"]["kernel"] == "capacity"
    assert rows["capacity"]["operator"] == "affine_accumulator"
    assert rows["time_window"]["kernel"] == "time_window"
    assert rows["time_window"]["operator"] == "affine_accumulator"
    # The tropical family is gone from the operator vocabulary: what made a time
    # window different from a plain accumulator is its semiring, not its operator.
    assert rows["time_window"]["semiring"] == "max_plus"
    assert rows["capacity"]["semiring"] == "arithmetic"

    twin = _declared_twin(generated_problem("cvrp", 24), "capacity")
    declared = prism_decoder.Decoder(twin)
    row = next(
        r for r in declared.resource_declarations if r["name"] == "declared_capacity"
    )
    # Interpreted: no compiled kernel executes it, but it says the same thing
    # the capacity row says.
    assert row["kernel"] == ""
    assert row["operator"] == "affine_accumulator"


# ---------------------------------------------------------------------------
# Semiring: the join is declared, not implied by the operator
# ---------------------------------------------------------------------------
def _joined_problem(
    upper: float, semiring: str | None = None, operand: float = 0.0
) -> dict:
    """One affine row accumulating distance under an upper bound.

    The bound is set where some rollouts overrun it, so the join is observable
    through feasibility alone: a capping join keeps every rollout inside the
    bound, a raising one pushes all of them out. No state readout needed.
    """
    torch.manual_seed(11)
    problem = generated_problem("cvrp", 12)
    explicit = problem_schema("cvrp")
    explicit.update(problem)
    explicit["constraints"] = []
    row: dict = {
        "name": "joined",
        "operator": "affine_accumulator",
        "increment": {"edge_attribute": "distance", "coefficient": 1.0},
        "bounds": [{"upper": upper, "check": "transition"}],
        "scope": "route",
    }
    if semiring is not None:
        row["semiring"] = semiring
        row["join"] = {"value": operand}
    explicit["resources"] = [row]
    return explicit


def _every_rollout_feasible(problem: dict) -> bool:
    solver = prism_decoder.Decoder(problem, n_rollouts=4, beta=2.0)
    solver.seed(7)
    return all(
        solution["feasible"]
        for solution in solver.sample(**_neutral_guidance(solver))
    )


def test_min_plus_caps_the_accumulated_value():
    """A min_plus row cannot climb past its operand; an unjoined one can.

    This is capability the operator vocabulary could not express: `affine_max`
    named one specific join, so a capping row had nowhere to be declared.
    """
    assert not _every_rollout_feasible(_joined_problem(0.6))
    assert _every_rollout_feasible(_joined_problem(0.6, "min_plus", 0.6)), (
        "capping at the bound should keep every rollout inside it"
    )


def test_max_plus_raises_the_accumulated_value():
    """The dual: an operand above the bound puts every rollout outside it."""
    assert _every_rollout_feasible(_joined_problem(2.0, "max_plus", 0.0)) is False
    solver_problem = _joined_problem(2.0, "max_plus", 5.0)
    solver = prism_decoder.Decoder(solver_problem, n_rollouts=4, beta=2.0)
    solver.seed(7)
    assert not any(
        solution["feasible"]
        for solution in solver.sample(**_neutral_guidance(solver))
    )


def test_arithmetic_row_may_not_declare_a_join_operand():
    """An operand with no join to consume it is a contradiction, not a default."""
    problem = _joined_problem(0.6, "arithmetic", 0.6)
    with pytest.raises(ValueError, match="only a tropical semiring"):
        prism_decoder.Decoder(problem)


def test_affine_max_is_the_max_plus_semiring():
    """The retired spelling still parses, and means exactly what it meant."""
    named = _joined_problem(2.0)
    named["resources"][0]["operator"] = "affine_max"
    named["resources"][0]["clamp"] = {"value": 5.0}
    declared = _joined_problem(2.0, "max_plus", 5.0)
    rows = [
        prism_decoder.Decoder(problem).resource_declarations[-1]
        for problem in (named, declared)
    ]
    assert rows[0]["operator"] == rows[1]["operator"] == "affine_accumulator"
    assert rows[0]["semiring"] == rows[1]["semiring"] == "max_plus"
    joins = [
        [term for term in row["terms"] if term["op"] == "join"]
        for row in rows
    ]
    assert len(joins[0]) == len(joins[1]) == 1
    assert list(joins[0][0]["values"]) == list(joins[1][0]["values"])


def test_join_and_clamp_are_one_field():
    """Declaring both spellings is a contradiction rather than a merge."""
    problem = _joined_problem(2.0, "max_plus", 5.0)
    problem["resources"][0]["clamp"] = {"value": 2.0}
    with pytest.raises(ValueError, match="not both"):
        prism_decoder.Decoder(problem)


def test_a_row_may_state_its_extension_in_terms() -> None:
    """The parser's contract is the term list; the named fields are shorthand.

    Only the instance generators should speak in named constraint concepts. A
    declaration states coordinates -- what it reads, where, which operation,
    phase, trigger, and gate consume it -- and the two spellings must agree exactly, or the shorthand is
    quietly a second semantics.
    """
    torch.manual_seed(5)
    problem = generated_problem("cvrp", 24, 12)
    explicit = problem_schema("cvrp")
    explicit.update(problem)
    explicit["constraints"] = []
    explicit["node_attributes"] = {
        "load": np.asarray(explicit["demand"], dtype=np.float32)
    }
    bounds = [{"lower": 0.0, "upper": 1.0, "check": "transition"}]
    shorthand = {
        "name": "cap",
        "operator": "affine_accumulator",
        "increment": {"node_attribute": "load", "coefficient": -1.0},
        "reset": {"value": 1.0, "at_depot": True},
        "initial": 1.0,
        "bounds": bounds,
    }
    terms = {
        "name": "cap",
        "operator": "affine_accumulator",
        "terms": [
            {
                "node_attribute": "load",
                "coefficient": -1.0,
                "at": "to",
                "op": "add",
                "phase": "before_bound",
            },
            {
                "value": 1.0,
                "op": "assign",
                "phase": "after_bound",
                "when": "reset_arrival",
                "at_depot": True,
            },
        ],
        "initial": 1.0,
        "bounds": bounds,
    }
    solutions = []
    for row in (shorthand, terms):
        candidate = dict(explicit)
        candidate["resources"] = [row]
        solver = prism_decoder.Decoder(candidate, n_rollouts=4, beta=2.0)
        solver.seed(9)
        best = solver.solve(4, **_neutral_guidance(solver))
        published = solver.resource_declarations[-1]
        solutions.append(
            (
                round(float(best["objective"]), 6),
                list(best["route"]),
                [
                    (
                        t["op"], t["phase"], t["when"], t["at"],
                        t["source"], t["coefficient"],
                    )
                    for t in published["terms"]
                ],
                np.asarray(solver.resource_row_properties).tolist(),
                np.asarray(solver.resource_term_properties).tolist(),
            )
        )
    assert solutions[0] == solutions[1]
    # And the shorthand really did lower to a term, rather than both being empty.
    assert solutions[0][2] == [
        ("add", "before_bound", "always", "to", "node_attribute", -1.0),
        ("assign", "after_bound", "reset_arrival", "to", "value", 1.0),
    ]


def test_reset_is_an_assign_term() -> None:
    """Reset firing and installed value are ordinary term coordinates."""
    torch.manual_seed(5)
    problem = generated_problem("cvrp", 12, 12)
    explicit = problem_schema("cvrp")
    explicit.update(problem)
    explicit["constraints"] = []
    explicit["node_attributes"] = {
        "load": np.asarray(explicit["demand"], dtype=np.float32)
    }
    explicit["resources"] = [
        {
            "name": "cap",
            "operator": "affine_accumulator",
            "terms": [
                {"node_attribute": "load", "op": "add"},
                {
                    "value": 0.0,
                    "op": "assign",
                    "phase": "after_bound",
                    "when": "reset_arrival",
                    "at_depot": True,
                },
            ],
            "bounds": [{"upper": 1.0, "check": "transition"}],
        }
    ]
    terms = prism_decoder.Decoder(explicit).resource_declarations[-1]["terms"]
    assert any(
        term["op"] == "assign" and term["when"] == "reset_arrival"
        for term in terms
    )


def test_terms_and_the_shorthand_may_not_be_mixed() -> None:
    """They would sum rather than conflict, which is the worse failure."""
    torch.manual_seed(5)
    problem = generated_problem("cvrp", 12, 12)
    explicit = problem_schema("cvrp")
    explicit.update(problem)
    explicit["constraints"] = []
    explicit["node_attributes"] = {
        "load": np.asarray(explicit["demand"], dtype=np.float32)
    }
    explicit["resources"] = [
        {
            "name": "cap",
            "operator": "affine_accumulator",
            "terms": [{"node_attribute": "load"}],
            "increment": {"node_attribute": "load"},
            "bounds": [{"upper": 1.0, "check": "transition"}],
        }
    ]
    with pytest.raises(ValueError, match="not both"):
        prism_decoder.Decoder(explicit)


@pytest.mark.parametrize(
    ("term", "message"),
    [
        (
            {"value": 0.0, "op": "assign", "phase": "before_bound"},
            "assign term must run after",
        ),
        (
            {"value": 0.0, "op": "checkpoint", "phase": "before_bound"},
            "checkpoint term must run after",
        ),
        (
            {"op": "restore", "phase": "after_bound", "when": "bound_failure"},
            "restore term must run before",
        ),
        (
            {"value": 0.0, "op": "assign", "phase": "after_bound",
             "when": "reset_arrival"},
            "requires at_depot or at_nodes",
        ),
    ],
)
def test_invalid_term_coordinate_combinations_are_rejected(
    term: dict, message: str
) -> None:
    problem = _accumulator_probe(10, cap_at_depot=100.0)
    problem["resources"][0].pop("increment", None)
    problem["resources"][0].pop("reset", None)
    problem["resources"][0]["terms"] = [term]
    with pytest.raises(ValueError, match=message):
        prism_decoder.Decoder(problem)


def test_term_sets_reject_order_dependent_overlapping_assigns() -> None:
    problem = _accumulator_probe(10, cap_at_depot=100.0)
    row = problem["resources"][0]
    row.pop("increment")
    row["terms"] = [
        {"value": value, "op": "assign", "phase": "after_bound",
         "when": "reset_arrival", "at_depot": True}
        for value in (0.0, 1.0)
    ]
    with pytest.raises(ValueError, match="depend on term order"):
        prism_decoder.Decoder(problem)
