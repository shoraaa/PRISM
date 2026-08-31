from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "test.py"
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("decoder_evaluation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
decoder_evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(decoder_evaluation)

from prism_eval import instances, methods, runner, summary  # noqa: E402
from prism_eval.methods import oracle  # noqa: E402
from prism_eval.methods.base import InProcessMethod, MethodRequest  # noqa: E402
from prism_eval.methods.urs import UrsMethod  # noqa: E402
from prism_eval.methods.ccl import CclMethod  # noqa: E402
from prism_eval import results  # noqa: E402
from prism_eval.results import Row  # noqa: E402


def test_compare_defaults_to_eight_dynamic_instances() -> None:
    args = decoder_evaluation.parse_args(["--checkpoint", "model.pt"])

    assert args.val_size == 8
    assert args.static_field is False
    assert args.variants == "all"
    assert args.tsptw_source == "dataset"
    assert args.tsptw_size == 100
    assert args.tsptw_hardness == "hard"
    assert args.tsptw_dataset_seed == 2025
    assert args.methods == "constant"
    assert args.min_changed_edges == 8
    assert args.random_escape is False
    assert args.n_node is None
    assert args.vrpdb_size == 100
    assert args.aug is None
    assert args.feasibility_risk_penalty is None


def test_compare_accepts_matched_risk_guidance_ablation() -> None:
    args = decoder_evaluation.parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--feasibility-risk-penalty",
            "0",
        ]
    )

    assert args.feasibility_risk_penalty == 0.0


def test_shared_augmentation_override_reaches_constructive_baselines() -> None:
    args = decoder_evaluation.parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--aug",
            "1",
            "--urs-baseline-id",
            "urs.pt",
        ]
    )

    assert args.aug == 1
    urs = UrsMethod(args)
    assert "aug=1" in urs.config()
    assert "aug=1" in CclMethod(args).config()
    assert not runner._compatible_config(
        urs, "checkpoint=urs.pt,aug=on,batch=50"
    )


def test_shared_augmentation_override_must_be_positive() -> None:
    with pytest.raises(SystemExit):
        decoder_evaluation.parse_args(
            ["--checkpoint", "model.pt", "--aug", "0"]
        )


def test_compare_n_node_enables_generated_benchmarks() -> None:
    dashed = decoder_evaluation.parse_args(
        ["--checkpoint", "model.pt", "--n-node", "500"]
    )
    underscored = decoder_evaluation.parse_args(
        ["--checkpoint", "model.pt", "--n_node", "500"]
    )

    assert dashed.n_node == 500
    assert underscored.n_node == 500


def test_compare_rejects_nonpositive_generated_size() -> None:
    with pytest.raises(SystemExit):
        decoder_evaluation.parse_args(
            ["--checkpoint", "model.pt", "--n-node", "0"]
        )


def test_resume_requires_an_existing_csv(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        decoder_evaluation.parse_args(
            ["--checkpoint", "model.pt", "--resume"]
        )
    with pytest.raises(SystemExit):
        decoder_evaluation.parse_args(
            [
                "--checkpoint",
                "model.pt",
                "--resume",
                "--csv",
                str(tmp_path / "missing.csv"),
            ]
        )

    path = tmp_path / "partial.csv"
    path.write_text(",".join(results.FIELDS) + "\n")
    args = decoder_evaluation.parse_args(
        ["--checkpoint", "model.pt", "--resume", "--csv", str(path)]
    )
    assert args.resume is True


def test_generated_benchmark_batch_is_reproducible() -> None:
    first = instances.generate_benchmark_problems(
        "cvrptw", size=12, count=2, seed=123
    )
    torch.rand(20)
    second = instances.generate_benchmark_problems(
        "cvrptw", size=12, count=2, seed=123
    )

    assert len(first) == 2
    assert first[0]["coordinates"].shape == (13, 2)
    for lhs, rhs in zip(first, second):
        assert torch.equal(
            torch.from_numpy(lhs["coordinates"]),
            torch.from_numpy(rhs["coordinates"]),
        )
        assert torch.equal(
            torch.from_numpy(lhs["tw_start"]),
            torch.from_numpy(rhs["tw_start"]),
        )


def test_compare_min_changed_edges_cli_override() -> None:
    args = decoder_evaluation.parse_args(
        ["--checkpoint", "model.pt", "--min-changed-edges", "6"]
    )

    assert args.min_changed_edges == 6


def test_compare_rejects_nonpositive_min_changed_edges() -> None:
    with pytest.raises(SystemExit):
        decoder_evaluation.parse_args(
            ["--checkpoint", "model.pt", "--min-changed-edges", "0"]
        )


def test_compare_random_escape_is_explicitly_enabled() -> None:
    args = decoder_evaluation.parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--srr-exploration-budget",
            "4",
            "--random-escape",
        ]
    )

    assert args.srr_exploration_budget == 4
    assert args.random_escape is True


def test_method_selection_always_measures_prism() -> None:
    for name in ("constant", "distance", "random"):
        args = decoder_evaluation.parse_args(
            ["--checkpoint", "model.pt", "--methods", name]
        )
        assert methods.resolve(args.methods) == ("prism", name)

    # Several methods in one run, which the old single-choice flag could not do.
    several = decoder_evaluation.parse_args(
        ["--checkpoint", "model.pt", "--methods", "distance,oracle"]
    )
    assert methods.resolve(several.methods) == ("prism", "distance", "oracle")

    # 'all' is the native controls: an external solver or a foreign checkpoint
    # costs wall clock or a checkpoint, so both stay opt-in.
    every = decoder_evaluation.parse_args(
        ["--checkpoint", "model.pt", "--methods", "all"]
    )
    assert methods.resolve(every.methods) == (
        "prism",
        "constant",
        "distance",
        "random",
    )

    alone = decoder_evaluation.parse_args(
        ["--checkpoint", "model.pt", "--methods", "none"]
    )
    assert methods.resolve(alone.methods) == ("prism",)

    # --baselines stays as an alias of --methods.
    alias = decoder_evaluation.parse_args(
        ["--checkpoint", "model.pt", "--baselines", "random"]
    )
    assert methods.resolve(alias.methods) == ("prism", "random")

    with pytest.raises(SystemExit):
        decoder_evaluation.parse_args(
            ["--checkpoint", "model.pt", "--methods", "not-a-method"]
        )


def test_each_method_owns_its_flags_and_their_validation() -> None:
    args = decoder_evaluation.parse_args(
        ["--checkpoint", "model.pt", "--methods", "oracle"]
    )

    assert args.oracle_time_limit == 10.0
    assert args.oracle_refresh is False
    assert args.oracle_cache.name == "oracle_cache.json"
    assert args.urs_cache.name == "urs_cache.json"

    with pytest.raises(SystemExit):
        decoder_evaluation.parse_args(
            ["--checkpoint", "model.pt", "--oracle-time-limit", "0"]
        )


def test_oracle_routes_pyvrp_families_to_pyvrp_and_the_rest_to_ortools() -> None:
    """PyVRP takes the cvrp* grid; OR-Tools covers everything else."""
    from problem_data import generated_problem, problem_schema

    for name in ("cvrp", "ocvrpbptw", "acvrpl", "mdcvrptw", "vrptw"):
        schema = problem_schema(name)
        assert oracle._pyvrp_unsupported(schema) is None
        assert oracle.oracle_solver_for(schema) == ("pyvrp", None)

    reasons = {
        name: oracle._pyvrp_unsupported(problem_schema(name))
        for name in ("tsp", "op", "pctsp", "pdcvrp")
    }
    assert reasons["tsp"] == "single_route"
    assert reasons["op"] == "constraints=tour_limit"
    assert reasons["pctsp"] == "constraints=prize_quota"
    assert reasons["pdcvrp"] == "constraints=pickup_delivery"
    for name in reasons:
        assert oracle.oracle_solver_for(
            problem_schema(name)
        ) == ("ortools", None)

    # The unseen-resource probes ride on the resource algebra, which has no
    # PyVRP equivalent even though their constraint list is a plain CVRP.
    data = instances.generate_evrp_data(8, 1, seed=1)
    evrp = instances.solver_problem(
        "evrp", instances._instance_data(data, 0)
    )
    assert oracle._pyvrp_unsupported(evrp) == "resource_algebra"
    assert oracle.oracle_solver_for(evrp) == ("ortools", None)

    problem = generated_problem("cvrptw", 8)
    data, scale = oracle._pyvrp_problem(problem)
    assert scale == 1_000_000
    assert data.num_depots == 1
    assert data.num_clients == 8
    assert data.num_vehicles == 8


def test_ortools_oracle_objective_is_measured_by_the_decoder() -> None:
    """The OR-Tools route is scored by the decoder, on PRISM's own scale."""
    from problem_data import generated_problem

    problem = generated_problem("pdcvrp", 8)
    assert oracle.oracle_solver_for(problem)[0] == "ortools"

    objective = oracle.solve_ortools_instance(
        problem, time_limit=1.0, candidates=16
    )

    assert objective > 0.0
    # A decoder-measured objective is by construction attainable: PRISM's own
    # search on the same instance cannot be better than a perfect solver, but
    # both live on the same scale, so the value is finite and comparable.
    assert objective < float("inf")


def _row(method: str, instance, objective, **overrides) -> Row:
    fields = dict(
        variant="cvrp",
        split="seen",
        n=100,
        seed=1234,
        method=method,
        method_config="",
        instance=instance,
        objective=objective,
        feasible=1 if objective != "" else "",
        seconds=1.0,
        status="ok",
        source="evaluated",
        direction="minimize",
        reference=10.0,
    )
    fields.update(overrides)
    return Row(**fields)


def test_prism_only_report_has_no_comparison() -> None:
    rows = [_row("prism", 0, 9.8), _row("prism", 1, 10.2)]

    report = summary.method_report(rows, "prism")

    assert report["variants"] == 1
    assert report["evaluated"] == 1
    assert report["failed"] == 0
    assert report["gaps"] == [0.0]
    assert summary.methods(rows) == ["prism"]
    assert summary.compare(rows, "prism", "oracle").variants == 0


def test_blank_method_row_reports_its_reason_without_a_number() -> None:
    rows = [
        _row("prism", 0, 12.0),
        _row(
            "oracle",
            "",
            "",
            status="unsupported(single_route)",
            seconds="",
            direction="",
            reference="",
        ),
    ]

    report = summary.method_report(rows, "oracle")

    assert report["evaluated"] == 0
    assert report["unsupported"] == 1
    assert report["gaps"] == []
    # A method with nothing measured never enters a comparison.
    assert summary.compare(rows, "prism", "oracle").variants == 0


def test_cached_csv_supplies_per_instance_baseline_objectives(
    tmp_path: Path,
) -> None:
    cached_path = tmp_path / "cached.csv"
    header = ",".join(results.FIELDS)
    cached_path.write_text(
        f"{header}\n"
        "tsp,seen,100,1,prism,c,0,7.0,1,0.1,ok,evaluated,minimize,7.0\n"
        "tsp,seen,100,1,distance,c,1,7.5,1,0.1,ok,evaluated,minimize,7.0\n"
        "tsp,seen,100,1,distance,c,0,7.1,1,0.1,ok,evaluated,minimize,7.0\n"
        "op,seen,100,1,distance,c,0,30.0,1,0.1,ok,evaluated,maximize,31.0\n"
        "cvrp,seen,100,1,oracle,c,,,,,unsupported(x),evaluated,,\n"
    )

    args = decoder_evaluation.parse_args(
        ["--checkpoint", "model.pt", "--cached", str(cached_path)]
    )
    cached, present = results.load_cached_rows(cached_path)

    assert args.methods == "cached"
    assert present == ("distance",)
    # Instance order is restored from the instance column, not file order, and
    # PRISM's own rows are never reused as a baseline.
    assert cached[("tsp", "distance")]["objectives"] == [7.1, 7.5]
    assert cached[("tsp", "distance")]["direction"] == "minimize"
    assert cached[("op", "distance")]["direction"] == "maximize"
    assert all(key[1] != "prism" for key in cached)
    assert ("cvrp", "oracle") not in cached


def test_urs_method_requires_identifier() -> None:
    with pytest.raises(SystemExit):
        decoder_evaluation.parse_args(
            ["--checkpoint", "model.pt", "--methods", "urs"]
        )

    supplied = decoder_evaluation.parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--methods",
            "urs",
            "--urs-baseline-id",
            "urs.pt",
        ]
    )
    assert supplied.urs_checkpoint == Path("urs.pt")


def test_urs_method_exports_the_exact_normalized_instance_batch() -> None:
    args = SimpleNamespace(
        urs_cache=None,
        urs_refresh=True,
        urs_checkpoint=Path("urs.pt"),
        urs_no_aug=True,
        urs_batch_size=2,
        urs_cuda=-1,
        aug=1,
    )
    data = {
        "xy": torch.rand(2, 4, 2),
        "prize": torch.rand(2, 4),
    }
    batch = instances.InstanceBatch(
        variant="op",
        count=2,
        n=3,
        signature="legacy-op.pt:123",
        data_path=Path("legacy-op.pt"),
        data=data,
    )
    method = UrsMethod(args)

    try:
        command = method.command(MethodRequest(batch=batch, seed=1234))
        exported = Path(command[command.index("--data-path") + 1])
        saved = torch.load(exported, map_location="cpu", weights_only=False)
        assert command[command.index("--aug-factor") + 1] == "1"
        assert saved.keys() == data.keys()
        assert torch.equal(saved["xy"], data["xy"])
        assert torch.equal(saved["prize"], data["prize"])
        assert exported != batch.data_path
    finally:
        assert method._temporary is not None
        method._temporary.cleanup()
        method._temporary = None


def test_csv_streams_each_instance_and_resume_skips_completed_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "interrupted.csv"
    batch = instances.InstanceBatch(
        variant="cvrp",
        count=2,
        n=3,
        signature="fixed",
        problems=[{"name": "cvrp"}, {"name": "cvrp"}],
    )
    monkeypatch.setattr(runner, "build_instances", lambda *_args: batch)
    args = SimpleNamespace(seed=1234)

    class CrashableMethod(InProcessMethod):
        name = "prism"

        def __init__(self, fail: bool):
            self.fail = fail
            self.calls = 0

        def config(self) -> str:
            return "fixed-config"

        def solve_one(self, problem, initial_route, index, request) -> float:
            self.calls += 1
            self._last_direction = "minimize"
            if index == 1 and self.fail:
                # Instance zero must be durable before instance one begins.
                streamed = results.read_rows(csv_path)
                assert [(row.instance, row.objective) for row in streamed] == [
                    (0, 10.0)
                ]
                raise RuntimeError("simulated crash")
            return 10.0 + index

    first = CrashableMethod(fail=True)
    with results.RowWriter(csv_path) as writer:
        first_rows = runner.run([first], ["cvrp"], args, None, writer)
    assert first.calls == 2
    assert [(row.instance, row.objective) for row in first_rows] == [
        (0, 10.0),
        ("", ""),
    ]

    prior = results.read_rows(csv_path)
    resumed = CrashableMethod(fail=False)
    with results.RowWriter(csv_path, append=True) as writer:
        resumed_rows = runner.run(
            [resumed],
            ["cvrp"],
            args,
            None,
            writer,
            resume_rows=prior,
        )
    assert resumed.calls == 2  # incomplete method batch is rerun
    assert not any(row.status.startswith("failed") for row in resumed_rows)
    persisted = results.read_rows(csv_path)
    assert [(row.instance, row.objective) for row in persisted] == [
        (0, 10.0),
        ("", ""),
        (1, 11.0),
    ]

    complete = CrashableMethod(fail=False)
    with results.RowWriter(csv_path, append=True) as writer:
        runner.run(
            [complete],
            ["cvrp"],
            args,
            None,
            writer,
            resume_rows=persisted,
        )
    assert complete.calls == 0
    assert results.read_rows(csv_path) == persisted


def test_resume_discards_only_an_unterminated_csv_tail(tmp_path: Path) -> None:
    path = tmp_path / "partial.csv"
    complete = _row("prism", 0, 10.0)
    with results.RowWriter(path) as writer:
        writer.write(complete)
    with path.open("ab") as destination:
        destination.write(b"cvrp,seen,100,1234,prism,partial")

    assert results.read_rows(path) == [complete]
    second = _row("prism", 1, 11.0)
    with results.RowWriter(path, append=True) as writer:
        writer.write(second)

    assert results.read_rows(path) == [complete, second]


def test_all_benchmarks_are_partitioned_into_training_splits() -> None:
    variants = instances.selected_variants("all")

    assert len(variants) == 110
    seen = len(instances.SEEN_VARIANTS)
    assert sum(
        instances.variant_split(name) == "seen" for name in variants
    ) == seen
    assert sum(
        instances.variant_split(name) == "heldout" for name in variants
    ) == 110 - seen
    assert "tsptw" not in variants


def test_tsptw_is_an_explicit_heldout_evaluator_variant() -> None:
    assert instances.selected_variants("tsptw") == ["tsptw"]
    assert instances.variant_split("tsptw") == "heldout"


def test_vrpdb_is_an_explicit_semantic_ood_evaluator_variant() -> None:
    assert instances.selected_variants("vrpdb") == ["vrpdb"]
    assert instances.variant_split("vrpdb") == "heldout"

    data = instances.generate_vrpdb_data(12, 2, seed=123)
    problem = instances.solver_problem(
        "vrpdb", instances._instance_data(data, 0)
    )

    assert problem["constraints"] == ["visit_all", "capacity"]
    assert problem["multi_route"] is True
    # The continuous-drive limit is derived per instance from the instance
    # radius, so the expectation reads it from the generator rather than
    # freezing a constant that drifts whenever the generator is retuned.
    drive_limit = float(data["max_continuous_drive"][0])
    assert problem["resources"] == [
        {
            "name": "continuous_driving_time",
            "operator": "affine_accumulator",
            "scope": "route",
            "direction": "forward",
            "initial": 0.0,
            "scale": drive_limit,
            "increment": {
                "edge_attribute": "distance",
                "coefficient": 1.0,
            },
            "reset": {
                "value": 0.0,
                "at_depot": True,
                "node_attribute": "break_allowed",
                "optional_before_transition": True,
                "duration": 0.75,
            },
            "bounds": [
                {
                    "upper": drive_limit,
                    "check": "transition",
                    "horizon": "return_construction",
                }
            ],
        }
    ]


def test_vrpdbtw_composes_driver_breaks_with_deadline_windows() -> None:
    assert instances.selected_variants("vrpdbtw") == ["vrpdbtw"]
    assert instances.variant_split("vrpdbtw") == "heldout"

    data = instances.generate_vrpdbtw_data(12, 2, seed=123)
    problem = instances.solver_problem(
        "vrpdbtw", instances._instance_data(data, 0)
    )

    assert problem["constraints"] == [
        "visit_all",
        "capacity",
        "time_windows",
    ]
    assert torch.count_nonzero(data["tw_start"]) == 0
    assert torch.all(data["tw_end"][:, 1:] > 0.0)
    assert problem["resources"][0]["name"] == "continuous_driving_time"
    assert problem["resources"][0]["reset"]["duration"] == 0.75


def test_loads_car_tsptw_dataset_and_lkh_reference(tmp_path: Path) -> None:
    rows = [
        (
            [[0.0, 0.0], [3.0, 4.0]],
            [0.0, 0.0],
            [0.0, 4.0],
            [100.0, 10.0],
        ),
        (
            [[0.0, 0.0], [6.0, 8.0]],
            [0.0, 0.0],
            [0.0, 9.0],
            [100.0, 20.0],
        ),
    ]
    with (tmp_path / "tsptw2_easy.pkl").open("wb") as destination:
        pickle.dump(rows, destination)
    with (tmp_path / "lkh_tsptw2_easy.pkl").open("wb") as destination:
        pickle.dump([(10.0, [1]), (20.0, [1])], destination)

    data, reference = instances.load_car_tsptw_data(
        tmp_path, size=2, hardness="easy", count=2
    )
    problem = instances.solver_problem(
        "tsptw", instances._instance_data(data, 0)
    )

    assert data["xy"].shape == (2, 2, 2)
    assert reference == 15.0
    assert problem["constraints"] == ["visit_all", "time_windows"]
    assert problem["depot_count"] == 1
    assert problem["multi_route"] is False
    assert problem["open_route"] is False
    assert problem["distance"][0, 1] == 5.0


def test_recovers_car_hard_generator_feasible_routes() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        xy = torch.rand(2, 5, 2) * 100.0
        expected = []
        customers = torch.arange(1, 5)
        for _ in range(2):
            expected.append(
                torch.cat(
                    (
                        torch.zeros(1, dtype=torch.long),
                        customers[torch.randperm(4)],
                    )
                )
            )

    actual = instances._car_hard_feasible_routes(
        xy, seed=123, generated_count=2
    )

    assert torch.equal(actual, torch.stack(expected))


def test_instance_data_preserves_batch_dimension() -> None:
    data = {
        "xy": torch.arange(24).reshape(3, 4, 2),
        "capacity": 1.0,
    }

    selected = instances._instance_data(data, 1)

    assert selected["xy"].shape == (1, 4, 2)
    assert torch.equal(selected["xy"], data["xy"][1:2])
    assert selected["capacity"] == 1.0


def test_split_report_keeps_results_and_failures_separate() -> None:
    rows = [
        _row("prism", 0, 9.0),
        _row("constant", 0, 10.0),
        _row("prism", 0, 12.0, variant="aop", split="heldout", direction="maximize"),
        _row("constant", 0, 15.0, variant="aop", split="heldout", direction="maximize"),
        _row(
            "prism",
            "",
            "",
            variant="acvrpb",
            split="heldout",
            status="failed(RuntimeError: boom)",
            seconds="",
            reference="",
        ),
    ]

    seen = summary.split_report(rows, "constant", "seen", base="prism")
    heldout = summary.split_report(rows, "prism", "heldout", base="prism")

    assert seen["variants"] == 1
    assert seen["evaluated"] == 1
    assert seen["failed"] == 0
    assert seen["comparison"].base_wins == 1
    assert seen["comparison"].method_wins == 0

    assert heldout["variants"] == 2
    assert heldout["evaluated"] == 1
    assert heldout["failed"] == 1

    # aop is a maximize variant, so the higher-scoring control wins there.
    comparison = summary.compare(
        [row for row in rows if row.split == "heldout"], "prism", "constant"
    )
    assert comparison.method_wins == 1
    assert comparison.base_wins == 0
