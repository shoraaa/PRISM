"""Where a variant's instances come from, behind one interface.

An evaluation run draws instances from four unrelated places: PRISM's saved
size-N benchmarks, its on-the-fly generators for the unseen-resource probes,
a fresh generator at an arbitrary node count (``--n-node``), and CaR's
external TSPTW release. Each carries its own reference, its own file on disk
(or none), and its own notion of what identifies the instances.

``build_instances`` resolves all four into one ``InstanceBatch``, so callers
iterate instances without caring which source produced them, and every method
gets the same two things it needs for caching and for file-based solvers: a
``signature`` that identifies exactly these instances, and a ``data_path``
when one exists.
"""

from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

from problem_data import (
    BENCHMARK_VARIANTS,
    DatasetFinder,
    TRAIN_VARIANTS,
    decoder_problem,
    generate_aevrp_data,
    generate_evrp_data,
    generate_evrpl_data,
    generate_evrptw_data,
    generate_vrpdb_data,
    generate_vrpdbtw_data,
    generated_problem,
    load_saved_data,
)


ROOT = Path(__file__).resolve().parent.parent

solver_problem = decoder_problem


# Variants with no saved dataset: instances are generated on the fly. "evrp" is
# the zero-shot unseen-resource probe (battery via the resource algebra) and
# "evrptw" composes that battery with the trained capacity + time-window
# channels; "aevrp" and "evrpl" push the composition further (asymmetric
# directed battery, and a duration cap alongside the battery). "vrpdb" adds an
# unseen continuous-driving accumulator with optional 45-minute resets at any
# node; "vrpdbtw" composes it with deadline time windows. None are in the 110
# benchmarks, so each must be requested explicitly.
GENERATED_VARIANTS = (
    "evrp", "evrptw", "aevrp", "evrpl", "vrpdb", "vrpdbtw"
)
OPTIONAL_VARIANTS = GENERATED_VARIANTS + ("tsptw",)
CAR_ROOT = ROOT / "baselines" / "CaR-constraint"


def selected_variants(value: str) -> list[str]:
    available = list(BENCHMARK_VARIANTS)
    if value == "all":
        return available
    if value == "ccl":
        return list(CCL_VARIANTS)
    requested = [name.strip() for name in value.split(",") if name.strip()]
    unknown = sorted(set(requested) - set(available) - set(OPTIONAL_VARIANTS))
    if unknown:
        raise ValueError("unknown evaluator variants: " + ", ".join(unknown))
    if not requested:
        raise ValueError("--variants selected no variants")
    return requested


SEEN_VARIANTS = frozenset(BENCHMARK_VARIANTS) & frozenset(TRAIN_VARIANTS)


def variant_split(name: str) -> str:
    """Match training's seen/held-out boundary over the 110 benchmarks."""
    if name in OPTIONAL_VARIANTS:
        return "heldout"
    if name not in BENCHMARK_VARIANTS:
        raise ValueError(f"unknown benchmark variant: {name}")
    return "seen" if name in SEEN_VARIANTS else "heldout"


def _instance_data(data: dict, index: int) -> dict:
    """Retain the batch dimension while selecting one loaded instance."""
    return {
        key: value[index : index + 1]
        if torch.is_tensor(value) and value.ndim > 0
        else value
        for key, value in data.items()
    }


def generate_benchmark_problems(
    name: str, size: int, count: int, seed: int
) -> list[dict]:
    """Generate a reproducible evaluator batch without perturbing global RNG."""
    if size < 1 or count < 1:
        raise ValueError("generated benchmark size and count must be positive")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return [generated_problem(name, size) for _ in range(count)]


def _load_car_tsptw_rows(path: Path, count: int) -> tuple[dict, int]:
    """Load CaR's native ``(xy, service, tw_start, tw_end)`` pickle format."""
    if not path.is_file():
        raise FileNotFoundError(f"CaR TSPTW dataset does not exist: {path}")
    with path.open("rb") as source:
        saved = pickle.load(source)
    rows = saved[:count]
    if len(rows) != count:
        raise ValueError(
            f"requested {count} TSPTW instances, but {path} contains "
            f"only {len(rows)}"
        )
    if any(not isinstance(row, (tuple, list)) or len(row) != 4 for row in rows):
        raise ValueError(f"unexpected CaR TSPTW row format in {path}")
    return (
        {
            "xy": torch.tensor([row[0] for row in rows], dtype=torch.float32),
            "service_time": torch.tensor(
                [row[1] for row in rows], dtype=torch.float32
            ),
            "tw_start": torch.tensor(
                [row[2] for row in rows], dtype=torch.float32
            ),
            "tw_end": torch.tensor(
                [row[3] for row in rows], dtype=torch.float32
            ),
        },
        len(saved),
    )


def _car_hard_feasible_routes(
    xy: torch.Tensor,
    *,
    seed: int,
    generated_count: int,
) -> torch.Tensor:
    """Recover the feasible permutations embedded by CaR's hard generator.

    CaR samples all coordinates, then one permutation per instance and builds
    each time-window sequence around that permutation. Replaying those two RNG
    operations recovers the generator witness without using the LKH solutions.
    Coordinate equality guards against applying a wrong seed or data provenance.
    """
    count, size, coordinate_dim = xy.shape
    if coordinate_dim != 2 or generated_count < count:
        raise ValueError("unexpected CaR hard TSPTW coordinate shape")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        expected_xy = torch.rand(generated_count, size, 2) * 100.0
        if not torch.equal(expected_xy[:count], xy):
            raise ValueError(
                "CaR hard TSPTW coordinates do not match --tsptw-dataset-seed; "
                "pass the seed used to generate this dataset"
            )
        routes = []
        customers = torch.arange(1, size, dtype=torch.long)
        for index in range(generated_count):
            permutation = customers[torch.randperm(size - 1)]
            if index < count:
                routes.append(
                    torch.cat((torch.zeros(1, dtype=torch.long), permutation))
                )
    return torch.stack(routes)


def _load_car_tsptw_reference(path: Path, count: int) -> float | None:
    """Average the paired CaR LKH costs when the reference file is available."""
    if not path.is_file():
        return None
    with path.open("rb") as source:
        rows = pickle.load(source)[:count]
    if len(rows) != count:
        raise ValueError(
            f"requested {count} TSPTW references, but {path} contains "
            f"only {len(rows)}"
        )
    costs = [row[0] if isinstance(row, (tuple, list)) else row for row in rows]
    values = torch.tensor(costs, dtype=torch.float64)
    if not torch.isfinite(values).all():
        raise ValueError(f"non-finite TSPTW reference in {path}")
    return float(values.mean())


def load_car_tsptw_data(
    data_dir: Path,
    size: int,
    hardness: str,
    count: int,
    dataset_seed: int = 2025,
) -> tuple[dict, float | None]:
    """Load paired instances/references distributed by CaR-constraint."""
    stem = f"tsptw{size}_{hardness}.pkl"
    data, saved_count = _load_car_tsptw_rows(data_dir / stem, count)
    if hardness == "hard":
        data["initial_route"] = _car_hard_feasible_routes(
            data["xy"], seed=dataset_seed, generated_count=saved_count
        )
    reference = _load_car_tsptw_reference(data_dir / f"lkh_{stem}", count)
    return data, reference


def generate_car_tsptw_data(
    size: int,
    hardness: str,
    count: int,
    seed: int,
) -> dict:
    """Generate fresh data through CaR's own TSPTW generator entry point."""
    script = CAR_ROOT / "generate_data.py"
    if not script.is_file():
        raise FileNotFoundError(f"CaR TSPTW generator does not exist: {script}")
    with tempfile.TemporaryDirectory(prefix="prism-car-tsptw-") as temporary:
        output_root = Path(temporary)
        command = [
            sys.executable,
            str(script),
            "--problem",
            "TSPTW",
            "--problem_size",
            str(size),
            "--pomo_size",
            str(size),
            "--hardness",
            hardness,
            "--num_samples",
            str(count),
            "--seed",
            str(seed),
            "--dir",
            str(output_root),
            "--no_cuda",
        ]
        result = subprocess.run(
            command,
            cwd=CAR_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"CaR TSPTW generator failed: {detail}")
        path = output_root / "TSPTW" / f"tsptw{size}_{hardness}.pkl"
        data, _ = _load_car_tsptw_rows(path, count)
        if hardness == "hard":
            data["initial_route"] = _car_hard_feasible_routes(
                data["xy"], seed=seed, generated_count=count
            )
        return data


@dataclass
class InstanceBatch:
    """One variant's instances, however they were produced.

    ``signature`` identifies these exact instances -- a dataset file and its
    mtime, or the generator settings that produced them -- and is what result
    caches key on, so a cached result is never reused across a changed dataset
    or a changed generator seed.
    """

    variant: str
    count: int
    n: int
    signature: str
    reference: float | None = None
    data_path: Path | None = None
    data: dict | None = None
    problems: list[dict] | None = None

    def __len__(self) -> int:
        return self.count

    def problem(self, index: int) -> tuple[dict, "torch.Tensor | None"]:
        """The decoder problem for one instance, and its bootstrap route.

        The bootstrap route is only present where the source supplies one (the
        CaR hard TSPTW generator embeds a feasible permutation); everywhere
        else construction starts from the field.
        """
        if self.problems is not None:
            return self.problems[index], None
        if self.data is None:
            raise RuntimeError(f"{self.variant} batch holds no instances")
        problem = solver_problem(self.variant, _instance_data(self.data, index))
        initial_route = self.data.get("initial_route")
        if torch.is_tensor(initial_route):
            initial_route = initial_route[index].detach().cpu().numpy()
        else:
            initial_route = None
        return problem, initial_route


GENERATORS = {
    "evrp": generate_evrp_data,
    "evrptw": generate_evrptw_data,
    "aevrp": generate_aevrp_data,
    "evrpl": generate_evrpl_data,
    "vrpdb": generate_vrpdb_data,
    "vrpdbtw": generate_vrpdbtw_data,
}


def build_instances(
    name: str,
    index: int,
    args: argparse.Namespace,
    finder: DatasetFinder,
) -> InstanceBatch:
    """Resolve one variant's instances from whichever source applies.

    ``index`` is the variant's position in the run; it offsets the generator
    seed so different variants do not all draw the same instances.
    """
    if name in GENERATED_VARIANTS:
        # Zero-shot unseen-resource probe: no saved dataset. Feasibility is
        # guaranteed by the exact decoder, and a reference only exists when one
        # is requested explicitly (--evrp-oracle), so the caller fills it in.
        size = args.vrpdb_size if name.startswith("vrpdb") else args.evrp_size
        data = GENERATORS[name](size, args.val_size, seed=args.seed)
        return InstanceBatch(
            variant=name,
            count=args.val_size,
            n=size,
            signature=f"generated:{name}:{size}:{args.seed}",
            data=data,
        )

    if name == "tsptw":
        # External single-tour time-window probe. It is intentionally absent
        # from the 110-variant registry and the training curriculum.
        if args.tsptw_source == "dataset":
            data, reference = load_car_tsptw_data(
                args.tsptw_data_dir,
                args.tsptw_size,
                args.tsptw_hardness,
                args.val_size,
                args.tsptw_dataset_seed,
            )
            signature = (
                f"tsptw:dataset:{args.tsptw_size}:{args.tsptw_hardness}:"
                f"{args.tsptw_dataset_seed}"
            )
        else:
            data = generate_car_tsptw_data(
                args.tsptw_size, args.tsptw_hardness, args.val_size, args.seed
            )
            reference = None
            signature = (
                f"tsptw:generated:{args.tsptw_size}:{args.tsptw_hardness}:"
                f"{args.seed}"
            )
        print(
            "TSPTW_DATA",
            f"source={args.tsptw_source}",
            f"size={args.tsptw_size}",
            f"hardness={args.tsptw_hardness}",
            "bootstrap="
            + (
                "car_generator_witness"
                if "initial_route" in data
                else "field_construction"
            ),
            "reference=" + ("lkh" if reference is not None else "none"),
            flush=True,
        )
        return InstanceBatch(
            variant=name,
            count=args.val_size,
            n=args.tsptw_size,
            signature=signature,
            reference=reference,
            data=data,
        )

    if args.n_node is not None:
        seed = args.seed + index * args.val_size
        problems = generate_benchmark_problems(
            name, args.n_node, args.val_size, seed
        )
        print(
            "GENERATED_DATA",
            f"variant={name}",
            f"n_node={args.n_node}",
            f"count={args.val_size}",
            f"seed={seed}",
            flush=True,
        )
        return InstanceBatch(
            variant=name,
            count=args.val_size,
            n=args.n_node,
            signature=f"n_node:{args.n_node}:{seed}",
            problems=problems,
        )

    paths = finder.get(name, args.dataset_scale)
    if paths is None:
        raise FileNotFoundError(
            f"no size-{args.dataset_scale} data for {name} under "
            f"{args.dataset_dir}"
        )
    data, reference = load_saved_data(
        paths["data_path"],
        name,
        args.val_size,
        solution_path=paths["solution_path"],
        allow_aggregate_reference=False,
    )
    data_path = Path(paths["data_path"]).resolve()
    return InstanceBatch(
        variant=name,
        count=args.val_size,
        n=args.dataset_scale,
        signature=f"{data_path}:{data_path.stat().st_mtime_ns}",
        reference=reference,
        data_path=data_path,
        data=data,
    )
