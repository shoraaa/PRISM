#!/usr/bin/env python3
"""Generate representation.md: the resource algebra, stated formally.

Hand-written documentation of a representation goes stale the first time the
representation changes, and this one changed four times in a day. Every
declaration and every measurement below is read out of
`Decoder.resource_declarations` and the property tensors, so the document
reports the build rather than describing it.

    python scripts/write_representation.py > representation.md
"""

from __future__ import annotations

import sys
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import prism_decoder as p  # noqa: E402
import net  # noqa: E402

from prism_eval.instances import (  # noqa: E402
    GENERATORS,
    _instance_data,
    selected_variants,
    solver_problem,
)
from problem_data import DatasetFinder, load_saved_data  # noqa: E402

PROBES = ("evrp", "evrptw", "vrpdb", "vrpdbtw")

OP_SYMBOL = {
    "add": r"+",
    "join": r"\vee",
    "assign": r"\leftarrow",
    "checkpoint": r"\downarrow",
    "restore": r"\uparrow",
}
SOURCE_TEX = {
    "distance": r"d(u,v)",
    "edge_attribute": r"w_{uv}",
    "node_attribute": None,      # rendered with its read point
    "value": None,               # rendered as the constant
}

ROW_AXES = [
    (0, r"\mathbb 1[\text{extension function}]", "carries state, versus constraining service order"),
    (1, r"\mathbb 1[\text{pairwise}]", "relation form; class order is the complement"),
    (2, r"\mathbb 1[\ell > -\infty]", "lower bound finite"),
    (3, r"\mathbb 1[u < \infty]", "upper bound finite"),
    (4, r"\mathbb 1[\ell\text{ or }u\text{ varies by node}]", "per-node bound"),
    (5, r"\mathbb 1[\chi = \text{transition}]", "check phase, ordinal with slot 6"),
    (6, r"\mathbb 1[\chi \neq \text{solution end}]", r"$(1,1)\prec(0,1)\prec(0,0)$"),
    (7, r"\mathbb 1[s = \text{route}]", "resets at the depot; solution scope is the complement"),
    (8, r"\mathbb 1[\text{horizon} = \text{return}]", "bound projected across the return leg"),
    (9, r"\mathbb 1[\text{horizon} = \text{return}^{c}]", "that projection is construction-only"),
    (10, r"\varsigma(d_r)", "state width, squashed"),
    (11, r"\varsigma(|x^0|/\text{scale})", "initial magnitude"),
    (12, r"\operatorname{sgn}^+(x^0)", "initial sign, kept off the magnitude axis"),
    (13, r"|{\sim}|/n", "declared-relation density"),
]

TERM_AXES = [
    (0, r"\mathbb 1[\sigma = \text{node}]", "reads a node attribute"),
    (1, r"\mathbb 1[\sigma \in \{\text{edge},d\}]", "reads the pair"),
    (2, r"\mathbb 1[\sigma = d]", "is the intrinsic metric (a constant reads none)"),
    (3, r"\mathbb 1[\pi = u]", "reads the origin rather than the destination"),
    (4, r"\mathbb 1[\omega\text{ linear}]", r"$x \mapsto x + c\,a$"),
    (5, r"\mathbb 1[\omega\text{ absorbing}]", "discards the incoming state"),
    (6, r"\mathbb 1[\omega\text{ idempotent}]", r"$\omega\circ\omega = \omega$"),
    (7, r"\mathbb 1[\omega\text{ shadow}]", "acts on the saved copy, not the live value"),
    (8, r"\tfrac12(1 + \operatorname{or}(\vee))", r"join orientation: $\min\!\to\!0$, none $\to\tfrac12$, $\max\!\to\!1$"),
    (9, r"\mathbb 1[\varphi = \text{after}]", "acts after the admissibility test"),
    (10, r"\mathbb 1[\theta \neq \text{always}]", "conditional"),
    (11, r"\mathbb 1[\theta = \text{fail}]", "failure-driven"),
    (12, r"\mathbb 1[\theta = \text{reset}^{\downarrow}]", "departure-side"),
    (13, r"\mathbb 1[\text{locus} = \text{depot}]", "fires at any depot"),
    (14, r"\mathbb 1[\text{locus} = \text{nodes}]", "fires at declared nodes"),
    (15, r"\mathbb 1[\gamma \neq \top]", "gated by the unserved remainder"),
    (16, r"\text{polarity}(\gamma)", "which side of the remainder the gate selects"),
    (17, r"\operatorname{sgn}^+(g)", "gate attribute sign"),
    (18, r"\varsigma(\overline{|c\,a|}/\text{scale})", "mean coefficient magnitude over the declared values"),
    (19, r"\operatorname{sgn}^+(c)", "coefficient sign, kept off the magnitude axis"),
]

DEAD_AXIS = {
    ("row", 6): "no solution-end check occurs",
    ("row", 9): "no construction-only horizon occurs",
    ("row", 10): r"every row is scalar, $d_r = 1$",
    ("term", 3): r"no term reads the origin, $\pi = v$ throughout",
    ("term", 7): "no checkpoint or restore",
    ("term", 11): "no failure-driven trigger",
    ("term", 14): "no event reset at declared nodes",
}


def decoder_for(variant, finder, size=100):
    if variant in GENERATORS:
        data = GENERATORS[variant](size, 1, seed=1234)
    else:
        paths = finder.get(variant, size)
        data, _ = load_saved_data(
            paths["data_path"], variant, 1,
            solution_path=paths["solution_path"],
            allow_aggregate_reference=False,
        )
    return p.Decoder(
        solver_problem(variant, _instance_data(data, 0)),
        candidate_config={"max_candidates": 64}, n_rollouts=1, beta=2.0,
    )


def term_tex(term) -> str:
    source = term["source"]
    if source == "value":
        read = f"{term['value']:g}"
    elif source == "distance":
        read = r"d(u,v)"
    elif source == "edge_attribute":
        read = r"w_{uv}"
    else:
        read = r"a_{v}" if term["at"] == "to" else r"a_{u}"
    coefficient = term["coefficient"]
    negative = coefficient < 0
    if abs(abs(coefficient) - 1.0) > 1e-9:
        read = f"{abs(coefficient):g}\\,{read}"
    op = OP_SYMBOL.get(term["op"], term["op"])
    if term["op"] == "assign":
        body = fr"x \leftarrow {read}"
    elif term["op"] == "add":
        # A negative coefficient reads as subtraction; "x + -a" is not notation.
        body = fr"x - {read}" if negative else fr"x + {read}"
    else:
        body = fr"x \;{op}\; {read}"
    where = [r"\varphi{=}\text{" + term["phase"].split("_")[0] + "}"]
    if term["when"] != "always":
        locus = "depot" if term["at_depot"] else (
            "nodes" if len(term["trigger_nodes"]) else "-")
        where.append(r"\theta{=}\text{" + term["when"].replace("_", "\\,")
                     + r"}\text{ at }\text{" + locus + "}")
    if term["gate"] != "always":
        where.append(r"\gamma{=}\text{" + term["gate"].replace("_", "\\,") + "}")
    return f"{body} \\quad [{', '.join(where)}]"


def bound_tex(row) -> str:
    if not row.get("bounds"):
        return "-"
    bound = row["bounds"][0]
    lower, upper = bound.get("lower"), bound.get("upper")
    lo = (f"{lower:g}" if isinstance(lower, (int, float)) and np.isfinite(lower)
          else (r"\ell_v" if len(bound.get("lower_values", [])) else r"-\infty"))
    hi = (f"{upper:g}" if isinstance(upper, (int, float)) and np.isfinite(upper)
          else (r"u_v" if len(bound.get("upper_values", [])) else r"\infty"))
    close = ")" if hi == r"\infty" else "]"
    return (f"x \\in [{lo},\\, {hi}{close}"
            f" \\quad [\\chi{{=}}\\text{{{bound.get('check','-').replace('_', chr(92)+',')}}},"
            f" \\text{{{bound.get('horizon','-').replace('_', chr(92)+',')}}}]")


def emit_family(w, name, variant, row, note=""):
    w(f"#### `{name}`" + (f" — {note}" if note else "") + "\n")
    if "terms" not in row:
        if row.get("declared"):
            w(r"A relation, not an extension function: a predicate over what a route")
            w(r"has served, so $T_r=\varnothing$ and admissibility is stated directly.")
            w("")
            w(fr"$$\rho = \bigl(\text{{{row.get('relation','-')}}},\;"
              fr" s{{=}}\text{{{row.get('scope','-')}}}\bigr),"
              fr" \qquad T = \varnothing$$" + "\n")
        else:
            w(f"Abstains: kernel `{row.get('kernel','-')}` is not reproduced by any "
              f"declaration the language can state.\n")
        return
    semiring = row["semiring"]
    join = {"max_plus": r"\max", "min_plus": r"\min"}.get(semiring, r"\text{--}")
    w(f"$$\\mathcal X = \\mathbb R,\\quad x^0 = {row.get('initial',0):g},"
      f"\\quad (\\oplus,\\otimes) = ({join}, +),\\quad s = "
      f"\\text{{{row.get('scope','-')}}}$$\n")
    w("$$" + bound_tex(row) + "$$\n")
    w("Terms:\n")
    for term in row["terms"]:
        w("$$" + term_tex(term) + "$$")
    w("")


def main() -> int:
    finder = DatasetFinder(ROOT / "datasets" / "benchmarks")
    variants = selected_variants("all")

    families, membership = OrderedDict(), OrderedDict()
    row_props, term_props = [], []
    for variant in variants:
        try:
            dec = decoder_for(variant, finder)
        except Exception:
            continue
        active = [r for r in dec.resource_declarations if r["active"]]
        membership[variant] = [r["name"] for r in active]
        row_props.append(np.asarray(dec.resource_row_properties).reshape(
            -1, p.RESOURCE_ROW_PROPERTY_DIM))
        term_props.append(np.asarray(dec.resource_term_properties).reshape(
            -1, p.RESOURCE_TERM_PROPERTY_DIM))
        for row in active:
            families.setdefault(row["name"], (variant, row))

    probes = OrderedDict()
    for probe in PROBES:
        try:
            dec = decoder_for(probe, finder)
        except Exception:
            continue
        probes[probe] = [r for r in dec.resource_declarations if r["active"]]

    out = []
    w = out.append
    w("# The resource representation\n")
    w(f"Schema `{net.MODEL_SCHEMA}`. Generated by "
      "`scripts/write_representation.py`: every declaration and every number is")
    w("read out of the decoder, so this reports the build rather than describing it.\n")

    # ---- formal object -------------------------------------------------
    w("## 1. The object\n")
    w("A routing variant is a finite set of **resources**. A resource is a row")
    w("$\\rho$ carrying a state, together with a finite multiset of **terms** that")
    w("say how the state moves:\n")
    w(r"$$\rho = \bigl(\mathcal X,\; x^{0},\; (\oplus,\otimes),\; T,\; \mathcal B\bigr)$$")
    w("")
    w("with $\\mathcal X\\subseteq\\mathbb R^{d}$ the state space, $x^{0}$ its initial")
    w("value, $(\\oplus,\\otimes)$ the semiring the update runs in, $T$ the terms, and")
    w("$\\mathcal B$ the admissibility predicate.\n")
    w("A term is a point in seven coordinates:\n")
    w(r"$$\tau = (\sigma,\; \pi,\; \omega,\; \varphi,\; \theta,\; \gamma,\; c)$$")
    w("")
    w("| | | |")
    w("|---|---|---|")
    w(r"| $\sigma$ | source | $\{\text{const},\ \text{node},\ \text{edge},\ d\}$ |")
    w(r"| $\pi$ | read point | $\{u,\ v\}$ |")
    w(r"| $\omega$ | operation | $\{\text{add},\ \text{join},\ \text{assign},\ \text{ckpt},\ \text{restore}\}$ |")
    w(r"| $\varphi$ | phase | $\{\text{before},\ \text{after}\}$ relative to the test |")
    w(r"| $\theta$ | trigger | $\{\top,\ \text{reset}^{\downarrow},\ \text{reset}^{\uparrow},\ \text{ckpt}^{\uparrow},\ \text{fail}\}$ |")
    w(r"| $\gamma$ | gate | a predicate over the unserved set |")
    w(r"| $c$ | coefficient | $\mathbb R$ |")
    w("")
    w("The value a term contributes on the transition $u\\to v$ is\n")
    w(r"$$\operatorname{val}_\tau(u,v) \;=\; c\cdot a_\tau\bigl(\pi(u,v)\bigr),"
      r"\qquad \pi(u,v)=\begin{cases}v & \pi=v\\ u & \pi=u\end{cases}$$")
    w("")
    w("and the operation applies it to the running state:\n")
    w(r"$$\text{add}: x \mapsto x + \operatorname{val},\qquad"
      r"\text{join}: x \mapsto x \oplus \operatorname{val},\qquad"
      r"\text{assign}: x \mapsto \operatorname{val}$$")
    w("")
    w(r"$$\text{ckpt}: \bar x \mapsto x, \qquad \text{restore}: x \mapsto \bar x$$")
    w("")

    # ---- transition ----------------------------------------------------
    w("## 2. The transition\n")
    w("Terms are dispatched to a fixed schedule of stages selected by their own")
    w("coordinates, not composed in the order they were written. Write")
    w("$\\Omega_{P}$ for the stage selecting the terms that satisfy $P$ and")
    w("combining them by an order-free operation: a sum for additions, the")
    w("semiring fold for joins, the unique applicable term for an overwrite.")
    w("A transition is\n")
    w(r"$$T_\rho(x,u,v)\;=\;"
      r"\Omega_{\theta=\text{reset}^{\uparrow}}\;\circ\;"
      r"\Omega_{\varphi=\text{after}}\;\circ\;"
      r"\Omega_{\varphi=\text{before}}\;\circ\;"
      r"\Omega_{\theta=\text{reset}^{\downarrow}}\;(x)$$")
    w("")
    w("with the admissibility test interposed between the two accumulation")
    w("points:\n")
    w(r"$$A_\rho(x,u,v) \;=\; \mathbb 1\Bigl["
      r"\Omega_{\varphi=\text{before}}\circ\Omega_{\theta=\text{reset}^{\downarrow}}(x)"
      r"\;\in\;[\ell_v,\,u_v]\Bigr] \quad\text{when }\chi\text{ fires at }v$$")
    w("")
    w("The transition therefore depends on $T$ only as a multiset. Sums and")
    w("semiring folds commute; two overwriting terms whose selectors can fire on")
    w("the same transition are rejected at parse time unless complementary gates")
    w("make them mutually exclusive. Term order carries no semantics, which is")
    w("why the encoding below may pool over terms without discarding any.\n")
    w("There are two accumulation points and two overwrite points, and the")
    w("symmetry is the reason the coordinates are $(\\omega,\\varphi,\\theta)$ rather")
    w("than a single stage enum: $\\text{reset}^{\\downarrow}$ is an overwrite on")
    w("*departure*, so it can read the node the route is opening toward, which no")
    w("arrival-side reset can see.\n")
    w("A gate suppresses a term by a predicate over the unserved set $U$:\n")
    w(r"$$\gamma_{\text{default}}(U) = \exists j\in U:\ \operatorname{sgn} g_j = s,"
      r"\qquad \gamma_{\text{alt}}(U) = \neg\gamma_{\text{default}}(U)"
      r"\ \wedge\ \exists j\in U:\ \operatorname{sgn} g_j = -s$$")
    w("")
    w("**Composition.** A variant with rows $R$ has product state")
    w("$X=\\prod_{r\\in R}\\mathcal X_r$, transitions componentwise, and")
    w("$A=\\bigwedge_{r\\in R}A_r$. Each $x_r$ is a sufficient statistic for its own")
    w("row's continuation feasibility, so $X$ is sufficient for the conjunction.\n")

    # ---- encoding ------------------------------------------------------
    w("## 3. The encoding the network reads\n")
    w("Two maps into fixed-width real vectors, one per row and one per term:\n")
    w(f"$$\\Phi_{{\\text{{row}}}}:\\rho\\mapsto\\mathbb R^{{{p.RESOURCE_ROW_PROPERTY_DIM}}},"
      f"\\qquad \\Phi_{{\\text{{term}}}}:\\tau\\mapsto\\mathbb R^{{{p.RESOURCE_TERM_PROPERTY_DIM}}}$$")
    w("")
    w("A row is then encoded by pooling its terms, so the width does not depend on")
    w("how many terms it has:\n")
    w(r"$$h_\rho \;=\; \psi\Bigl(\Phi_{\text{row}}(\rho),\;"
      r"\tfrac{1}{|T|}\textstyle\sum_{\tau\in T}\phi\bigl(\Phi_{\text{term}}(\tau)\bigr),\;"
      r"|T|\Bigr)$$")
    w("")
    w("The sum is permutation-invariant and $|T|$ is carried explicitly, so adding a")
    w("term moves $\\rho$ to a new point without changing any tensor width.\n")
    w(r"$\varsigma(z)=z/(1+z)$ below, and $\operatorname{sgn}^+(z)$ is the three-valued "
      r"sign: $1$ for $z>0$, $0$ for $z<0$, $\tfrac12$ for $z=0$, so an absent "
      r"quantity sits between the two signs rather than sharing a value with one.")
    w("")
    for label, axes in (("Row", ROW_AXES), ("Term", TERM_AXES)):
        w(f"**$\\Phi_{{\\text{{{label.lower()}}}}}$**\n")
        w("| $i$ | coordinate | reading |")
        w("|---|---|---|")
        for index, tex, text in axes:
            w(f"| {index} | ${tex}$ | {text} |")
        w("")
    w("Every coordinate is a property, not a membership test. An operation is")
    w("located by linearity, absorption, idempotence and whether it touches the")
    w("shadow copy, so a further operation is a point in those four rather than a")
    w("new axis; the check phase and the join orientation are ordinals, so a new")
    w("value falls *between* existing ones.\n")

    # ---- families ------------------------------------------------------
    w("## 4. The constraint families\n")
    w("Every benchmark variant is a composition of these rows, shown as the decoder")
    w("publishes them rather than as their kernels are written.\n")
    for name, (variant, row) in families.items():
        emit_family(w, name, variant, row, note=f"first in `{variant}`")

    # ---- membership ----------------------------------------------------
    w("## 5. Which rows each variant activates\n")
    names = list(families)
    w("$R(v)\\subseteq$ " + ", ".join(f"`{n}`" for n in names) + "\n")
    w("| variant | " + " | ".join(f"`{n}`" for n in names) + " | $|R|$ |")
    w("|---" * (len(names) + 2) + "|")
    for variant, active in membership.items():
        marks = " | ".join("$\\bullet$" if n in active else "" for n in names)
        w(f"| `{variant}` | {marks} | {len(active)} |")
    w("")
    counts = Counter(len(a) for a in membership.values())
    w("$|R|$ distribution: " + ", ".join(
        f"${k}$ in {v} variants" for k, v in sorted(counts.items())) + ".\n")

    # ---- probes --------------------------------------------------------
    if probes:
        w("## 6. Resources outside the benchmark family\n")
        w("These are declared through the same interface and executed by the same")
        w("transition; no kernel and no neural parameter is specific to them.\n")
        family_row = np.vstack(row_props)
        family_term = np.vstack(term_props)
        row_const = {i for i in range(family_row.shape[1])
                     if family_row[:, i].std() <= 1e-9}
        term_const = {i for i in range(family_term.shape[1])
                      if family_term[:, i].std() <= 1e-9}
        for probe, rows in probes.items():
            w(f"### `{probe}`\n")
            for row in rows:
                emit_family(w, row["name"], probe, row)
            dec = decoder_for(probe, finder)
            pr = np.asarray(dec.resource_row_properties).reshape(
                -1, p.RESOURCE_ROW_PROPERTY_DIM)
            pt = np.asarray(dec.resource_term_properties).reshape(
                -1, p.RESOURCE_TERM_PROPERTY_DIM)
            novel_row = sorted(i for i in row_const
                               if np.abs(pr[:, i] - family_row[0, i]).max() > 1e-9)
            novel_term = sorted(i for i in term_const
                                if np.abs(pt[:, i] - family_term[0, i]).max() > 1e-9)
            if novel_row or novel_term:
                w(f"Axes `{probe}` exercises that the 110 benchmark variants never "
                  f"vary — untrained directions, so the transfer claim rests on the "
                  f"remaining coordinates:\n")
                for label, idx in (("row", novel_row), ("term", novel_term)):
                    for i in idx:
                        w(f"- $\\Phi_{{\\text{{{label}}}}}$ axis ${i}$: "
                          f"{DEAD_AXIS.get((label, i), '?')}")
                w("")
            else:
                w(f"Every axis `{probe}` exercises is one the benchmark family also "
                  f"varies: it is a new **point** in the trained space, not a new "
                  f"direction.\n")

    # ---- coverage ------------------------------------------------------
    w("## 7. Coverage of the encoding\n")
    Rm, Tm = np.vstack(row_props), np.vstack(term_props)
    for label, M in (("row", Rm), ("term", Tm)):
        distinct = len(np.unique(np.round(M, 2), axis=0))
        rank = np.linalg.matrix_rank(M - M.mean(0), tol=1e-6)
        dead = [i for i in range(M.shape[1]) if M[:, i].std() <= 1e-9]
        w(f"- $\\Phi_{{\\text{{{label}}}}}$: {M.shape[1]} axes, {distinct} distinct "
          f"images over {len(membership)} variants, $\\operatorname{{rank}} = {rank}$")
        if dead:
            w("  - constant on this family: " + "; ".join(
                f"${i}$ ({DEAD_AXIS.get((label, i), '?')})" for i in dead))
    w("")
    w("The gap between the axis count and the rank is deliberate: the surplus axes")
    w("are what make a new declaration a **point** in this space rather than a change")
    w("of tensor width. A constant axis is a different matter. Its weights receive no")
    w("gradient, so a resource that first exercises one is reading an untrained")
    w("direction, and the transfer claim for such a resource is correspondingly")
    w("weaker. Those axes are what a randomized training program has to cover.\n")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
