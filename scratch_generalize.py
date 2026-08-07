import sys
import numpy as np, torch, prism_decoder
import problem_data as pd
from net import RefinementDecoder, build_decoder_data
from refine import run_refine_group, refine_pomo_loss, bootstrap_incumbent, candidate_adjacency

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "cvrptw"
RM_NUM = int(sys.argv[2]) if len(sys.argv) > 2 else 3
SIZE = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"variant={VARIANT} size={SIZE} rm_num={RM_NUM} device={DEVICE}", flush=True)

torch.manual_seed(0); np.random.seed(0)


def make(problem=None):
    p = problem if problem is not None else pd.generated_problem(VARIANT, SIZE)
    d = prism_decoder.Decoder(p, {"max_candidates": 10}, {}, 8, 2.0)
    d.seed(int(np.random.randint(1e9)))
    return d


# Fixed eval set (same instances every eval) for a comparable metric.
EVAL_PROBLEMS = [pd.generated_problem(VARIANT, SIZE) for _ in range(6)]


@torch.no_grad()
def eval_improve(m, steps=20):
    imps = []
    for p in EVAL_PROBLEMS:
        d = make(p); b = bootstrap_incumbent(d); base = b["objective"]; best = base
        cur = list(b["route"])
        dc = int(d.metadata["depot_count"]); n = int(d.metadata["node_count"])
        r = int(d.metadata["resource_count"])
        for _ in range(steps):
            d.set_incumbent(cur); g = build_decoder_data(d, device=DEVICE)
            live = torch.as_tensor(d.incumbent_live_state, device=DEVICE).view(n, r)
            adj = candidate_adjacency(d, DEVICE)
            nr, _, _ = m(g, cur, live, dc, greedy=True, adj=adj)
            s = d.evaluate(nr)
            if s["feasible"]:
                cur = list(s["route"]); best = min(best, s["objective"])
        imps.append(100 * (base - best) / base)
    return float(np.mean(imps))


m = RefinementDecoder(units=32, rm_num=RM_NUM).to(DEVICE)
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
print("epoch  0  greedy-improve %.2f%%" % eval_improve(m), flush=True)
for epoch in range(1, 401):
    groups = [run_refine_group(make(), m, group_size=8, improve_steps=10,
                               device=DEVICE)[0]
              for _ in range(2)]
    loss = refine_pomo_loss(groups)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    if epoch % 50 == 0:
        print("epoch %3d  loss %+.4f  greedy-improve %.2f%%"
              % (epoch, float(loss), eval_improve(m)), flush=True)
