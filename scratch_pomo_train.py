import numpy as np, torch, prism_decoder
from net import RefinementDecoder, build_decoder_data
from refine import run_refine_group, refine_pomo_loss, bootstrap_incumbent, candidate_adjacency

torch.manual_seed(0); np.random.seed(0)
rng = np.random.default_rng(7)
coords = rng.random((20, 2), dtype=np.float32)
dist = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1).astype(np.float32)
demand = np.r_[0.0, rng.uniform(0.02, 0.08, 19)].astype(np.float32)


def make():
    d = prism_decoder.Decoder(
        {"name": "cvrp", "coordinates": coords, "distance": dist,
         "demand": demand, "capacity": 0.6},
        {"max_candidates": 8}, {}, 4, 2.0)
    d.seed(int(rng.integers(1e9)))
    return d


@torch.no_grad()
def eval_improve(m, steps=16, trials=8):
    imps = []
    for _ in range(trials):
        d = make(); b = bootstrap_incumbent(d); base = b["objective"]; best = base
        cur = list(b["route"])
        dc = int(d.metadata["depot_count"]); n = int(d.metadata["node_count"])
        r = int(d.metadata["resource_count"])
        for _ in range(steps):
            d.set_incumbent(cur); g = build_decoder_data(d)
            live = torch.as_tensor(d.incumbent_live_state).view(n, r)
            adj = candidate_adjacency(d, "cpu")
            nr, _, _ = m(g, cur, live, dc, greedy=True, adj=adj)
            s = d.evaluate(nr)
            if s["feasible"]:
                cur = list(s["route"]); best = min(best, s["objective"])
        imps.append(100 * (base - best) / base)
    return float(np.mean(imps))


m = RefinementDecoder(units=32, rm_num=1)
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
print("epoch  0  greedy-improve %.2f%%" % eval_improve(m), flush=True)
for epoch in range(1, 301):
    groups = [run_refine_group(make(), m, group_size=8, improve_steps=8)[0]
              for _ in range(2)]
    loss = refine_pomo_loss(groups)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    if epoch % 50 == 0:
        print("epoch %3d  loss %+.4f  greedy-improve %.2f%%"
              % (epoch, float(loss), eval_improve(m)), flush=True)
