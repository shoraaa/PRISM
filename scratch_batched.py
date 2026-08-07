import sys, numpy as np, torch, prism_decoder
import problem_data as pd
from net import BatchedRelocate, build_decoder_data
from refine import run_batched_group, batched_pomo_loss, bootstrap_incumbent, candidate_adjacency
from route_eval import RouteEvaluator

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "cvrptw"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
print(f"variant={VARIANT} n={N} device={DEVICE} BATCHED", flush=True)


def make(problem=None):
    p = problem if problem is not None else pd.generated_problem(VARIANT, N)
    d = prism_decoder.Decoder(p, {"max_candidates": 10}, {}, 8, 2.0)
    d.seed(int(np.random.randint(1e9)))
    return d, p


EVAL = [pd.generated_problem(VARIANT, N) for _ in range(8)]


@torch.no_grad()
def eval_improve(model, steps=20):
    imps = []
    for p in EVAL:
        d, _ = make(p)
        lp, en, rw, base, best = run_batched_group(
            d, p, model, group_size=1, improve_steps=steps, device=DEVICE, greedy=True
        )
        imps.append(100 * (base - best) / base)
    return float(np.mean(imps))


model = BatchedRelocate(units=32).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
print("epoch  0  greedy-improve %.2f%%" % eval_improve(model), flush=True)
for epoch in range(1, 601):
    losses = []
    for _ in range(8):  # 8 instances/epoch
        d, p = make()
        lp, en, rw, base, best = run_batched_group(
            d, p, model, group_size=32, improve_steps=10, device=DEVICE
        )
        losses.append(batched_pomo_loss(lp, en, rw))
    loss = torch.stack(losses).mean()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if epoch % 50 == 0:
        print("epoch %3d  loss %+.4f  greedy-improve %.2f%%"
              % (epoch, float(loss), eval_improve(model)), flush=True)
