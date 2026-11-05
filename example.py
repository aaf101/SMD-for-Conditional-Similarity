"""Two-alternative conditional SMD; run: python example.py --epsilon 0.02 --delta 0.05.

Requires numpy, scipy, matplotlib. Writes the paper figure/table and raw results
into results/. No study data are used for training. Powers are independently
computed by Gauss-Hermite quadrature, not recycled training estimates.
"""
from pathlib import Path
import argparse
import csv
import json

import numpy as np
from scipy.special import ndtr, ndtri, roots_hermitenorm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "results"
ALPHA = 0.05
M, TAU = np.array([-1.5, 2.0]), np.array([0.5, 2.0])
ENVELOPE = ndtr(ndtri(ALPHA) + np.abs(M))


def iteration_budget(epsilon, delta, draws, alternatives=2):
    """Proposition 1: additive regret tolerance, failure probability -> K, eta."""
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not np.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta must lie strictly between zero and one")
    if not isinstance(draws, (int, np.integer)) or draws < 1:
        raise ValueError("draws must be a positive integer")
    if not isinstance(alternatives, (int, np.integer)) or alternatives < 2:
        raise ValueError("at least two alternatives are required")
    coefficient = np.sqrt(2*np.log(alternatives)) + np.sqrt(2*np.log((alternatives+1)/delta)/draws)
    iterations = int(np.ceil((coefficient/epsilon)**2))
    eta = np.sqrt(2*np.log(alternatives)/iterations)
    bound = coefficient/np.sqrt(iterations)
    return iterations, eta, bound


def tails(w, u):
    """Exact two-tail best response up to bisection/endpoint tolerances.

    Sigma=identity and beta0=0, so S~N(m,1), U~N(tau,1) independently.
    The LR is strictly convex in S. Set lower-null-tail mass q and upper
    mass alpha-q; equal LR at their endpoints determines the optimal q.
    Broadcasting supports many weights and conditioning values together.
    """
    log_a = np.log(w) - 0.5 * (np.asarray(u)[..., None] - TAU)**2 - 0.5 * M**2
    lo = np.full(log_a.shape[:-1], 1e-14)
    hi = np.full_like(lo, ALPHA - 1e-14)
    for _ in range(44):
        q = (lo + hi) / 2
        left, right = ndtri(q), -ndtri(ALPHA - q)
        ll = np.logaddexp(log_a[..., 0] + M[0]*left, log_a[..., 1] + M[1]*left)
        lr = np.logaddexp(log_a[..., 0] + M[0]*right, log_a[..., 1] + M[1]*right)
        lo, hi = np.where(ll > lr, q, lo), np.where(ll > lr, hi, q)
    q = (lo + hi) / 2
    return ndtri(q), -ndtri(ALPHA - q)


def powers(weights, order):
    """Integrate conditional Gaussian power under each alternative."""
    nodes, mass = roots_hermitenorm(order)
    mass /= np.sqrt(2*np.pi)
    out = np.empty((len(weights), 2))
    for i in range(2):
        for start in range(0, len(weights), 256):
            block = weights[start:start+256]
            lower, upper = tails(block[:, None, :], TAU[i] + nodes[None, :])
            out[start:start+len(block), i] = (ndtr(lower - M[i]) + ndtr(M[i] - upper)) @ mass
    return out


def main(epsilon, delta, N, seed, out):
    K, ETA, bound = iteration_budget(epsilon, delta, N, len(M))
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    w = np.full(2, 0.5)
    weights = np.empty((K, 2))
    for k in range(K):
        weights[k] = w
        # Fresh independent draws for each alternative and each iteration.
        u = rng.normal(TAU[:, None], 1, (2, N))
        s = rng.normal(M[:, None], 1, (2, N))
        lower, upper = tails(w, u)
        p_hat = ((s < lower) | (s > upper)).mean(axis=1)
        log_w = np.log(w) + ETA * (ENVELOPE - p_hat)
        w = np.exp(log_w - np.logaddexp.reduce(log_w))

    p = powers(weights, 160)
    check = powers(weights, 320)
    integration_error = float(np.max(np.abs(p - check)))
    assert integration_error < 1e-6, integration_error
    p = check
    average_power = np.cumsum(p, axis=0) / np.arange(1, K+1)[:, None]
    worst = np.max(ENVELOPE - average_power, axis=1)
    dual = np.maximum.accumulate(np.sum(weights * (ENVELOPE - p), axis=1))
    two_sided = ndtr(ndtri(ALPHA/2) - M) + ndtr(M + ndtri(ALPHA/2))
    rows = [("Equal-tail test", two_sided), ("Initial best response", p[0])]
    rows += [(f"SMD average, {k} iterations", average_power[k-1]) for k in sorted({min(100, K), K})]
    # Verify conditional size, including extreme conditioning values.
    lower, upper = tails(weights[:, None, :], np.linspace(-6, 8, 101)[None, :])
    size_error = float(np.max(np.abs(ndtr(lower) + ndtr(-upper) - ALPHA)))
    assert size_error < 1e-12
    assert np.all(p <= ENVELOPE + 1e-9)
    assert dual[-1] <= worst[-1] + 1e-6
    assert bound <= epsilon * (1 + 1e-12)

    with (out / "example_table.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["test", "conditional_size", "power_1", "power_2", "max_regret"])
        for name, power in rows:
            writer.writerow([name, ALPHA, *power, max(ENVELOPE-power)])
    table = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Test & Power at $\theta_1$ & Power at $\theta_2$ & Max. regret \\",
        r"\midrule",
    ]
    for name, power in rows:
        max_regret = max(ENVELOPE - power)
        table.append(
            f"{name} & {power[0]:.3f} & {power[1]:.3f} & {max_regret:.3f}"
            + r" \\"
        )
    table += [r"\bottomrule", r"\end{tabular}"]
    (out / "example_table.tex").write_text("\n".join(table) + "\n")
    np.savetxt(out / "example_history.csv", np.column_stack((np.arange(1,K+1),weights,p,average_power,worst,dual)), delimiter=",", header="iteration,w1,w2,power1,power2,average_power1,average_power2,max_regret,dual_lower", comments="")
    summary = dict(epsilon=epsilon, delta=delta, confidence_level=1-delta, theoretical_error_bound=bound, seed=seed, iterations=K, draws_per_alternative=N, eta=ETA, envelope=ENVELOPE.tolist(), final_weights=weights[-1].tolist(), final_max_regret=worst[-1], numerical_lower_bound=dual[-1], numerical_gap=worst[-1]-dual[-1], quadrature_difference=integration_error, max_size_error=size_error)
    (out / "example_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({"font.family":"serif", "font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    fig1, axis1 = plt.subplots(figsize=(5.8, 3.1), layout="constrained")
    fig2, axis2 = plt.subplots(figsize=(5.8, 3.1), layout="constrained")
    ax = [axis1, axis2]
    ax[0].plot(np.arange(1,K+1), worst, color="#185a83", label="Average-test maximum regret")
    ax[0].plot(np.arange(1,K+1), dual, color="#b06b27", linestyle="--", label="Numerical dual lower bound")
    ax[0].set(xlabel="Iteration", ylabel="Power regret")
    ax[0].legend(frameon=False, fontsize=8)
    u = np.linspace(-2, 5, 250)
    initial_lower, _ = tails(weights[0], u)
    final_lower, _ = tails(weights[-1], u)
    ax[1].axhline(ALPHA/2, color="0.65", linestyle=":", label="Equal-tail test")
    ax[1].plot(u, ndtr(initial_lower), color="#b06b27", linestyle="--", label="Initial best response")
    ax[1].plot(u, ndtr(final_lower), color="#185a83", label="Final best response")
    ax[1].set(xlabel="Conditioning statistic $u$", ylabel="Left-tail null rejection probability", ylim=(0, ALPHA))
    ax[1].legend(frameon=False, fontsize=8)
    for figure, name in [(fig1, "figure_1_convergence"), (fig2, "figure_2_tail_allocation")]:
        figure.savefig(out / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
        figure.savefig(out / f"{name}.png", dpi=300)
        plt.close(figure)
    print(json.dumps(summary, indent=2))
    print((out / "example_table.csv").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon", type=float, default=0.02, help="additive regret tolerance (default: 0.02)")
    parser.add_argument("--delta", type=float, default=0.05, help="failure probability (default: 0.05)")
    parser.add_argument("--draws", type=int, default=512, help="draws per alternative per iteration (default: 512)")
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    try:
        iteration_budget(args.epsilon, args.delta, args.draws, len(M))
    except ValueError as exc:
        parser.error(str(exc))
    main(args.epsilon, args.delta, args.draws, args.seed, args.output_dir)
