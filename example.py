"""Two-alternative conditional SMD example; run: python example.py.

Requires numpy, scipy, matplotlib. Writes the paper figure/table and raw results
into results/. No study data are used for training. Powers are independently
computed by Gauss-Hermite quadrature, not recycled training estimates.
"""
from pathlib import Path
import csv
import json

import numpy as np
from scipy.special import ndtr, ndtri, roots_hermitenorm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "results"
ALPHA, K, N, SEED = 0.05, 1000, 512, 20261102
M, TAU = np.array([-1.5, 2.0]), np.array([0.5, 2.0])
ETA = np.sqrt(2 * np.log(2) / K)
ENVELOPE = ndtr(ndtri(ALPHA) + np.abs(M))


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
        lower, upper = tails(weights[:, None, :], TAU[i] + nodes[None, :])
        out[:, i] = (ndtr(lower - M[i]) + ndtr(M[i] - upper)) @ mass
    return out


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
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
    rows += [(f"SMD average, {k} iterations", average_power[k-1]) for k in (100, 1000)]
    # Verify conditional size, including extreme conditioning values.
    lower, upper = tails(weights[:, None, :], np.linspace(-6, 8, 101)[None, :])
    size_error = float(np.max(np.abs(ndtr(lower) + ndtr(-upper) - ALPHA)))
    assert size_error < 1e-12
    assert np.all(p <= ENVELOPE + 1e-9)
    assert worst[-1] < worst[0]
    assert dual[-1] <= worst[-1] + 1e-6

    with (OUT / "example_table.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test", "conditional_size", "power_1", "power_2", "max_regret"])
        for name, power in rows:
            writer.writerow([name, ALPHA, *power, max(ENVELOPE-power)])
    table = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Test & Size & Power at $\theta_1$ & Power at $\theta_2$ & Max. regret \\", r"\midrule"]
    for name, power in rows:
        table.append(f"{name} & {ALPHA:.3f} & {power[0]:.3f} & {power[1]:.3f} & {max(ENVELOPE-power):.3f}" + r" \\")
    table += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "example_table.tex").write_text("\n".join(table) + "\n")
    np.savetxt(OUT / "example_history.csv", np.column_stack((np.arange(1,K+1),weights,p,average_power,worst,dual)), delimiter=",", header="iteration,w1,w2,power1,power2,average_power1,average_power2,max_regret,dual_lower", comments="")
    summary = dict(seed=SEED, iterations=K, draws_per_alternative=N, eta=ETA, envelope=ENVELOPE.tolist(), final_weights=weights[-1].tolist(), final_max_regret=worst[-1], numerical_lower_bound=dual[-1], numerical_gap=worst[-1]-dual[-1], quadrature_difference=integration_error, max_size_error=size_error)
    (OUT / "example_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

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
        figure.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
        figure.savefig(OUT / f"{name}.png", dpi=300)
        plt.close(figure)
    print(json.dumps(summary, indent=2))
    print((OUT / "example_table.csv").read_text())


if __name__ == "__main__":
    main()
