#!/usr/bin/env python3
"""Generate every chart in the wiki as a theme-aware SVG.

Run:  pip install matplotlib numpy && python scripts/make_figures.py

Every curve is computed from a formula or from published numbers cited in
the caption; nothing is hand-drawn. Colours come from a validated palette
(categorical slots 1-4 pass CVD and contrast checks in both light and dark
mode). Each SVG carries its own light and dark colours and switches with
the reader's system theme, so it stays legible on GitHub and on the site.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# role -> (light, dark). Categorical slots are the validated palette order.
ROLES = {
    "surface": ("#fcfcfb", "#1a1a19"),
    "ink":     ("#0b0b0b", "#ffffff"),
    "ink2":    ("#52514e", "#c3c2b7"),
    "muted":   ("#898781", "#898781"),
    "grid":    ("#e1e0d9", "#2c2c2a"),
    "axis":    ("#c3c2b7", "#383835"),
    "s1": ("#2a78d6", "#3987e5"),   # blue
    "s2": ("#eb6834", "#d95926"),   # orange
    "s3": ("#1baf7a", "#199e70"),   # aqua
    "s4": ("#eda100", "#c98500"),   # yellow
    # ordinal blue ramp (low -> high); on dark, low sits nearest the surface
    "q1": ("#86b6ef", "#184f95"),
    "q2": ("#5598e7", "#256abf"),
    "q3": ("#2a78d6", "#3987e5"),
    "q4": ("#1c5cab", "#6da7ec"),
    "q5": ("#104281", "#9ec5f4"),
}
# Draw with unique sentinel colours, then rewrite them to CSS variables.
SENT = {role: f"#0{i + 1:02x}0a0" for i, role in enumerate(ROLES)}
C = SENT  # shorthand used by the chart code

FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"

plt.rcParams.update({
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.size": 11,
    "figure.facecolor": C["surface"],
    "axes.facecolor": C["surface"],
    "axes.edgecolor": C["axis"],
    "axes.linewidth": 1,
    "axes.labelcolor": C["ink2"],
    "axes.titlecolor": C["ink"],
    "xtick.color": C["axis"], "ytick.color": C["axis"],
    "xtick.labelcolor": C["muted"], "ytick.labelcolor": C["muted"],
    "grid.color": C["grid"], "grid.linewidth": 1, "grid.linestyle": "-",
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False, "legend.labelcolor": C["ink2"],
    "text.color": C["ink2"],
})


def new(title, subtitle=None, size=(7.2, 3.8)):
    fig, ax = plt.subplots(figsize=size)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    fig.text(0.012, 0.965, title, color=C["ink"], fontsize=13, fontweight="bold", va="top")
    if subtitle:
        fig.text(0.012, 0.895, subtitle, color=C["ink2"], fontsize=10.5, va="top")
    return fig, ax


def label(ax, x, y, text, dx=6, dy=0, ha="left", va="center"):
    """Direct label in ink (never in the series colour)."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va=va, color=C["ink2"], fontsize=10.5)


def dot(ax, x, y, color):
    ax.plot([x], [y], "o", ms=8, color=color, mec=C["surface"], mew=2, zorder=5)


def legend_row(ax, handles, labels, fontsize=10):
    """Legend as one row between subtitle and plot. Handles are passed
    explicitly so a legend entry can never bind to the wrong series."""
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0, 1.01),
              ncol=len(labels), fontsize=fontsize, handlelength=1.6,
              columnspacing=1.4, borderaxespad=0)


def save(fig, name, top=0.74):
    fig.subplots_adjust(top=top, left=0.1, right=0.86, bottom=0.15)
    path = OUT / f"{name}.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    svg = path.read_text()
    for role, s in SENT.items():
        svg = svg.replace(s, f"var(--{role})")
    svg = re.sub(r"font-family:[^;\"]*", f"font-family: {FONT}", svg)
    light = ";".join(f"--{r}:{v[0]}" for r, v in ROLES.items())
    dark = ";".join(f"--{r}:{v[1]}" for r, v in ROLES.items())
    style = (f"<style>svg{{{light}}}"
             f"@media (prefers-color-scheme: dark){{svg{{{dark}}}}}</style>")
    svg = re.sub(r"(<svg[^>]*>)", r"\1" + style, svg, count=1)
    svg = re.sub(r"<metadata>.*?</metadata>", "", svg, flags=re.S)
    path.write_text(svg)
    leftover = re.findall(r"#0[0-9a-f]{2}0a0", svg)
    assert not leftover, f"{name}: unreplaced sentinels {set(leftover)}"
    print(f"  {name}.svg  {len(svg)/1024:.0f} KB")


# ----------------------------------------------------------------------------
def compute_growth():
    # Sourced training compute (FLOPs). Llama 2 is derived via 6ND and marked so.
    pts = [
        ("AlexNet", 2012.75, 5.0e17),        # 0.0058 pfs-days, OpenAI "AI and Compute"
        ("Transformer (big)", 2017.45, 2.3e19),  # Vaswani et al. 2017, Table 2
        ("GPT-3", 2020.40, 3.14e23),         # Brown et al. 2020
        ("Chinchilla", 2022.25, 5.76e23),    # Hoffmann et al. 2022
        ("PaLM", 2022.30, 2.53e24),          # Chowdhery et al. 2022
        ("Llama 2 70B*", 2023.55, 8.4e23),   # 6 x 70e9 x 2e12 (derived)
        ("Llama 3.1 405B", 2024.55, 3.8e25), # Grattafiori et al. 2024
    ]
    yr = np.array([p[1] for p in pts]); lc = np.log10([p[2] for p in pts])
    slope, icpt = np.polyfit(yr, lc, 1)
    doubling_months = 12 * np.log10(2) / slope
    fig, ax = new("Training compute of landmark models",
                  f"FLOPs, log scale · fitted trend doubles every {doubling_months:.1f} months")
    xs = np.linspace(2012.3, 2025, 50)
    ax.plot(xs, 10 ** (icpt + slope * xs), color=C["muted"], lw=1.5, zorder=1)
    for name, x, y in pts:
        dot(ax, x, y, C["s1"])
    offsets = {"AlexNet": (8, 10, "left"), "Transformer (big)": (8, 0, "left"),
               "GPT-3": (-8, 10, "right"), "Chinchilla": (0, -16, "center"),
               "PaLM": (-8, 10, "right"), "Llama 2 70B*": (8, 0, "left"),
               "Llama 3.1 405B": (-8, 6, "right")}
    for name, x, y in pts:
        dx, dy, ha = offsets[name]
        label(ax, x, y, name, dx=dx, dy=dy, ha=ha)
    ax.set_yscale("log"); ax.set_ylim(1e17, 1e26); ax.set_xlim(2012, 2026.6)
    ax.set_ylabel("Training FLOPs")
    save(fig, "compute-growth", top=0.82)
    return doubling_months


def emergence():
    logc = np.linspace(20, 25, 300)
    p = 1 / (1 + np.exp(-1.6 * (logc - 22.2)))      # per-digit accuracy: smooth
    p = 0.1 + 0.9 * p
    exact = p ** 5                                   # 5-digit answer, all digits right
    fig, ax = new("One capability, two metrics",
                  "Synthetic: a smooth per-digit gain looks 'emergent' under exact match")
    h1, = ax.plot(10 ** logc, p, color=C["s1"]); h2, = ax.plot(10 ** logc, exact, color=C["s2"])
    i = np.searchsorted(logc, 21.4); label(ax, 10 ** logc[i], p[i], "per-digit accuracy", dx=-6, dy=10, ha="right")
    j = np.searchsorted(logc, 23.4); label(ax, 10 ** logc[j], exact[j], "exact match (5 digits)", dx=8, dy=-4)
    ax.set_xscale("log"); ax.set_ylim(0, 1.05); ax.set_xlim(1e20, 1e25)
    ax.set_xlabel("Training compute (FLOPs)"); ax.set_ylabel("Accuracy")
    legend_row(ax, [h1, h2], ["per-digit accuracy", "exact match (5 digits)"])
    save(fig, "emergence-metrics")


def kl_fits():
    x = np.linspace(-6, 6, 4001); dx = x[1] - x[0]
    N = lambda m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    p = 0.5 * N(-2, 0.6) + 0.5 * N(2, 0.6)
    # forward KL(p||q) over Gaussians q is minimized by moment matching
    mf = np.sum(x * p) * dx; sf = np.sqrt(np.sum((x - mf) ** 2 * p) * dx)
    # reverse KL(q||p): grid search
    best = None
    for m in np.linspace(-3, 3, 121):
        for s in np.linspace(0.2, 3, 113):
            q = N(m, s); mask = q > 1e-12
            kl = np.sum(q[mask] * (np.log(q[mask]) - np.log(p[mask] + 1e-300))) * dx
            if best is None or kl < best[0]:
                best = (kl, m, s)
    _, mr, sr = best
    mr = abs(mr)
    fig, ax = new("Fitting one Gaussian to two modes",
                  f"Forward KL: μ={mf:.2f}, σ={sf:.2f}   ·   Reverse KL: μ={mr:.2f}, σ={sr:.2f}")
    xp = x[::10]; pp = p[::10]
    ax.fill_between(xp, pp, color=C["muted"], alpha=0.18, lw=0)
    hp, = ax.plot(xp, pp, color=C["muted"], lw=1.5)
    hf, = ax.plot(xp, N(mf, sf)[::10], color=C["s1"]); hr, = ax.plot(xp, N(mr, sr)[::10], color=C["s2"])
    label(ax, -2, p.max(), "p (data)", dx=0, dy=8, ha="center", va="bottom")
    label(ax, 0, N(mf, sf).max(), "forward KL", dx=0, dy=24, ha="center", va="bottom")
    label(ax, mr + 0.55, N(mr, sr)[np.searchsorted(x, mr + 0.55)], "reverse KL", dx=8)
    legend_row(ax, [hp, hf, hr], ["p (data)", "min KL(p‖q): covers both", "min KL(q‖p): picks one"])
    ax.set_xlim(-6, 6); ax.set_ylim(0, 0.75); ax.set_xlabel("x"); ax.set_ylabel("density")
    save(fig, "kl-forward-reverse")
    return mf, sf, mr, sr


def thin_shell():
    from math import lgamma
    r = np.linspace(0.001, 30, 3000)
    fig, ax = new("Where Gaussian samples actually live",
                  "Density of ‖z‖ for z ~ N(0, I_d): a shell at √d whose width stays ≈ 0.7")
    hs = []
    for d, col in [(2, "s1"), (16, "s2"), (128, "s3"), (512, "s4")]:
        logf = (d - 1) * np.log(r) - r ** 2 / 2 - ((d / 2 - 1) * np.log(2) + lgamma(d / 2))
        f = np.exp(logf)
        hs += ax.plot(r, f, color=C[col])
        i = np.argmax(f)
        label(ax, r[i], f[i], f"d = {d}", dx=0, dy=8, ha="center", va="bottom")
    ax.set_xlim(0, 30); ax.set_ylim(0, 0.75)
    ax.set_xlabel("‖z‖  (distance from the origin)"); ax.set_ylabel("density")
    legend_row(ax, hs, [f"d = {d}" for d in (2, 16, 128, 512)])
    save(fig, "gaussian-thin-shell")


def lr_schedules():
    T, warm, peak, lo = 100_000, 2_000, 3e-4, 3e-5
    t = np.arange(T)
    cos = np.where(t < warm, peak * t / warm,
                   lo + 0.5 * (peak - lo) * (1 + np.cos(np.pi * (t - warm) / (T - warm))))
    decay_start = int(0.9 * T)
    wsd = np.where(t < warm, peak * t / warm,
                   np.where(t < decay_start, peak,
                            peak - (peak - lo) * (t - decay_start) / (T - decay_start)))
    fig, ax = new("Learning-rate schedules",
                  "2k-step warmup, peak 3e-4, floor 3e-5 over a 100k-step run")
    h1, = ax.plot(t / 1000, cos * 1e4, color=C["s1"]); h2, = ax.plot(t / 1000, wsd * 1e4, color=C["s2"])
    label(ax, 55, cos[55_000] * 1e4, "warmup + cosine", dx=-6, dy=-12, ha="right")
    label(ax, 60, peak * 1e4, "warmup–stable–decay", dy=10, ha="center", dx=0)
    legend_row(ax, [h1, h2], ["warmup + cosine", "warmup–stable–decay (WSD)"])
    ax.set_xlim(0, 100); ax.set_ylim(0, 3.4)
    ax.set_xlabel("training step (thousands)"); ax.set_ylabel("learning rate (×10⁻⁴)")
    save(fig, "lr-schedules")


def critical_batch():
    B = np.logspace(1, 5, 200); Bc = 1000
    steps = 1 + Bc / B
    fig, ax = new("Diminishing returns from bigger batches",
                  "Relative steps to a target loss ∝ 1 + B_crit / B   (B_crit = 1,000)")
    ax.plot(B, steps, color=C["s1"])
    dot(ax, Bc, 2, C["s1"])
    label(ax, Bc, 2, "B = B_crit: 2× the minimum steps", dx=10, dy=8)
    ax.axhline(1, color=C["axis"], lw=1)
    label(ax, 3e2, 1, "floor: can't go below 1×", dy=5, dx=0, ha="center", va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(10, 1e5); ax.set_ylim(0.8, 150)
    ax.set_xlabel("batch size B (sequences)"); ax.set_ylabel("steps (× minimum)")
    save(fig, "critical-batch-size", top=0.82)


def rnn_gradients():
    k = np.arange(0, 101)
    fig, ax = new("Gradient size after k steps back through time",
                  "Scales like λᵏ, where λ is the recurrent matrix's largest singular value")
    hs = []
    for lam, col in [(0.9, "s1"), (0.99, "s2"), (1.01, "s3"), (1.1, "s4")]:
        hs += ax.plot(k, lam ** k, color=C[col])
        label(ax, 100, lam ** 100, f"λ = {lam}")
    ax.set_yscale("log"); ax.set_xlim(0, 100); ax.set_ylim(1e-5, 1e5)
    ax.set_xlabel("time steps back (k)"); ax.set_ylabel("gradient scale")
    legend_row(ax, hs, [f"λ = {l}" for l in (0.9, 0.99, 1.01, 1.1)])
    save(fig, "rnn-vanishing-gradients")


def gan_gradients():
    d = np.linspace(0.001, 0.999, 400)
    fig, ax = new("Generator gradient vs how well it fools D",
                  "|∂loss/∂logit|: minimax gives ≈0 exactly when G is bad (left); non-saturating doesn't")
    h1, = ax.plot(d, d, color=C["s1"]); h2, = ax.plot(d, 1 - d, color=C["s2"])
    label(ax, 0.8, 0.8, "minimax", dx=-6, dy=8, ha="right", va="bottom")
    label(ax, 0.2, 0.8, "non-saturating", dx=8, dy=6, va="bottom")
    legend_row(ax, [h1, h2], ["minimax: min log(1 − D(G(z)))", "non-saturating: max log D(G(z))"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_xlabel("D(G(z))  — discriminator's belief that a fake is real")
    ax.set_ylabel("gradient magnitude")
    save(fig, "gan-gradients")


def scaling_isoflops():
    E, A, B_, a, b = 1.69, 406.4, 410.7, 0.34, 0.28
    L = lambda N, D: E + A / N ** a + B_ / D ** b
    Ns = np.logspace(7.5, 12, 400)
    fig, ax = new("IsoFLOP curves from the Chinchilla fit",
                  "Loss vs model size at a fixed compute budget; dots mark the optimum")
    budgets = [1e19, 1e20, 1e21, 1e22, 1e23]
    opt = []; hs = []
    for Cb, col in zip(budgets, ["q1", "q2", "q3", "q4", "q5"]):
        D = Cb / (6 * Ns); loss = L(Ns, D)
        hs += ax.plot(Ns, loss, color=C[col])
        i = np.argmin(loss); opt.append((Ns[i], loss[i], D[i], Cb))
        dot(ax, Ns[i], loss[i], C[col])
    ax.set_xscale("log"); ax.set_xlim(10 ** 7.5, 1e12); ax.set_ylim(1.9, 4.2)
    ax.set_xlabel("parameters N"); ax.set_ylabel("predicted loss (nats/token)")
    legend_row(ax, hs, [f"{c:.0e} FLOPs".replace("e+", "e") for c in budgets], fontsize=9.5)
    save(fig, "scaling-isoflops")
    return opt


def best_of_n():
    n = np.logspace(0, 8, 400, base=2)
    fig, ax = new("pass@n with a perfect verifier",
                  "P(at least one of n independent samples is correct) = 1 − (1 − p)ⁿ")
    hs = []
    for p, col in [(0.01, "s1"), (0.05, "s2"), (0.2, "s3")]:
        y = 1 - (1 - p) ** n
        hs += ax.plot(n, y, color=C[col])
        n50 = np.log(0.5) / np.log(1 - p)          # where pass@n crosses 0.5
        label(ax, n50, 0.5, f"p = {p}", dx=-8, dy=4, ha="right", va="bottom")
    ax.set_xscale("log", base=2); ax.set_xlim(1, 256); ax.set_ylim(0, 1.05)
    ax.set_xticks([1, 4, 16, 64, 256]); ax.set_xticklabels(["1", "4", "16", "64", "256"])
    ax.set_xlabel("samples n"); ax.set_ylabel("pass@n")
    legend_row(ax, hs, ["p = 0.01 per sample", "p = 0.05", "p = 0.2"])
    save(fig, "best-of-n")


def noise_schedules():
    T = 1000; t = np.arange(1, T + 1)
    lin = np.cumprod(1 - np.linspace(1e-4, 0.02, T))
    s = 0.008; f = lambda u: np.cos((u / T + s) / (1 + s) * np.pi / 2) ** 2
    cos = f(t) / f(0)
    fig, ax = new("How fast each schedule destroys the image",
                  "Signal remaining, √ᾱₜ, over T = 1000 steps")
    h1, = ax.plot(t, np.sqrt(lin), color=C["s1"]); h2, = ax.plot(t, np.sqrt(cos), color=C["s2"])
    label(ax, 380, np.sqrt(lin[379]), "linear β (DDPM)", dx=-6, dy=-8, ha="right")
    label(ax, 650, np.sqrt(cos[649]), "cosine (Nichol & Dhariwal)", dx=8, dy=6)
    legend_row(ax, [h1, h2], ["linear β", "cosine"])
    ax.set_xlim(0, 1000); ax.set_ylim(0, 1.05)
    ax.set_xlabel("timestep t"); ax.set_ylabel("√ᾱₜ  (signal fraction)")
    save(fig, "noise-schedules")


def logit_normal():
    t = np.linspace(0.001, 0.999, 500)
    dens = 1 / (np.sqrt(2 * np.pi) * t * (1 - t)) * np.exp(-0.5 * np.log(t / (1 - t)) ** 2)
    fig, ax = new("Where training timesteps are sampled",
                  "Uniform vs logit-normal(0, 1), as used by Stable Diffusion 3")
    h1, = ax.plot(t, np.ones_like(t), color=C["s1"]); h2, = ax.plot(t, dens, color=C["s2"])
    label(ax, 0.5, dens[250], "logit-normal: focus on mid-t", dy=10, dx=0, ha="center", va="bottom")
    label(ax, 0.9, 1.0, "uniform", dy=10, dx=0, ha="center", va="bottom")
    legend_row(ax, [h1, h2], ["uniform", "logit-normal(0, 1)"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.9)
    ax.set_xlabel("t   (0 = noise, 1 = data)"); ax.set_ylabel("sampling density")
    save(fig, "logit-normal-timesteps")


def agent_compounding():
    k = np.arange(1, 51)
    fig, ax = new("Why long agent tasks fail",
                  "Task success = pᵏ for k steps at per-step reliability p")
    hs = []
    for p, col in [(0.9, "s1"), (0.95, "s2"), (0.99, "s3"), (0.999, "s4")]:
        y = p ** k
        hs += ax.plot(k, y, color=C[col])
        label(ax, 50, y[-1], f"p = {p}")
    ax.set_xlim(1, 50); ax.set_ylim(0, 1.02)
    ax.set_xlabel("steps in the task (k)"); ax.set_ylabel("probability of success")
    legend_row(ax, hs, [f"p = {p}" for p in (0.9, 0.95, 0.99, 0.999)])
    save(fig, "agent-compounding")


def elo():
    gap = np.linspace(-600, 600, 400)
    wp = 1 / (1 + 10 ** (-gap / 400))
    fig, ax = new("Elo rating gap → head-to-head win rate",
                  "P(A beats B) = 1 / (1 + 10^((R_B − R_A)/400))")
    ax.plot(gap, wp, color=C["s1"])
    for g in (100, 200, 400):
        w = 1 / (1 + 10 ** (-g / 400)); dot(ax, g, w, C["s1"])
        label(ax, g, w, f"+{g} → {w:.0%}", dx=8, dy=-10)
    ax.set_xlim(-600, 600); ax.set_ylim(0, 1)
    ax.set_xlabel("rating gap R_A − R_B"); ax.set_ylabel("A's win probability")
    save(fig, "elo-win-probability", top=0.82)


if __name__ == "__main__":
    print(f"writing to {OUT}")
    dm = compute_growth(); print(f"    fitted compute doubling time: {dm:.2f} months")
    emergence()
    mf, sf, mr, sr = kl_fits(); print(f"    KL fits: forward mu={mf:.3f} s={sf:.3f}, reverse mu={mr:.3f} s={sr:.3f}")
    thin_shell(); lr_schedules(); critical_batch(); rnn_gradients(); gan_gradients()
    for N, l, D, Cb in scaling_isoflops():
        print(f"    isoFLOP C={Cb:.0e}: N*={N:.3g}, D*={D:.3g}, D/N={D/N:.1f}, loss={l:.3f}")
    best_of_n(); noise_schedules(); logit_normal(); agent_compounding(); elo()
