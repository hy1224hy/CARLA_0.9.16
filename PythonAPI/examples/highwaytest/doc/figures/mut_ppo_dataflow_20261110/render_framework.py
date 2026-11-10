"""Draw a code-native Mut-PPO framework; no training or simulator calls.

Source of truth: highwaytest/straight/{model,learning,runner}.py.
Run with the repository venv. Outputs are sibling SVG and PNG files.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "svg.fonttype": "none",
})

INK = "#263544"
DATA = "#384653"
ACTION = "#167D8D"
COPY = "#416BA4"
GRAD = "#B24B54"
FILLS = {
    "buffer": "#FFF3CC", "model": "#E4F4E9", "teacher": "#E6EFFB",
    "loss": "#DDEFF3", "total": "#F7E2E5", "env": "#F7E2E5",
    "gae": "#F0EDF8", "action": "#E5F3F1",
}


def label(ax, x, y, value, *, size=11, color=INK, rotation=0, ha="center", bg=False):
    return ax.text(x, y, value, fontsize=size, color=color, ha=ha, va="center",
                   rotation=rotation, linespacing=1.35,
                   bbox=dict(facecolor="white", edgecolor="none", pad=1.4) if bg else None,
                   zorder=8)


def box(ax, x, y, w, h, title, lines=(), kind="model", title_size=12, body_size=10,
        dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                              facecolor=FILLS[kind], edgecolor=COPY if dashed else INK,
                              linewidth=1.25, linestyle=(0, (5, 3)) if dashed else "solid",
                              zorder=4))
    if lines:
        label(ax, x+w/2, y+h-0.34, title, size=title_size)
        usable = h-0.64
        for j, item in enumerate(lines):
            label(ax, x+w/2, y+usable*(1-(j+0.55)/len(lines)), item, size=body_size)
    else:
        label(ax, x+w/2, y+h/2, title, size=title_size)


def arrow(ax, points, *, color=DATA, style="solid", width=1.3, head=True, z=3):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, linestyle=style, linewidth=width, zorder=z,
            solid_capstyle="round")
    if head:
        a, b = points[-2:]
        ax.annotate("", xy=b, xytext=a,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=width,
                                    linestyle=style, mutation_scale=12,
                                    shrinkA=0, shrinkB=0), zorder=z)


def draw_learner(ax, name, mirror=False):
    peer = "B" if name == "A" else "A"
    sx = lambda x: 30-x if mirror else x

    def b(x, y, w, h, *args, **kwargs):
        box(ax, sx(x+w) if mirror else x, y, w, h, *args, **kwargs)

    def t(x, y, s, **kwargs):
        label(ax, sx(x), y, s, **kwargs)

    def a(points, **kwargs):
        arrow(ax, [(sx(x), y) for x, y in points], **kwargs)

    px = sx(13.0) if mirror else 0.9
    ax.add_patch(FancyBboxPatch((px, 3.5), 12.1, 14.0,
                              boxstyle="round,pad=0,rounding_size=0.18",
                              fc="#F8FAFC", ec=COPY, lw=1.2,
                              linestyle=(0, (7, 4)), zorder=0))
    t(6.95, 17.08, f"Learner {name}  |  Single-agent PPO", size=17)

    b(2, 15.05, 10.3, 1.38, f"On-policy Rollout Buffer {name}", [
        r"$o_t,\ z_t,\ \ell_t^{\mathrm{old}},\ r_t,\ v_t,\ v_{t+1},\ \mathrm{terminated}_t,\ \mathrm{done}_t$",
        "Written during rollout: Actor z / log-prob; Critic v / v_next",
    ], kind="buffer", title_size=13, body_size=10.0)

    # The framed trainable group is the target of one optimizer step.
    gx = sx(8.95) if mirror else 1.85
    ax.add_patch(FancyBboxPatch((gx, 11.72), 7.10, 2.20,
                              boxstyle="round,pad=0,rounding_size=0.08",
                              fc="none", ec="#98AF9F", lw=0.9, zorder=1))
    b(2, 12, 2.8, 1.65, f"Critic {name}", [rf"$V_{{\phi_{name}}}(o)$", "Scalar state value"], body_size=11)
    b(6, 12, 2.8, 1.65, f"Actor {name}", [rf"$\mu_{name}(o),\ \sigma_{name}$", rf"$p_{name}(z\mid o)$"], body_size=11)
    b(9.5, 12, 2.8, 1.65, f"Frozen Actor {peer}", [
        rf"$\bar p_{peer}(z\mid o_{name})$", "stopgrad / no_grad"
    ], kind="teacher", dashed=True, body_size=10.5)

    # The shared observation input is deliberate: one student's own input row.
    a([(7.15, 15.05), (7.15, 14.16)], head=False)
    a([(3.4, 14.16), (10.9, 14.16)], head=False)
    for x in (3.4, 7.4, 10.9):
        a([(x, 14.16), (x, 13.65)])
    t(3.45, 14.57, rf"Same own observation $o_{name}$", size=10.5, bg=True)

    b(2, 9.7, 2.8, 1.20, "GAE + Adv. Norm", [r"$\hat A_t,\ \hat R_t$", "r, stored v / v_next, flags"],
      kind="gae", body_size=9.1)
    b(2, 7, 6.8, 1.65, f"Base PPO Loss {name}", [
        r"$L_{\mathrm{clip}}+0.5 L_V-c_H H$",
        r"$L_V=\mathrm{mean}(V_{\phi}(o)-\hat R)^2$",
    ], kind="loss", title_size=13, body_size=12)
    b(9.5, 7, 2.8, 1.65, f"Mutual Loss {name}", [
        rf"$\mathrm{{KL}}(p_{name}\Vert\bar p_{peer})$",
        rf"$o\sim\mathcal{{D}}_{name}$; pre-tanh",
    ], kind="loss", body_size=10.5)
    b(5, 4.6, 5.5, 1.12, f"Total Loss {name}", [
        rf"$L_{name}=L_{{\mathrm{{PPO}},{name}}}+\beta(n)L_{{\mathrm{{mut}},{name}}}$",
    ], kind="total", title_size=12.5, body_size=11.5)
    t(6.9, 4.1, "KL gradients reach the student Actor only", size=10.1)

    # Saved rollout data go to GAE and PPO; these are not fresh critic values.
    a([(2, 15.52), (1.4, 15.52), (1.4, 7.85), (2, 7.85)])
    a([(1.4, 10.3), (2, 10.3)])
    t(1.23, 12.0, "stored rollout data", size=9, rotation=90, bg=True)
    t(1.15, 8.83, r"$z,\ \ell^{\mathrm{old}}$", size=10, rotation=90, bg=True)
    a([(3.4, 9.7), (3.4, 8.65)])
    t(3.4, 9.16, r"$\hat A,\ \hat R$", size=11, bg=True)

    # Current critic and actor evaluations contribute to base PPO loss.
    a([(4.8, 12.7), (5.16, 12.7), (5.16, 8.65)])
    t(5.14, 10.45, r"$V_{\phi}(o)$", size=10.2, rotation=90, bg=True)
    a([(6, 12.27), (5.68, 12.27), (5.68, 8.65)])
    t(5.7, 10.36, r"$\ell^{\mathrm{new}},\ H$", size=10.2, rotation=90, bg=True)

    # Both distributions are evaluated at the exact same own observation.
    a([(8.8, 12.35), (9.10, 12.35), (9.10, 9.15), (10.15, 9.15), (10.15, 8.65)])
    t(9.12, 10.9, rf"$p_{name}(\cdot\mid o_{name})$", size=10, rotation=90, bg=True)
    a([(10.9, 12), (10.9, 8.65)])
    t(11.54, 10.50, rf"$\bar p_{peer}(\cdot\mid o_{name})$", size=10.5, bg=True)

    a([(5.45, 7), (5.45, 6.35), (6.4, 6.35), (6.4, 5.72)])
    a([(10.9, 7), (10.9, 6.35), (9.1, 6.35), (9.1, 5.72)])
    t(10.05, 6.6, r"$\beta(n)$", size=11, bg=True)

    # One total loss and one Adam optimizer target the trainable model group.
    a([(5, 5.16), (1.76, 5.16), (1.76, 12.5), (1.85, 12.5)],
      color=GRAD, style=(0, (2, 3)), width=1.65)
    t(3.25, 5.48, "Adam: update Actor + Critic", color=GRAD, size=9.5, bg=True)

    # Separate action execution path; no loss-to-environment connection.
    a([(7.4, 12), (7.4, 9.50), (12.68, 9.50), (12.68, 2.63), (12.3, 2.63)],
      color=ACTION, width=1.75, z=2)
    t(12.72, 6.15, "policy distribution", color=ACTION, size=9.5, rotation=90, bg=True)
    b(2, 2.2, 10.3, 0.82, rf"Action sampling: $z\sim p_{name},\ a=\tanh(z)$",
      kind="action", title_size=12)
    b(2, 0.45, 10.3, 1.2, rf"Highway Environment $E_{name}$", [
        "One controlled vehicle + fixed target controller",
    ], kind="env", title_size=13, body_size=10)
    a([(7.15, 2.2), (7.15, 1.65)], color=ACTION, width=1.75)
    t(9.9, 1.93, "longitudinal / steering", color=ACTION, size=9)

    a([(2, 1.05), (0.42, 1.05), (0.42, 15.9), (2, 15.9)], width=1.5)
    t(0.20, 8.3, "observations, rewards, next observations, termination flags",
      size=10, rotation=90, bg=True)
    t(7.15, 3.35, "Online observations drive rollout; stored observations drive updates", size=9.7)


def main():
    fig, ax = plt.subplots(figsize=(24, 15.8), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_xlim(-0.05, 30.05)
    ax.set_ylim(-1.05, 19.2)
    ax.axis("off")
    label(ax, 15, 18.72, "Mut-PPO: Single-Agent Policy Mutual Learning", size=23)
    label(ax, 15, 18.12,
          "Independent on-policy rollouts  |  Same-observation Gaussian KL  |  Frozen peer snapshots", size=12)
    draw_learner(ax, "A")
    draw_learner(ax, "B", mirror=True)

    box(ax, 13.35, 11.3, 3.3, 2.14, "Mutual Learning", [
        "Frozen Actor", "snapshot exchange", "Every F rounds", "after A & B update",
    ], kind="teacher", title_size=12, body_size=10.0)
    for x0, xt in ((6.6, 14.1), (23.4, 15.9)):
        arrow(ax, [(x0, 13.65), (x0, 14.82), (xt, 14.82), (xt, 13.44)],
              color=COPY, style=(0, (5, 3)), width=1.5, z=5)
    arrow(ax, [(13.35, 12.35), (12.3, 12.35)], color=COPY, style=(0, (5, 3)), width=1.5, z=5)
    arrow(ax, [(16.65, 12.35), (17.7, 12.35)], color=COPY, style=(0, (5, 3)), width=1.5, z=5)
    label(ax, 12.85, 12.72, r"$\bar\theta_B$", color=COPY, size=11, bg=True)
    label(ax, 17.15, 12.72, r"$\bar\theta_A$", color=COPY, size=11, bg=True)
    label(ax, 15, 10.35, "Model copies only\nNo rollout exchange\nNo teacher gradients", size=10.5, color=COPY)

    label(ax, 15, 0.03, "Same single-vehicle task; A and B have independent environment copies", size=12)
    legends = [
        (2.0, DATA, "solid", "Data / loss inputs"),
        (9.0, ACTION, "solid", "Policy / action execution"),
        (16.0, COPY, (0, (5, 3)), "Frozen snapshot copy"),
        (23.2, GRAD, (0, (2, 3)), "Parameter update"),
    ]
    for x, color, style, title in legends:
        arrow(ax, [(x, -0.55), (x+0.9, -0.55)], color=color, style=style, width=1.6)
        label(ax, x+1.1, -0.55, title, size=10.2, color=color, ha="left")
    fig.subplots_adjust(left=0.012, right=0.988, top=0.987, bottom=0.02)
    for suffix in ("svg", "png"):
        fig.savefig(OUT / f"mut_ppo_dataflow.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)
    print(OUT / "mut_ppo_dataflow.svg")
    print(OUT / "mut_ppo_dataflow.png")


if __name__ == "__main__":
    main()
