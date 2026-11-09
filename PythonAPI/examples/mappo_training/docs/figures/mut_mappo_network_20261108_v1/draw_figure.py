"""Draw a Mut-MAPPO methods figure using the supplied Mut-DQN layout as a reference.

This is the no-communication, two-blue-team architecture. Miniature neuron
counts are schematic; widths in the labels state the actual default sizes.
Run with the repository venv. No CARLA connection is made.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as MplPath


OUT = Path(__file__).resolve().parent
W, H = 2000, 1390
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})
fig, ax = plt.subplots(figsize=(16, 11.12))
fig.patch.set_facecolor("white")
ax.set(xlim=(0, W), ylim=(H, 0))
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

INK = "#21212a"
NAVY = "#243c79"
BLUE = "#2a3995"
GRAY = "#777d89"
PANEL = "#f0f0f0"
NET = "#e0efe0"
BUFFER = "#fff1ca"
RL = "#c3e0e9"
MUTUAL = "#deedf0"
TOTAL = "#f2d9d9"
ENV = "#f8dce3"
NEURON = "#ffc824"


def txt(x, y, value, size=16, *, color=INK, weight="normal", ha="center", va="center", **kwargs):
    return ax.text(x, y, value, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va=va, zorder=8, **kwargs)


def box(x, y, w, h, color="white", *, rounded=False, edge=INK, lw=1.1, z=3):
    if rounded:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=17",
                               facecolor=color, edgecolor=edge, linewidth=lw, zorder=z)
    else:
        patch = Rectangle((x, y), w, h, facecolor=color, edgecolor=edge, linewidth=lw, zorder=z)
    ax.add_patch(patch)
    return patch


def arrow(points, *, color=INK, lw=1.05, dashed=False, head=12, z=5):
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
    patch = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=head,
                            linewidth=lw, color=color,
                            linestyle=(0, (5, 4)) if dashed else "solid", zorder=z,
                            capstyle="round", joinstyle="round")
    ax.add_patch(patch)
    return patch


def neural_net(cx, cy, width, height, layers):
    xs = [cx - width / 2 + k * width / (len(layers) - 1) for k in range(len(layers))]
    positions = []
    for x, count in zip(xs, layers):
        ys = ([cy] if count == 1 else
              [cy - height / 2 + j * height / (count - 1) for j in range(count)])
        positions.append([(x, yy) for yy in ys])
    for left, right in zip(positions, positions[1:]):
        for a in left:
            for b in right:
                ax.plot([a[0], b[0]], [a[1], b[1]], color="#5d757a", lw=.42, alpha=.75, zorder=4)
    for layer in positions:
        for x, y in layer:
            ax.add_patch(Circle((x, y), 7.1, facecolor=NEURON, edgecolor="#3e666a", lw=.65, zorder=5))


# Symmetric student panels; the gap carries only training-time distillation.
for x in (45, 1160):
    patch = box(x, 35, 795, 1055, PANEL, edge=NAVY, lw=1.1, z=0)
    patch.set_linestyle((0, (7, 5)))
txt(442, 79, "Student A / Blue Team", 20, color=NAVY, weight="bold")
txt(1558, 79, "Student B / Blue Team", 20, color=NAVY, weight="bold")

# On-policy rollout data, not DQN replay.
for x, name in ((200, "A"), (1300, "B")):
    box(x, 125, 500, 120, BUFFER)
    txt(x + 250, 153, f"Rollout Buffer {name}", 19)
    txt(x + 250, 190, rf"$D_{name}=\{{o_{{i,t}},x_t,z_{{i,t}},\log\pi^{{\mathrm{{old}}}}_{{{name},i}}\}}$", 16)
    txt(x + 250, 221, r"$r_t,\;\mathrm{terminated}_t,\;\mathrm{truncated}_t$", 15)

# Critic and actor blocks mirror the reference, with actors toward the center.
for x, name in ((90, "A"), (1630, "B")):
    box(x, 350, 280, 245, NET)
    txt(x + 140, 380, "Centralized Critic", 18)
    neural_net(x + 140, 455, 155, 74, (4, 5, 5, 1))
    txt(x + 140, 515, rf"$V_{{{name}}}(x_t,i)$", 18)
    txt(x + 140, 550, "Input: 80 + role ID (4)", 12.5)
    txt(x + 140, 579, "84 → 128 → 128 → 1", 12.5)

for x, name in ((450, "A"), (1200, "B")):
    box(x, 350, 350, 245, NET)
    txt(x + 175, 380, f"Actors {name}2 and {name}3", 18)
    neural_net(x + 90, 478, 114, 66, (3, 4, 4, 2))
    neural_net(x + 260, 478, 114, 66, (3, 4, 4, 2))
    txt(x + 90, 414, r"$i=2$", 14)
    txt(x + 260, 414, r"$i=3$", 14)
    txt(x + 175, 545, r"$\mu_i(o_i),\;\mathrm{learned}\ \log\sigma_i$", 17)
    txt(x + 175, 578, "Each: 20 → 128 → 128 → 2", 14)

# Replay-style top branches become training inputs for actor and critic.
arrow([(450, 245), (450, 290), (230, 290), (230, 350)])
arrow([(450, 290), (625, 290), (625, 350)])
txt(304, 273, r"$x_t,i$", 16)
txt(565, 273, r"$o_{i,t}$", 16)
arrow([(1550, 245), (1550, 290), (1375, 290), (1375, 350)])
arrow([(1550, 290), (1770, 290), (1770, 350)])
txt(1456, 273, r"$o_{i,t}$", 16)
txt(1706, 273, r"$x_t,i$", 16)

# Environmental action paths stay distinct from gradient paths.
arrow([(800, 477), (834, 477), (834, 1158)], color=BLUE, lw=1.2)
arrow([(1200, 477), (1166, 477), (1166, 1158)], color=NAVY, lw=1.2)
txt(822, 1010, r"$a^A_{2,3}$", 17, color=BLUE, ha="right")
txt(1178, 1010, r"$a^B_{2,3}$", 17, color=NAVY, ha="left")

# Frozen peer snapshots feed a central, same-role/same-input comparison.
box(880, 518, 240, 73, "#fffaf0", rounded=True)
txt(1000, 541, "Scene Gate (optional)", 15)
txt(1000, 571, r"$g_{B\to A}(c),\;g_{A\to B}(c)$", 15)
arrow([(1000, 591), (1000, 634)], color=NAVY, dashed=True)

box(865, 635, 270, 142, MUTUAL, rounded=True)
txt(1000, 662, "Mutual Learning", 19, weight="bold")
txt(1000, 698, "Same role / same input", 13.1)
txt(1000, 727, "Frozen peer; stop-grad", 13.1)
txt(1000, 757, r"$D_{\mathrm{KL}}(\pi_{k,i}\Vert\bar\pi_{\mathrm{peer},i})$", 17)

arrow([(800, 420), (853, 420), (853, 673), (865, 673)], color=NAVY, dashed=True, lw=1.2)
arrow([(1200, 420), (1147, 420), (1147, 673), (1135, 673)], color=NAVY, dashed=True, lw=1.2)
txt(819, 392, "snapshot", 11.5, color=NAVY, ha="left")
txt(1181, 392, "snapshot", 11.5, color=NAVY, ha="right")

# RL loss contains PPO/entropy and value regression; KL is actor-only.
for x, name in ((90, "A"), (1630, "B")):
    box(x, 815, 280, 105, RL, rounded=True)
    txt(x + 140, 841, "MAPPO Loss", 17)
    txt(x + 140, 876, rf"$L_{{\mathrm{{PPO}},{name}}}+c_VL_{{V,{name}}}$", 17)
    txt(x + 140, 907, "GAE / entropy / value regression", 11)

for x, name in ((510, "A"), (1220, "B")):
    box(x, 815, 280, 105, "white", rounded=True, edge=NAVY)
    txt(x + 140, 841, "Mutual Loss", 17, color=NAVY)
    txt(x + 140, 876, rf"$L_{{\mathrm{{mut}},{name}}}=\mathbb{{E}}[g\cdot D_{{\mathrm{{KL}}}}]$", 16)
    txt(x + 140, 907, "Actors only", 11, color=NAVY)

# Critic predictions and actor log probabilities contribute to the RL loss.
arrow([(230, 595), (230, 815)])
txt(214, 700, r"$V_A(x_t,i)$", 17, ha="right")
arrow([(1770, 595), (1770, 815)])
txt(1786, 700, r"$V_B(x_t,i)$", 17, ha="left")
arrow([(535, 595), (535, 715), (335, 715), (335, 815)])
arrow([(1465, 595), (1465, 715), (1665, 715), (1665, 815)])
txt(437, 694, r"$\log\pi_{A,i},\;H(\pi_{A,i})$", 16)
txt(1563, 694, r"$\log\pi_{B,i},\;H(\pi_{B,i})$", 16)

# Reward and terminal flags are taken from each learner's own rollout.
arrow([(408, 245), (408, 779), (370, 779), (370, 845)], lw=.85)
arrow([(1592, 245), (1592, 779), (1630, 779), (1630, 845)], lw=.85)
txt(391, 635, r"$r_t,d_t$", 15, rotation=90)
txt(1609, 635, r"$r_t,d_t$", 15, rotation=90)

arrow([(865, 744), (805, 744), (805, 795), (650, 795), (650, 815)], color=NAVY, dashed=True, lw=1.2)
arrow([(1135, 744), (1195, 744), (1195, 795), (1360, 795), (1360, 815)], color=NAVY, dashed=True, lw=1.2)

# Each student updates only its own parameters.
for x, name in ((240, "A"), (1350, "B")):
    box(x, 970, 410, 80, TOTAL, rounded=True)
    txt(x + 205, 993, "Total Loss", 18)
    txt(x + 205, 1026, rf"$L_{name}=L_{{\mathrm{{MAPPO}},{name}}}+\beta L_{{\mathrm{{mut}},{name}}}$", 18)

arrow([(230, 920), (230, 945), (385, 945), (385, 970)])
arrow([(650, 920), (650, 945), (510, 945), (510, 970)])
txt(676, 945, r"$\beta$", 17)
arrow([(1360, 920), (1360, 945), (1490, 945), (1490, 970)])
arrow([(1770, 920), (1770, 945), (1620, 945), (1620, 970)])
txt(1334, 945, r"$\beta$", 17)

# Light gray backpropagation routes are not environment control signals.
arrow([(240, 1010), (65, 1010), (65, 625), (200, 625), (200, 595)],
      color=GRAY, lw=.85, dashed=True, head=10, z=2)
arrow([(650, 1010), (813, 1010), (813, 620), (753, 620), (753, 595)],
      color=GRAY, lw=.85, dashed=True, head=10, z=2)
arrow([(1350, 1010), (1187, 1010), (1187, 620), (1247, 620), (1247, 595)],
      color=GRAY, lw=.85, dashed=True, head=10, z=2)
arrow([(1760, 1010), (1935, 1010), (1935, 625), (1800, 625), (1800, 595)],
      color=GRAY, lw=.85, dashed=True, head=10, z=2)
txt(442, 1071, "Independent optimizer / ValueNorm / random stream", 12.2, color=GRAY)
txt(1558, 1071, "Independent optimizer / ValueNorm / random stream", 12.2, color=GRAY)

# One physical environment is used serially by the two learner teams.
box(650, 1160, 700, 75, ENV)
txt(1000, 1186, "CARLA Environment", 23)
txt(1000, 1217, "Frozen red 0/1 + blue 2/3; A/B collect serially", 14)
arrow([(1000, 1235), (1000, 1260), (15, 1260), (15, 184), (200, 184)], color=BLUE, lw=1.1)
arrow([(1000, 1260), (1985, 1260), (1985, 184), (1800, 184)], color=NAVY, lw=1.1)
txt(450, 1240, "Own on-policy trajectories", 13.5, color=BLUE)
txt(1550, 1240, "Own on-policy trajectories", 13.5, color=NAVY)

txt(1000, 1315, "Fig. 1. Network and training structure of Mut-MAPPO.", 23)
txt(1000, 1361,
    "Solid: data/control flow     Navy dashed: policy distillation     Gray dashed: parameter updates     No execution-time communication",
    12.0)

# Ensure every artist fits on canvas; preserve editable text in the SVG.
fig.canvas.draw()
for text_artist in ax.texts:
    bb = text_artist.get_window_extent(fig.canvas.get_renderer())
    if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fig.bbox.width + 1 or bb.y1 > fig.bbox.height + 1:
        raise RuntimeError(f"Text lies outside the figure: {text_artist.get_text()}")

for ext, options in (("png", {"dpi": 260}), ("svg", {}), ("pdf", {})):
    target = OUT / f"mut_mappo_network.{ext}"
    fig.savefig(target, facecolor="white", **options)
    print(target)
plt.close(fig)
