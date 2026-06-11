"""
I.4 (continued) — visualizing the MD vector vs. CAV hyperplane.

The mean-difference (MD) vector and the linear-probe CAV are both candidate
"concept directions" in activation space. `md_cav_cosine` (see I4_vector_evaluation)
summarizes their alignment as a single number; this module makes that alignment
visible.

The trick: project every sample onto the 2D plane spanned by {MD, CAV} (an
orthonormal basis built via Gram-Schmidt, with e1 = MD direction and e2 = the
component of CAV orthogonal to MD). Because this plane *contains* both vectors
exactly, the angle between their projections equals the true angle between them
in the full D-dimensional space — cosine similarity is preserved, not distorted
by an unrelated PCA basis. The "decision boundary" for each direction is drawn
as the line perpendicular to it, through the midpoint of the two classes' means
projected onto that direction.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def project_onto_md_cav_plane(
    H_pos: np.ndarray,    # (N_pos, D)
    H_neg: np.ndarray,    # (N_neg, D)
    md_vector: np.ndarray,    # (D,)
    cav_vector: np.ndarray,   # (D,)
) -> dict:
    """
    Build an orthonormal basis {e1, e2} spanning {md_vector, cav_vector}
    (e1 = MD direction, e2 = CAV component orthogonal to MD) and project both
    classes' activations onto it, centered at the global mean.

    Returns a dict with the 2D projections, the in-plane MD/CAV unit vectors
    (md2d == [1, 0] by construction), and their cosine similarity.
    """
    md  = md_vector  / np.linalg.norm(md_vector)
    cav = cav_vector / np.linalg.norm(cav_vector)
    cosine = float(md @ cav)

    e1 = md
    cav_orth = cav - cosine * e1
    e2_norm = np.linalg.norm(cav_orth)
    if e2_norm < 1e-8:
        # MD and CAV point in (numerically) the same direction — pick an
        # arbitrary axis orthogonal to e1 just so the plot has a y-extent.
        e2 = np.zeros_like(e1)
        e2[int(np.argmin(np.abs(e1)))] = 1.0
        e2 = e2 - (e2 @ e1) * e1
        e2 = e2 / np.linalg.norm(e2)
    else:
        e2 = cav_orth / e2_norm

    basis = np.stack([e1, e2])  # (2, D)
    centroid = np.vstack([H_pos, H_neg]).mean(axis=0)

    pos2d = (H_pos - centroid) @ basis.T
    neg2d = (H_neg - centroid) @ basis.T
    md2d  = basis @ md   # == [1, 0]
    cav2d = basis @ cav  # == [cosine, sin(theta)]

    return dict(
        pos2d=pos2d, neg2d=neg2d,
        md2d=md2d, cav2d=cav2d,
        cosine=cosine, basis=basis, centroid=centroid,
    )


def plot_md_cav_alignment(
    H_pos: np.ndarray,
    H_neg: np.ndarray,
    md_vector: np.ndarray,
    cav_vector: np.ndarray,
    pos_label: str,
    neg_label: str,
    title: str = "",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Scatter samples in the MD/CAV plane, draw both candidate directions as
    arrows from the global centroid, draw each direction's separating
    hyperplane (perpendicular line through the class-mean midpoint), and
    annotate the angle between MD and CAV (= md_cav_cosine).
    """
    proj = project_onto_md_cav_plane(H_pos, H_neg, md_vector, cav_vector)
    pos2d, neg2d = proj["pos2d"], proj["neg2d"]
    md2d, cav2d, cosine = proj["md2d"], proj["cav2d"], proj["cosine"]

    fig, ax = plt.subplots(figsize=(7.5, 7))

    ax.scatter(neg2d[:, 0], neg2d[:, 1], s=20, alpha=0.5, color="#3b6fb6",
               edgecolors="none", label=neg_label, zorder=2)
    ax.scatter(pos2d[:, 0], pos2d[:, 1], s=20, alpha=0.5, color="#e0622f",
               edgecolors="none", label=pos_label, zorder=2)

    pos_mean, neg_mean = pos2d.mean(axis=0), neg2d.mean(axis=0)
    ax.scatter(*pos_mean, marker="X", s=160, color="#8a2e0e",
               edgecolors="white", linewidths=1.2, zorder=4)
    ax.scatter(*neg_mean, marker="X", s=160, color="#1c3f73",
               edgecolors="white", linewidths=1.2, zorder=4)

    extent = float(np.abs(np.vstack([pos2d, neg2d])).max())
    arrow_len = 0.8 * extent
    lim = 1.15 * extent

    # MD and CAV direction arrows from the global centroid (= origin here)
    ax.annotate("", xy=md2d * arrow_len, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="seagreen", lw=2.5))
    ax.text(*(md2d * arrow_len * 1.08), "MD vector", color="seagreen",
            fontsize=10, ha="center", fontweight="bold")

    ax.annotate("", xy=cav2d * arrow_len, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="darkorange", lw=2.5))
    ax.text(*(cav2d * arrow_len * 1.08), "CAV (probe normal)", color="darkorange",
            fontsize=10, ha="center", fontweight="bold")

    # Decision boundary for each direction: perpendicular line through the
    # midpoint of the two class means' projections onto that direction.
    L = lim * 1.4
    for d, color, name in [(md2d, "seagreen", "MD"), (cav2d, "darkorange", "CAV")]:
        s_mid = 0.5 * (pos_mean @ d + neg_mean @ d)
        p0 = s_mid * d
        perp = np.array([-d[1], d[0]])
        line = np.stack([p0 - L * perp, p0 + L * perp])
        ax.plot(line[:, 0], line[:, 1], ls="--", lw=1.6, color=color, alpha=0.8,
                label=f"{name} decision boundary")

    # Angle arc + cosine annotation
    theta_md  = np.degrees(np.arctan2(md2d[1], md2d[0]))
    theta_cav = np.degrees(np.arctan2(cav2d[1], cav2d[0]))
    arc_r = 0.18 * arrow_len
    arc = mpatches.Arc((0, 0), 2 * arc_r, 2 * arc_r, angle=0,
                        theta1=min(theta_md, theta_cav), theta2=max(theta_md, theta_cav),
                        color="dimgray", lw=1.5)
    ax.add_patch(arc)
    angle_deg = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    ax.text(arc_r * 1.5, arc_r * 0.5, f"θ={angle_deg:.1f}°\ncos={cosine:.3f}",
            fontsize=9.5, color="dimgray")

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color="lightgray", lw=0.6, zorder=0)
    ax.axvline(0, color="lightgray", lw=0.6, zorder=0)
    ax.set_xlabel("MD direction  (e₁)")
    ax.set_ylabel("CAV-orthogonal component  (e₂)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
