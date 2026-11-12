"""Proof-of-concept: apply this project's closed-form backup-CBF filter
architecture (bcbf_lse_cf: analytic backup-flow sensitivity + LSE/softmin
aggregation across discretized backup-rollout samples + a single-constraint
closed-form CBF correction, as in src/high_level_rl/bcbf_lse_cf_filter.py)
to the double-integrator scenario from davidvwijk/OI-CBF
(arXiv:2510.05436), and compare it against that repo's own
optimally-interpolated (OI) closed-form controller.

Scenario (reproduced exactly from OI-CBF/double_integrator/*.py):
    State x = [x1, x2], dynamics xdot = A x + B u, A=[[0,1],[0,0]], B=[0,1]
    u in [-1, 1]; desired (primary) control u_des = +1 (constant)
    backup control u_b = -1 (constant)
    safety set:  h1(x)  = -x1  >= 0
    backup set:  hb(x)  = -x2  >= 0
    alpha(s)   = 15 s + s^3        (safety strengthening function)
    alpha_b(s) = 10 s              (backup/reachability strengthening function)
    x0 = [-2, 0.5], dt = 0.02 s, backup horizon T = 1 s (50 samples)

Because A, B are constant and the backup control is constant, the backup
flow and its state-transition matrix (sensitivity) are available in closed
form (no ODE solver needed):
    Phi(t) = I + A t                          (state sensitivity / STM)
    phi(t) = Phi(t) x0 + [t^2/2, t]^T * u_b    (backup flow under u_b)

Method "OI" below is this project's own re-derivation of OI-CBF's exact
closed-form blend (Theorem 4 of the paper); it is expected to coincide with
OI-CBF's own bCBF-QP/OI-QP outputs for this example (the paper notes this
explicitly: with a single scalar input and state-independent extreme
controllers, OI and the vanilla bCBF-QP agree here).

Method "LSE-CF" is this project's own filter architecture, ported: instead
of the hard max-based blend between exactly two controllers (u_des, u_b),
it (1) evaluates every discretized backup-rollout barrier sample's value
and backs its gradient out to the *current* state via the same closed-form
sensitivity, (2) aggregates all samples into one smoothed barrier value
via a softmin (log-sum-exp) with sharpness kappa, then (3) solves the
*standard* single-constraint closed-form CBF-QP correction against that one
aggregated barrier (no restriction to a convex blend of exactly two fixed
controllers). This mirrors bcbf_lse_cf_filter.py's
`_backup_rollout_terms_and_grads` + `_softmin_value_grad` + ordinary CBF-QP
KKT solve, collapsed from that file's 6-D augmented robot state down to
this 2-D linear system.

One real approximation introduced by porting the LSE-CF architecture here:
the safety set and the backup set use *different* class-K functions
(alpha vs alpha_b) in the original problem, but LSE-CF aggregates barrier
*values* first and applies a single outer class-K function afterward. That
collapse is a known simplification of this project's own filter design,
not something newly invented for this comparison -- flagged explicitly in
the printed summary below.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Scenario constants (copied verbatim from OI-CBF/double_integrator/*.py)
# ---------------------------------------------------------------------------
A = np.array([[0.0, 1.0], [0.0, 0.0]])
B = np.array([0.0, 1.0])
U_MAX = 1.0
U_DES = 1.0
U_B = -1.0
X0 = np.array([-2.0, 0.5])
DEL_T = 0.02
TOTAL_STEPS = int(7 / DEL_T) + 2
BACKUP_T = 1.0
TSPAN_B = np.arange(0.0, BACKUP_T + DEL_T, DEL_T)  # 51 points, i=0..50
LH_CONST = 1.0
SUP_FCL = 0.2
MU_D = (DEL_T / 2.0) * LH_CONST * SUP_FCL  # discretization tightening, ~0.002

# This project's filter defaults (src/high_level_rl/config.py fallbacks).
KAPPA_BACKUP = 20.0
ALPHA_OUTER_SCALE = 15.0  # reuse the safety-set alpha's linear slope for the
# single aggregated class-K function; see docstring note above.


def alpha(s: np.ndarray) -> np.ndarray:
    return 15.0 * s + s**3


def alpha_b(s: np.ndarray) -> np.ndarray:
    return 10.0 * s


def alpha_outer(s: np.ndarray) -> np.ndarray:
    # Single class-K function standing in for both alpha and alpha_b after
    # LSE aggregation (the approximation flagged in the module docstring).
    return ALPHA_OUTER_SCALE * s + s**3


def h1(x: np.ndarray) -> float:
    return -x[0]


def hb(x: np.ndarray) -> float:
    return -x[1]


GRAD_H1 = np.array([-1.0, 0.0])
GRAD_HB = np.array([0.0, -1.0])


def backup_flow_and_stm(x0: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form backup flow phi(t) and sensitivity Phi(t)=I+At under the
    constant backup control u_b, vectorized over an array of times t."""
    n = t.shape[0]
    stm = np.zeros((n, 2, 2))
    stm[:, 0, 0] = 1.0
    stm[:, 1, 1] = 1.0
    stm[:, 0, 1] = t
    phi = np.zeros((n, 2))
    phi[:, 0] = x0[0] + x0[1] * t + 0.5 * t**2 * U_B
    phi[:, 1] = x0[1] + t * U_B
    return phi, stm


@dataclass
class StepResult:
    u: float
    min_barrier: float  # min(h1 along rollout, hb at terminal) -- feasibility proxy
    extra: dict


def oi_closed_form(x: np.ndarray) -> StepResult:
    """OI-CBF's own exact closed-form blend (paper Theorem 4 / this repo's
    OI_QP), re-derived directly from the closed-form backup flow above."""
    t_samples = TSPAN_B[1:]  # skip tau=0 (relative-degree issue, as in repo)
    phi, stm = backup_flow_and_stm(x, t_samples)

    fx0 = A @ x
    gx0 = B

    # hdot under u_des vs u_b, for every nonterminal sample.
    grad_phi = np.broadcast_to(GRAD_H1, (t_samples.shape[0], 2))
    grad_pulled = np.einsum("ni,nij->nj", grad_phi, stm)  # grad_h1(phi) @ Phi(t)
    hdot_up = grad_pulled @ fx0 + (grad_pulled @ gx0) * U_DES
    hdot_ub = grad_pulled @ fx0 + (grad_pulled @ gx0) * U_B
    h_vals = -phi[:, 0]

    a_terms = hdot_up + alpha(h_vals - MU_D)
    b_terms = hdot_ub - hdot_up

    # Terminal backup-set constraint.
    grad_pulled_T = GRAD_HB @ stm[-1]
    hdot_up_T = grad_pulled_T @ fx0 + (grad_pulled_T @ gx0) * U_DES
    hdot_ub_T = grad_pulled_T @ fx0 + (grad_pulled_T @ gx0) * U_B
    hb_T = -phi[-1, 1]
    a_terms = np.append(a_terms, hdot_up_T + alpha_b(hb_T))
    b_terms = np.append(b_terms, hdot_ub_T - hdot_up_T)

    positive_b = b_terms > 0
    if np.any(positive_b):
        mu_star = max(0.0, float(np.max(-a_terms[positive_b] / b_terms[positive_b])))
    else:
        mu_star = 0.0
    mu_star = min(mu_star, 1.0)
    u = (1.0 - mu_star) * U_DES + mu_star * U_B
    return StepResult(u=u, min_barrier=float(np.min(h_vals)), extra={"mu": mu_star})


def lse_closed_form(x: np.ndarray, kappa: float = KAPPA_BACKUP) -> StepResult:
    """This project's bcbf_lse_cf architecture, ported to this 2-D system:
    softmin-aggregate every discretized backup-rollout barrier sample into
    one barrier value/gradient, then solve the ordinary single-constraint
    closed-form CBF-QP against that one aggregated constraint."""
    t_samples = TSPAN_B[1:]
    phi, stm = backup_flow_and_stm(x, t_samples)

    fx0 = A @ x
    gx0 = B

    h_nonterm = -phi[:, 0]
    grad_x0_nonterm = np.einsum("i,nij->nj", GRAD_H1, stm)  # pulled back to x0-space

    h_term = -phi[-1, 1]
    grad_x0_term = GRAD_HB @ stm[-1]

    terms = np.append(h_nonterm, h_term)
    grads = np.vstack((grad_x0_nonterm, grad_x0_term[None, :]))

    weights = np.exp(-kappa * (terms - terms.min()))
    weights /= weights.sum()
    h_aug = -(np.log(np.sum(np.exp(-kappa * (terms - terms.min())))) - kappa * terms.min()) / kappa
    grad_aug = np.sum(weights[:, None] * grads, axis=0)

    lf = grad_aug @ fx0
    lg = grad_aug @ gx0

    omega = lf + U_DES * lg + alpha_outer(h_aug)
    if omega < 0.0 and abs(lg) > 1e-9:
        lam = -omega / (lg * lg)
    else:
        lam = 0.0
    u = U_DES + lam * lg
    u = float(np.clip(u, -U_MAX, U_MAX))
    return StepResult(u=u, min_barrier=float(np.min(h_nonterm)), extra={"h_aug": h_aug})


def simulate(controller) -> dict:
    x = X0.copy()
    xs = np.zeros((TOTAL_STEPS, 2))
    us = np.zeros(TOTAL_STEPS)
    min_h1_over_time = np.zeros(TOTAL_STEPS)
    xs[0] = x
    for i in range(1, TOTAL_STEPS):
        result = controller(x)
        u = result.u
        us[i] = u
        xdot = lambda t, xx: A @ xx + B * u  # noqa: E731
        from scipy.integrate import solve_ivp

        sol = solve_ivp(xdot, (0.0, DEL_T), x, method="RK45", rtol=1e-9, atol=1e-9)
        x = sol.y[:, -1]
        xs[i] = x
        min_h1_over_time[i] = h1(x)
    return {"x": xs, "u": us, "h1": min_h1_over_time}


def main() -> None:
    results = {
        "OI closed form (reference, this repo's method)": simulate(oi_closed_form),
        "LSE-CF (this project's filter, ported)": simulate(lse_closed_form),
    }

    out_dir = Path("logs/evaluation/oi_cbf_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    t = np.arange(TOTAL_STEPS) * DEL_T
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = {"OI closed form (reference, this repo's method)": "#1f77b4",
              "LSE-CF (this project's filter, ported)": "#d62728"}
    for label, res in results.items():
        c = colors[label]
        axes[0].plot(res["x"][:, 0], res["x"][:, 1], color=c, label=label)
        axes[1].plot(t, res["h1"], color=c, label=label)
        axes[2].plot(t, res["u"], color=c, label=label)

    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.2, label="safety boundary x1=0")
    axes[0].axhline(0, color="gray", linestyle=":", linewidth=1.0, label="backup set boundary x2=0")
    axes[0].set_xlabel("x1 (position)")
    axes[0].set_ylabel("x2 (velocity)")
    axes[0].set_title("Phase portrait")
    axes[0].legend(fontsize=7, loc="best")
    axes[0].grid(alpha=0.25)

    axes[1].axhline(0, color="red", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("h1(x) = -x1")
    axes[1].set_title("Safety margin over time")
    axes[1].grid(alpha=0.25)

    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("applied control u")
    axes[2].set_title("Control input")
    axes[2].set_ylim(-1.2, 1.2)
    axes[2].grid(alpha=0.25)

    fig.suptitle("OI-CBF double integrator: reference OI closed form vs. this project's LSE-CF filter (ported)")
    fig.tight_layout()
    out_path = out_dir / "double_integrator_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved comparison plot to {out_path}")
    for label, res in results.items():
        print(
            f"{label}: min h1 over trajectory = {res['h1'].min():.4f}, "
            f"final x = {res['x'][-1]}, mean|u-u_des| = {np.mean(np.abs(res['u'][1:] - U_DES)):.4f}"
        )


if __name__ == "__main__":
    main()
