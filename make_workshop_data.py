"""Generate the workshop's CSV files: a 24-run Latin hypercube DOE extended by 48 Bayesian-optimization infill runs, and a dense reference sweep.

The infill alternates two exploitation steps (upper confidence bound, small kappa) with one exploration step (largest epistemic std), and never
places a run closer than a minimum spacing to an existing one, so the log shows the exploitation/exploration trade-off rather than a pile-up on the ridge.

The synthetic aerodynamics live here, not in the notebook. Re-running this script reproduces the files exactly.
"""

from __future__ import annotations

from enum import IntEnum, ReprEnum
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm, qmc
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

type FloatArray = NDArray[np.float64]


class Aero(float, ReprEnum):
    """Design box, atmosphere, synthetic wing and noise map.

    Attributes:
        MACH_MIN: Lower Mach bound.
        MACH_MAX: Upper Mach bound.
        AOA_MIN: Lower angle of attack, degrees.
        AOA_MAX: Upper angle of attack, degrees.
        SPEED_OF_SOUND: m/s.
        DENSITY: kg/m³.
        VISCOSITY: Pa·s.
        LIFT_SLOPE: Incompressible lift-curve slope per degree.
        ZERO_LIFT_AOA: Zero-lift angle, degrees.
        CL_MAX: Low-speed maximum lift coefficient.
        STALL_AOA: Low-speed stall angle, degrees.
        CD0: Parasite drag at low Mach.
        PARASITE_DROP: Fractional parasite-drag reduction across the Mach range.
        INDUCED_K: Induced-drag factor.
        MACH_DD: Drag-divergence Mach at zero lift.
        WAVE_DRAG: Lock's-law factor on (M − M_dd)⁴.
        NOISE_BASE: Noise std in clean attached flow.
        NOISE_STALL: Extra noise std deep in stall.
        NOISE_BUFFET: Extra noise std in transonic buffet.
        BUFFET_MACH: Mach where buffet noise switches on.
    """

    MACH_MIN = 0.2
    MACH_MAX = 0.95
    AOA_MIN = -2.0
    AOA_MAX = 16.0
    SPEED_OF_SOUND = 340.0
    DENSITY = 1.225
    VISCOSITY = 1.8e-5
    LIFT_SLOPE = 0.0986
    ZERO_LIFT_AOA = -2.0
    CL_MAX = 1.45
    STALL_AOA = 14.0
    CD0 = 0.02
    PARASITE_DROP = 0.5
    INDUCED_K = 0.0468
    MACH_DD = 0.72
    WAVE_DRAG = 40.0
    NOISE_BASE = 0.3
    NOISE_STALL = 1.5
    NOISE_BUFFET = 1.0
    BUFFET_MACH = 0.84


class Infill(float, ReprEnum):
    """Acquisition settings for the infill.

    Attributes:
        KAPPA: Exploration weight of the UCB used on exploitation steps.
        SPACING: Minimum distance between runs in the unit design box.
    """

    KAPPA = 1.0
    SPACING = 0.04


class Sizes(IntEnum):
    """Sample sizes and seeds.

    Attributes:
        GRID: Points per axis of the reference sweep.
        LHS: Runs in the initial Latin hypercube.
        EXPLORE_EVERY: Every n-th infill step explores instead of exploiting.
        POOL: Total runs in the DOE file (LHS plus BO infill).
        POOL_SEED: Seed of the LHS and the run noise.
        REF_SEED: Seed of the reference sweep's noisy realisation.
    """

    GRID = 60
    LHS = 24
    EXPLORE_EVERY = 3
    POOL = 72
    POOL_SEED = 3
    REF_SEED = 99


OUT = Path(__file__).parent / "data"
COLUMNS = ["mach", "aoa", "v_inf", "q_inf", "reynolds"]


def sigmoid(x: FloatArray) -> FloatArray:
    """Logistic function.

    Args:
        x: Inputs.

    Returns:
        1 / (1 + exp(-x)).
    """
    return 1.0 / (1.0 + np.exp(-x))


def stall_aoa(mach: FloatArray) -> FloatArray:
    """Stall angle, dropping in the transonic regime.

    Args:
        mach: Mach numbers.

    Returns:
        Stall angle in degrees.
    """
    return Aero.STALL_AOA - 7.0 * sigmoid((mach - 0.78) / 0.05)


def lift_to_drag(mach: FloatArray, aoa: FloatArray) -> FloatArray:
    """Noise-free L/D: compressible lift with smooth stall over parasite + induced + wave + separation drag.

    Args:
        mach: Mach numbers.
        aoa: Angles of attack in degrees.

    Returns:
        Lift-to-drag ratio.
    """
    post_stall = 2.0 * np.logaddexp(0.0, (aoa - stall_aoa(mach)) / 2.0)
    slope = Aero.LIFT_SLOPE / np.sqrt(1.0 - mach ** 2) * (1.0 - 0.45 * sigmoid((mach - 0.85) / 0.03))
    cl_max = Aero.CL_MAX * (1.0 - 0.5 * sigmoid((mach - 0.80) / 0.05))
    cl = cl_max * np.tanh(slope * (aoa - Aero.ZERO_LIFT_AOA) / cl_max) - 0.02 * post_stall ** 1.5
    parasite = Aero.CD0 * (1.0 - Aero.PARASITE_DROP * (mach - Aero.MACH_MIN) / (Aero.MACH_MAX - Aero.MACH_MIN))
    mach_dd = Aero.MACH_DD - 0.05 * np.clip(aoa, 0.0, Aero.AOA_MAX) / Aero.AOA_MAX
    cd = parasite + Aero.INDUCED_K * cl ** 2 + Aero.WAVE_DRAG * np.clip(mach - mach_dd, 0.0, None) ** 4 + 0.01 * post_stall ** 2
    return cl / cd


def noise_sigma(mach: FloatArray, aoa: FloatArray) -> FloatArray:
    """Noise std map: clean attached flow, noisy stall, noisy buffet.

    Args:
        mach: Mach numbers.
        aoa: Angles of attack in degrees.

    Returns:
        Noise std in L/D units.
    """
    stall = Aero.NOISE_STALL * sigmoid((aoa - stall_aoa(mach)) / 1.5)
    buffet = Aero.NOISE_BUFFET * sigmoid((mach - Aero.BUFFET_MACH) / 0.03)
    return Aero.NOISE_BASE + stall + buffet


def run_log(mach: FloatArray, aoa: FloatArray) -> pd.DataFrame:
    """The five columns a run log records: M, α, V∞, q∞, Re.

    Args:
        mach: Mach numbers.
        aoa: Angles of attack in degrees.

    Returns:
        Frame with the five feature columns.
    """
    velocity = mach * Aero.SPEED_OF_SOUND
    return pd.DataFrame({"mach": mach, "aoa": aoa, "v_inf": velocity, "q_inf": 0.5 * Aero.DENSITY * velocity ** 2, "reynolds": Aero.DENSITY * velocity / Aero.VISCOSITY})


def build_pipeline(seed: int = 0) -> Pipeline:
    """Scaler → PCA(2) → Gaussian process, the surrogate used for the infill.

    Args:
        seed: Seed for the GP optimizer restarts.

    Returns:
        Unfitted pipeline.
    """
    kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF([0.5, 0.5], (1e-2, 5.0)) + WhiteKernel(0.1, (1e-5, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=seed)
    return Pipeline([("scaler", StandardScaler()), ("reducer", PCA(n_components=2)), ("estimator", gp)])


def epistemic(pipe: Pipeline, candidates: pd.DataFrame) -> tuple[FloatArray, FloatArray]:
    """Predicted mean and epistemic std (total std with the fitted noise level removed).

    Args:
        pipe: Fitted pipeline.
        candidates: Candidate features.

    Returns:
        Mean and epistemic std per candidate.
    """
    mean, std_total = pipe.predict(candidates, return_std=True)
    gp = pipe[-1]
    std = np.sqrt(np.clip(std_total ** 2 - float(gp.kernel_.k2.noise_level) * float(gp._y_train_std ** 2), 1e-12, None))
    return mean, std


def unit_box(mach: FloatArray, aoa: FloatArray) -> FloatArray:
    """Map (M, α) into the unit square.

    Args:
        mach: Mach numbers.
        aoa: Angles of attack in degrees.

    Returns:
        (n, 2) array in [0, 1]².
    """
    return np.column_stack([(mach - Aero.MACH_MIN) / (Aero.MACH_MAX - Aero.MACH_MIN), (aoa - Aero.AOA_MIN) / (Aero.AOA_MAX - Aero.AOA_MIN)])


def make_runs(reference: pd.DataFrame) -> pd.DataFrame:
    """The DOE: a 24-run Latin hypercube, then infill picked from the reference grid alternating exploitation and exploration, with noisy L/D.

    Args:
        reference: The reference sweep (candidate conditions for the infill).

    Returns:
        192-row run log with run_id, phase (lhs / exploit / explore) and lift_to_drag columns.
    """
    rng = np.random.default_rng(int(Sizes.POOL_SEED))
    unit = qmc.LatinHypercube(d=2, seed=int(Sizes.POOL_SEED)).random(int(Sizes.LHS))
    box = qmc.scale(unit, [Aero.MACH_MIN, Aero.AOA_MIN], [Aero.MACH_MAX, Aero.AOA_MAX])
    mach, aoa = box[:, 0], box[:, 1]
    y = lift_to_drag(mach, aoa) + rng.normal(size=mach.size) * noise_sigma(mach, aoa)
    phase = ["lhs"] * mach.size
    candidates = reference[COLUMNS]
    grid = unit_box(reference["mach"].to_numpy(), reference["aoa"].to_numpy())
    for step in range(int(Sizes.POOL) - int(Sizes.LHS)):
        pipe = build_pipeline(seed=step).fit(run_log(mach, aoa), y)
        mean, std = epistemic(pipe, candidates)
        explore = step % int(Sizes.EXPLORE_EVERY) == int(Sizes.EXPLORE_EVERY) - 1
        score = std if explore else mean + Infill.KAPPA * std
        gap = np.min(np.linalg.norm(grid[:, None, :] - unit_box(mach, aoa)[None, :, :], axis=2), axis=1)
        score = np.where(gap < Infill.SPACING, -np.inf, score)
        nxt = int(np.argmax(score))
        mach, aoa = np.append(mach, reference["mach"][nxt]), np.append(aoa, reference["aoa"][nxt])
        y = np.append(y, reference["ld_true"][nxt] + rng.normal() * reference["noise_sigma"][nxt])
        phase.append("explore" if explore else "exploit")
    frame = run_log(mach, aoa)
    frame.insert(0, "run_id", np.arange(1, mach.size + 1))
    frame.insert(1, "phase", phase)
    frame["lift_to_drag"] = y
    return frame


def make_reference() -> pd.DataFrame:
    """Dense 60×60 sweep: noise-free L/D, the noise std, and one noisy CFD realisation per point.

    Returns:
        3600-row frame, AoA-major order (matches np.meshgrid ravel).
    """
    mach_axis = np.linspace(Aero.MACH_MIN, Aero.MACH_MAX, int(Sizes.GRID))
    aoa_axis = np.linspace(Aero.AOA_MIN, Aero.AOA_MAX, int(Sizes.GRID))
    mach, aoa = (g.ravel() for g in np.meshgrid(mach_axis, aoa_axis))
    frame = run_log(mach, aoa)
    frame["ld_true"] = lift_to_drag(mach, aoa)
    frame["noise_sigma"] = noise_sigma(mach, aoa)
    frame["ld_cfd"] = frame["ld_true"] + np.random.default_rng(int(Sizes.REF_SEED)).normal(size=mach.size) * frame["noise_sigma"]
    return frame


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    reference = make_reference()
    reference.to_csv(OUT / "cfd_reference.csv", index=False, float_format="%.6g")
    make_runs(reference).to_csv(OUT / "cfd_runs.csv", index=False, float_format="%.6g")
    print("written", sorted(p.name for p in OUT.iterdir()))
