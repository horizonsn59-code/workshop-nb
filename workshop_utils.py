

"""Plumbing for the surrogate-modeling workshop notebook.

Constants, data loading, the reference-sweep answer key and Plotly rendering
live here so the notebook only shows the parts that are being taught. Keep this
file next to ``workshop.ipynb`` with the CSVs in a ``data/`` folder beside it,
both in ``docs/source/`` and in the repository the live (Thebe) kernel starts from.
"""

import uuid
import warnings
from dataclasses import dataclass
from enum import IntEnum, ReprEnum, StrEnum
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import HTML, display
from numpy.typing import NDArray
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots
from scipy.interpolate import RegularGridInterpolator
from sklearn.exceptions import ConvergenceWarning

__all__ = [
    "FEATURES",
    "Aero",
    "Files",
    "FloatArray",
    "ReferenceSweep",
    "Sizes",
    "Trace",
    "animate",
    "field",
    "fields_figure",
    "load",
    "nearest_run",
    "points",
    "show",
]

type FloatArray = NDArray[np.float64]
type Trace = go.Heatmap | go.Scatter


class Aero(float, ReprEnum):
    """Design box, atmosphere and the buffet onset.

    Attributes:
        MACH_MIN: Lower Mach bound.
        MACH_MAX: Upper Mach bound.
        AOA_MIN: Lower angle of attack, degrees.
        AOA_MAX: Upper angle of attack, degrees.
        SPEED_OF_SOUND: m/s.
        DENSITY: kg/m³.
        VISCOSITY: Pa·s.
        BUFFET_MACH: Mach where shock buffet starts.
    """

    MACH_MIN = 0.2
    MACH_MAX = 0.95
    AOA_MIN = -2.0
    AOA_MAX = 16.0
    SPEED_OF_SOUND = 340.0
    DENSITY = 1.225
    VISCOSITY = 1.8e-5
    BUFFET_MACH = 0.84


class Files(StrEnum):
    """Workshop data files, shipped in the data folder next to the notebook.

    Attributes:
        RUNS: The CFD run log (the DOE).
        REFERENCE: Dense reference sweep used only for grading.
    """

    RUNS = "cfd_runs.csv"
    REFERENCE = "cfd_reference.csv"


class Sizes(IntEnum):
    """Sample sizes and seeds.

    Attributes:
        GRID: Points per axis of the reference sweep.
        LHS: Runs in the initial Latin hypercube (the first rows of cfd_runs.csv).
        INFILL: CFD runs added by the infill loop.
        SEED: Observation-noise seed of the infill run shown in the slider.
        HEIGHT: Figure height in pixels.
    """

    GRID = 60
    LHS = 24
    INFILL = 16
    SEED = 5
    HEIGHT = 480


FEATURES = ["mach", "aoa", "v_inf", "q_inf", "reynolds"]
DATA_DIRS = (Path(__file__).resolve().parent / "data", Path("data"), Path("../data"))
_PLOTLY_JS_EMITTED = [False]

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def load(name: Files) -> pd.DataFrame:
    """Read a workshop CSV from the local data folder.

    Args:
        name: File to load.

    Returns:
        The table.

    Raises:
        FileNotFoundError: If the file is in none of DATA_DIRS.
    """
    for folder in DATA_DIRS:
        path = folder / name
        if path.is_file():
            return pd.read_csv(path)
    searched = ", ".join(str(folder) for folder in DATA_DIRS)
    raise FileNotFoundError(f"{name} not found in: {searched}")


@dataclass(frozen=True)
class ReferenceSweep:
    """The dense Mach × AoA reference sweep: the answer key, used only for grading.

    Rows are ordered angle-of-attack-major (AoA outer, Mach inner), so any
    per-point array reshapes to (n_aoa, n_mach).

    Attributes:
        mach_axis: Distinct Mach values (heatmap x axis).
        aoa_axis: Distinct angles of attack (heatmap y axis).
        mach: Mach number of every grid point.
        aoa: Angle of attack of every grid point.
        X: The five run-log columns at every grid point.
        y_true: Noise-free L/D.
        sigma: Solver noise standard deviation.
        y_cfd: One noisy CFD result per point.
    """

    mach_axis: FloatArray
    aoa_axis: FloatArray
    mach: FloatArray
    aoa: FloatArray
    X: FloatArray
    y_true: FloatArray
    sigma: FloatArray
    y_cfd: FloatArray

    @classmethod
    def from_frame(cls, ref: pd.DataFrame) -> Self:
        """Build from the cfd_reference.csv table.

        Args:
            ref: The reference table.

        Returns:
            The sweep.
        """
        return cls(
            mach_axis=np.unique(ref["mach"]),
            aoa_axis=np.unique(ref["aoa"]),
            mach=ref["mach"].to_numpy(),
            aoa=ref["aoa"].to_numpy(),
            X=ref[FEATURES].to_numpy(),
            y_true=ref["ld_true"].to_numpy(),
            sigma=ref["noise_sigma"].to_numpy(),
            y_cfd=ref["ld_cfd"].to_numpy(),
        )

    def grid(self, z: FloatArray) -> FloatArray:
        """Reshape a per-point array to (n_aoa, n_mach).

        Args:
            z: One value per grid point.

        Returns:
            The values as a 2-D grid.
        """
        return z.reshape(self.aoa_axis.size, self.mach_axis.size)

    def true_ld(self, mach: FloatArray, aoa: FloatArray) -> FloatArray:
        """Noise-free L/D at arbitrary conditions, interpolated from the sweep.

        Args:
            mach: Mach numbers.
            aoa: Angles of attack in degrees.

        Returns:
            Lift-to-drag ratio.
        """
        interp = RegularGridInterpolator((self.aoa_axis, self.mach_axis), self.grid(self.y_true))
        return interp(np.column_stack([aoa, mach]))

    def cfd_run(self, index: NDArray[np.int64], rng: np.random.Generator) -> FloatArray:
        """Stand-in for launching CFD jobs at grid points: the true value plus the solver's noise.

        Args:
            index: Grid indices to run.
            rng: Noise generator.

        Returns:
            Noisy L/D per run.
        """
        return self.y_true[index] + rng.normal(size=np.size(index)) * self.sigma[index]


def nearest_run(mach: FloatArray, aoa: FloatArray, run_mach: FloatArray, run_aoa: FloatArray) -> FloatArray:
    """Distance from each condition to the closest run, in design-box widths.

    Args:
        mach: Mach numbers of the conditions.
        aoa: Angles of attack of the conditions.
        run_mach: Mach numbers of the runs.
        run_aoa: Angles of attack of the runs.

    Returns:
        Distance per condition (1 = the full width of the box).
    """
    dm = (mach[:, None] - run_mach[None, :]) / (Aero.MACH_MAX - Aero.MACH_MIN)
    da = (aoa[:, None] - run_aoa[None, :]) / (Aero.AOA_MAX - Aero.AOA_MIN)
    return np.sqrt(dm**2 + da**2).min(axis=1)


def show(fig: go.Figure, title: str, subtitle: str = "") -> None:
    """Apply the light theme and title, then render as self-drawing HTML.

    The HTML draws once its element is in the page, so the same output works in
    Jupyter, Thebe, nbconvert and the Sphinx build. Plotly.js is embedded once,
    with the first figure.

    Args:
        fig: Figure to render.
        title: Headline.
        subtitle: Muted line under the headline.
    """
    sub = f"<br><sup><span style='color:#666'>{subtitle}</span></sup>" if subtitle else ""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"size": 13},
        title={"text": f"<b>{title}</b>{sub}", "x": 0.0, "xanchor": "left", "font": {"size": 17}},
    )
    uid = uuid.uuid4().hex
    draw = (
        f"<script>(function(){{var fig={fig.to_json()};var tries=0;(function draw(){{"
        f'var el=document.getElementById("{uid}");'
        "if(!el||!window.Plotly){if(tries++<400)setTimeout(draw,25);return;}"
        "Plotly.newPlot(el,fig.data,fig.layout,{displaylogo:false,responsive:true})"
        ".then(function(){if(fig.frames)Plotly.addFrames(el,fig.frames);});})();})();</script>"
    )
    plotly_js = "" if _PLOTLY_JS_EMITTED[0] else f"<script>{get_plotlyjs()}</script>"
    _PLOTLY_JS_EMITTED[0] = True
    display(HTML(f'{plotly_js}<div id="{uid}"></div>{draw}'))


def animate(
    fig: go.Figure,
    frames: list[list[Trace]],
    labels: list[str],
    prefix: str = "",
    dropdown: bool = False,
) -> None:
    """Expose per-step trace lists through a slider or a dropdown.

    Args:
        fig: Figure whose base traces equal frames[0].
        frames: One trace list per step.
        labels: One label per step.
        prefix: Text before the active label (slider only).
        dropdown: Use a dropdown instead of a slider.
    """
    fig.frames = [go.Frame(data=data, name=label) for data, label in zip(frames, labels)]
    opts = {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}
    entries = [{"method": "animate", "args": [[label], opts], "label": label} for label in labels]
    if dropdown:
        fig.update_layout(
            updatemenus=[{"buttons": entries, "x": 0, "xanchor": "left", "y": 1.2, "yanchor": "top"}],
            margin={"t": 120},
        )
    else:
        fig.update_layout(
            sliders=[{"steps": entries, "currentvalue": {"prefix": prefix}, "pad": {"t": 110}, "x": 0.02, "len": 0.96}]
        )


def fields_figure(titles: tuple[str, ...]) -> go.Figure:
    """Empty 1×k subplot figure with Mach / AoA axes.

    Args:
        titles: One subplot title per column.

    Returns:
        Figure ready for field and points traces.
    """
    fig = make_subplots(rows=1, cols=len(titles), subplot_titles=titles, horizontal_spacing=0.36 / len(titles))
    fig.update_xaxes(title_text="Mach")
    fig.update_yaxes(title_text="angle of attack [deg]", col=1)
    fig.update_layout(height=Sizes.HEIGHT, legend={"orientation": "h", "y": -0.15, "x": 0})
    return fig


def field(
    fig: go.Figure,
    sweep: ReferenceSweep,
    z: FloatArray,
    col: int,
    colorscale: str,
    zrange: tuple[float, float] | None = None,
    name: str = "",
) -> go.Heatmap:
    """Design-space heatmap of per-grid-point values for subplot column col.

    Args:
        fig: Figure the trace will live in (used to place the colour bar).
        sweep: Grid the values live on.
        z: One value per grid point.
        col: 1-based subplot column.
        colorscale: Plotly colour scale.
        zrange: Fixed colour range.
        name: Colour-bar title.

    Returns:
        Heatmap trace bound to that column's axes.
    """
    suffix = "" if col == 1 else str(col)
    zmin, zmax = zrange if zrange else (None, None)
    colorbar = {"title": name, "len": 0.8, "thickness": 14, "x": fig.layout[f"xaxis{suffix}"].domain[1] + 0.012}
    return go.Heatmap(
        x=sweep.mach_axis,
        y=sweep.aoa_axis,
        z=sweep.grid(z),
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        zsmooth="best",
        xaxis=f"x{suffix}",
        yaxis=f"y{suffix}",
        colorbar=colorbar,
        hovertemplate="Mach %{x:.2f}<br>AoA %{y:.1f}°<br>%{z:.2f}<extra></extra>",
    )


def points(
    mach: FloatArray,
    aoa: FloatArray,
    col: int,
    name: str = "CFD runs",
    color: str = "#d1495b",
    size: int = 7,
    symbol: str = "circle",
    outline: str = "white",
) -> go.Scatter:
    """Markers for subplot column col.

    Args:
        mach: Mach numbers.
        aoa: Angles of attack.
        col: 1-based subplot column.
        name: Legend name.
        color: Marker colour.
        size: Marker size.
        symbol: Marker symbol.
        outline: Marker outline colour.

    Returns:
        Scatter trace bound to that column's axes.
    """
    suffix = "" if col == 1 else str(col)
    return go.Scatter(
        x=mach,
        y=aoa,
        mode="markers",
        name=name,
        xaxis=f"x{suffix}",
        yaxis=f"y{suffix}",
        showlegend=col == 1,
        marker={"color": color, "size": size, "symbol": symbol, "line": {"color": outline, "width": 1.5}},
        hovertemplate="Mach %{x:.2f}<br>AoA %{y:.1f}°<extra></extra>",
    )
