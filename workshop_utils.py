"""Plumbing for the surrogate-modeling workshop notebook.

Constants, data loading and Plotly rendering live here so the notebook only
shows the parts that are being taught. Keep this file next to ``workshop.ipynb``
with the CSVs in a ``data/`` folder beside it, both in ``docs/source/`` and in
the repository the live (Thebe) kernel starts from.
"""

import uuid
import warnings
from enum import IntEnum, ReprEnum, StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import HTML, display
from numpy.typing import NDArray
from plotly.offline import get_plotlyjs
from sklearn.exceptions import ConvergenceWarning

__all__ = [
    "FEATURES",
    "TRAIN_SIZES",
    "Acquisition",
    "Aero",
    "Files",
    "FloatArray",
    "Sizes",
    "Trace",
    "animate",
    "load",
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
        LHS: Runs in the initial Latin hypercube (the first rows of the DOE).
        DEMO: DOE rows used for the single-surrogate figures and exercises.
        BO_INIT: BO initial design size (the Latin hypercube).
        BO_ITER: BO acquisition steps.
        BO_SEED: Observation-noise seed of the BO run.
        HEIGHT: Figure height in pixels.
    """

    GRID = 60
    LHS = 24
    DEMO = 72
    BO_INIT = 24
    BO_ITER = 16
    BO_SEED = 5
    HEIGHT = 480


class Acquisition(StrEnum):
    """Acquisition rules for run_bo.

    Attributes:
        EI: Expected improvement.
        PI: Probability of improvement.
        UCB: Upper confidence bound.
    """

    EI = "EI"
    PI = "PI"
    UCB = "UCB"


TRAIN_SIZES = (24, 36, 48, 72)
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