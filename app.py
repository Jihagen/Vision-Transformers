#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, pickle, textwrap
from typing import Any, Dict, List, Optional, Tuple

import dash
from dash import dcc, html
try:
    from dash import Input, Output, State, ALL, ctx, no_update
except ImportError:
    from dash.dependencies import Input, Output, State, ALL
    from dash import no_update
    ctx = dash.callback_context
import plotly.graph_objects as go
import numpy as np
from flask import send_from_directory

# ─────────────────────────────────────────────────────────────
# Absolute paths
# ─────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
HYPOTHESES_ROOT   = os.path.join(BASE_DIR, "new_results", "hypotheses")
IMAGES_ROOT       = os.path.join(BASE_DIR, "data", "imagesDemographics")

# ─────────────────────────────────────────────────────────────
# Safe unpickler: remap numpy._core.* → numpy.core.*
# ─────────────────────────────────────────────────────────────
class RemappingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)

def load_pickle_remap(path: str) -> Any:
    with open(path, "rb") as f:
        return RemappingUnpickler(f).load()

# ─────────────────────────────────────────────────────────────
# Discover runs under new_results/hypotheses/
# ─────────────────────────────────────────────────────────────
MODEL_PKLS: Dict[str, str] = {}
for pkl_path in sorted(glob.glob(os.path.join(HYPOTHESES_ROOT, "*", "gdv.pkl"))):
    MODEL_PKLS[os.path.basename(os.path.dirname(pkl_path))] = pkl_path
if not MODEL_PKLS:
    raise FileNotFoundError(
        f"No gdv.pkl found under {HYPOTHESES_ROOT}/*/"
    )

# ─────────────────────────────────────────────────────────────
# Load all runs
# ─────────────────────────────────────────────────────────────
all_data: Dict[str, Dict[str, Any]] = {}
load_errors: Dict[str, str] = {}
for name, pth in MODEL_PKLS.items():
    try:
        data = load_pickle_remap(pth)
        for k in ["sorted_layers", "layer_data", "gdv_per_layer"]:
            if k not in data:
                raise KeyError(f"{name}: missing '{k}' in {pth}")

        # Best (lowest GDV) layer
        gdv_vals = np.array(
            [data["gdv_per_layer"][L] for L in data["sorted_layers"]], float
        )
        best_idx = int(np.argmin(gdv_vals))
        data["best_layer_idx"] = best_idx
        data["best_layer_key"] = data["sorted_layers"][best_idx]

        # Global axis range across all layers (for static-axes mode)
        coords = []
        for L in data["sorted_layers"]:
            ld = data["layer_data"][L]
            coords += list(ld["x"]) + list(ld["y"])
        arr = np.array(coords, float)
        data["static_min"], data["static_max"] = (
            (float(np.nanmin(arr)), float(np.nanmax(arr))) if arr.size else (-1.0, 1.0)
        )

        # Image filenames — shared across layers, stored in samples dict of first layer
        first_ld = data["layer_data"][data["sorted_layers"][0]]
        samples = first_ld.get("samples", {})
        data["filenames"] = list(samples.get("filenames", []))  # e.g. ["0881.jpg", ...]
        data["static_square_ranges"] = {}

        all_data[name] = data
    except Exception as exc:
        load_errors[name] = f"{type(exc).__name__}: {exc}"

if not all_data:
    error_lines = [
        "No valid hypothesis runs could be loaded from "
        f"{HYPOTHESES_ROOT}.",
        "The following gdv.pkl files failed to load:",
    ]
    error_lines.extend(
        f"- {name}: {message}" for name, message in sorted(load_errors.items())
    )
    raise RuntimeError("\n".join(error_lines))

# ─────────────────────────────────────────────────────────────
# Image URL helper
# ─────────────────────────────────────────────────────────────
def image_url(fname: str) -> str:
    """Return the served URL for a filename, or '' if missing."""
    if not fname:
        return ""
    if os.path.exists(os.path.join(IMAGES_ROOT, fname)):
        return f"/_images/{fname}"
    return ""


def projection_options(ld: Dict[str, Any]) -> List[Dict[str, str]]:
    options = [{"label": "GDV", "value": "gdv"}]
    projections = ld.get("projections", {})

    for tag in sorted(projections.get("umap", {})):
        options.append({"label": f"UMAP - {tag}", "value": f"umap|{tag}"})
    for tag in sorted(projections.get("tsne", {})):
        options.append({"label": f"TSNE - {tag}", "value": f"tsne|{tag}"})

    return options


def projection_label(choice: Optional[str]) -> str:
    if not choice or choice == "gdv":
        return "GDV"

    algo, tag = choice.split("|", 1)
    return f"{algo.upper()} - {tag}"


def get_projection_xy(
    ld: Dict[str, Any], projection_choice: Optional[str], *, allow_fallback: bool = True
) -> Tuple[np.ndarray, np.ndarray, str, str]:
    if not projection_choice or projection_choice == "gdv":
        return np.array(ld["x"], float), np.array(ld["y"], float), "PC 1", "PC 2"

    projections = ld.get("projections", {})
    algo, tag = projection_choice.split("|", 1)
    node = projections.get(algo, {}).get(tag, {})
    coords = node.get("coords") if isinstance(node, dict) else None

    if coords is None:
        if allow_fallback:
            return get_projection_xy(ld, "gdv", allow_fallback=False)
        return np.array([], float), np.array([], float), "Dim 1", "Dim 2"

    arr = np.asarray(coords, float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        if allow_fallback:
            return get_projection_xy(ld, "gdv", allow_fallback=False)
        return np.array([], float), np.array([], float), "Dim 1", "Dim 2"

    return arr[:, 0], arr[:, 1], "Dim 1", "Dim 2"


def square_ranges(
    x_vals: np.ndarray, y_vals: np.ndarray, pad_fraction: float = 0.05
) -> Tuple[List[float], List[float]]:
    x_arr = np.asarray(x_vals, float)
    y_arr = np.asarray(y_vals, float)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)

    if not finite.any():
        return [-1.0, 1.0], [-1.0, 1.0]

    x_arr = x_arr[finite]
    y_arr = y_arr[finite]

    x_min, x_max = float(np.min(x_arr)), float(np.max(x_arr))
    y_min, y_max = float(np.min(y_arr)), float(np.max(y_arr))
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)

    span = max(x_max - x_min, y_max - y_min)
    if not np.isfinite(span) or span <= 0:
        span = 1.0

    half_span = 0.5 * span * (1.0 + pad_fraction)
    return [x_mid - half_span, x_mid + half_span], [y_mid - half_span, y_mid + half_span]


def square_ranges_from_viewport(viewport: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    x0, x1 = float(viewport["x0"]), float(viewport["x1"])
    y0, y1 = float(viewport["y0"]), float(viewport["y1"])
    x_mid = 0.5 * (x0 + x1)
    y_mid = 0.5 * (y0 + y1)
    half_span = 0.5 * max(abs(x1 - x0), abs(y1 - y0), 1e-9)
    return [x_mid - half_span, x_mid + half_span], [y_mid - half_span, y_mid + half_span]


def get_static_square_ranges(
    model_data: Dict[str, Any], projection_choice: Optional[str]
) -> Tuple[List[float], List[float]]:
    cache = model_data["static_square_ranges"]
    cache_key = projection_choice or "gdv"

    if cache_key not in cache:
        x_all: List[float] = []
        y_all: List[float] = []
        for layer_key in model_data["sorted_layers"]:
            ld = model_data["layer_data"][layer_key]
            x_vals, y_vals, _, _ = get_projection_xy(
                ld, projection_choice, allow_fallback=False
            )
            if x_vals.size and y_vals.size:
                x_all.extend(x_vals.tolist())
                y_all.extend(y_vals.tolist())
        cache[cache_key] = square_ranges(np.array(x_all, float), np.array(y_all, float))

    return cache[cache_key]

# ─────────────────────────────────────────────────────────────
# Dash app + image route
# ─────────────────────────────────────────────────────────────
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

@server.route("/_images/<path:filename>")
def _serve_image(filename):
    return send_from_directory(IMAGES_ROOT, filename)

model_names    = list(all_data.keys())
default_model  = model_names[0]
_d0            = all_data[default_model]
default_n_lay  = len(_d0["sorted_layers"])
default_best   = _d0["best_layer_idx"] + 1
default_projection_options = projection_options(
    _d0["layer_data"][_d0["sorted_layers"][default_best - 1]]
)

# ─────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────
app.layout = html.Div([

    # Header
    html.H1("Activation Explorer", style={
        "textAlign": "center", "marginBottom": "0.4rem",
        "fontSize": "1.5rem", "fontWeight": "600",
    }),

    # Controls
    html.Div([
        html.Div([
            html.Label("Run:", style={"fontWeight": "bold", "whiteSpace": "nowrap"}),
            dcc.Dropdown(
                id="model-selector",
                options=[{"label": n, "value": n} for n in model_names],
                value=default_model, clearable=False,
                style={"minWidth": "220px"},
            ),
        ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
        html.Div([
            dcc.Checklist(
                id="axis-mode",
                options=[{"label": " Static axes", "value": "static"}],
                value=[],
                labelStyle={"display": "inline-block"},
            ),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={
        "display": "flex", "justifyContent": "center",
        "gap": "2rem", "marginBottom": "0.6rem", "flexWrap": "wrap",
    }),

    # GDV / layer info bar (replaces persona bar)
    html.Div(id="layer-info-bar", style={
        "textAlign": "center", "color": "#555",
        "fontSize": "0.84em", "marginBottom": "0.6rem",
    }),

    # ── Main row: square scatter | image gallery ─────────────
    html.Div([

        # Left: square scatter
        html.Div([
            html.Div([
                html.Label("View:", style={"fontWeight": "bold", "whiteSpace": "nowrap"}),
                dcc.Dropdown(
                    id="projection-selector",
                    options=default_projection_options,
                    value="gdv",
                    clearable=False,
                    style={"flex": "1", "minWidth": "0"},
                ),
            ], style={
                "display": "flex", "alignItems": "center",
                "gap": "8px", "marginBottom": "0.5rem",
            }),
            html.Div(
                dcc.Graph(
                    id="main-graph",
                    config={
                        "displayModeBar": True,
                        "scrollZoom": True,
                        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                    },
                    style={"width": "100%", "height": "100%"},
                ),
                style={"flex": "1", "minHeight": "0"},
            ),
        ],
            id="scatter-container",
            style={
                "flex": "0 0 auto",
                "width":  "min(52vw, 580px)",
                "height": "min(52vw, 580px)",
                "display": "flex",
                "flexDirection": "column",
            },
        ),

        # Right: image gallery
        html.Div([
            html.Div(id="gallery-header", style={
                "fontSize": "0.78em", "color": "#888",
                "marginBottom": "5px", "fontStyle": "italic",
            }),
            html.Div(
                id="image-gallery",
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fill, minmax(90px, 1fr))",
                    "gap": "4px",
                    "overflowY": "auto",
                    "maxHeight": "min(52vw, 580px)",
                    "padding": "2px",
                },
            ),
        ], style={"flex": "1", "minWidth": "0", "padding": "0 0.8rem"}),

    ], style={
        "display": "flex", "alignItems": "flex-start",
        "gap": "0.5rem", "padding": "0 1rem",
    }),

    # Layer slider
    html.Div([
        html.Label("Layer:", style={
            "fontWeight": "bold", "whiteSpace": "nowrap", "flexShrink": "0",
        }),
        html.Div(
            dcc.Slider(
                id="layer-slider",
                min=1, max=default_n_lay, step=1, value=default_best,
                marks={}, updatemode="drag",
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            style={"flex": "1", "minWidth": "0"},
        ),
    ], style={
        "display": "flex", "alignItems": "center",
        "gap": "12px", "padding": "0.6rem 2rem 0 2rem",
    }),

    # Stores
    dcc.Store(id="viewport-store", data={}),
    dcc.Store(id="selected-point-store", data=None),
    dcc.Store(id="hidden-labels-store", data=[]),

], style={
    "maxWidth": "1500px", "margin": "0 auto",
    "fontFamily": "system-ui, sans-serif", "padding": "0.6rem",
})


# ─────────────────────────────────────────────────────────────
# Callback: update slider when model changes
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output("layer-slider", "max"),
    Output("layer-slider", "marks"),
    Output("layer-slider", "value"),
    Input("model-selector", "value"),
)
def update_slider(model_name):
    data    = all_data[model_name]
    n       = len(data["sorted_layers"])
    best    = data["best_layer_idx"] + 1
    marks   = {i + 1: "" for i in range(n)}
    marks[best] = {
        "label": f"★{best}",
        "style": {"color": "#c0392b", "fontWeight": "bold"},
    }
    return n, marks, best


@app.callback(
    Output("projection-selector", "options"),
    Output("projection-selector", "value"),
    Input("layer-slider", "value"),
    Input("model-selector", "value"),
    State("projection-selector", "value"),
)
def update_projection_selector(layer_1based, model_name, current_value):
    if layer_1based is None:
        layer_1based = 1

    data = all_data[model_name]
    idx0 = int(layer_1based) - 1
    layer_key = data["sorted_layers"][idx0]
    options = projection_options(data["layer_data"][layer_key])
    valid_values = {opt["value"] for opt in options}
    value = current_value if current_value in valid_values else "gdv"
    return options, value


# ─────────────────────────────────────────────────────────────
# Callback: layer info bar (GDV + layer key)
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output("layer-info-bar", "children"),
    Input("layer-slider", "value"),
    Input("model-selector", "value"),
    Input("projection-selector", "value"),
)
def update_info_bar(layer_1based, model_name, projection_choice):
    if layer_1based is None:
        return ""
    data  = all_data[model_name]
    idx0  = int(layer_1based) - 1
    Lkey  = data["sorted_layers"][idx0]
    gdv   = float(data["gdv_per_layer"][Lkey])
    best  = data["best_layer_key"]
    bgdv  = float(data["gdv_per_layer"][best])
    n     = len(data["sorted_layers"])
    return (
        f"Layer {layer_1based}/{n}:  {Lkey}  |  GDV = {gdv:.4f}  "
        f"  ·  View: {projection_label(projection_choice)}"
        f"  ·  Best: {best}  (GDV = {bgdv:.4f})"
    )


# ─────────────────────────────────────────────────────────────
# Callback: accumulate viewport on zoom/pan; reset on model change
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output("viewport-store", "data"),
    Input("main-graph", "relayoutData"),
    Input("model-selector", "value"),
    Input("projection-selector", "value"),
    State("viewport-store", "data"),
    prevent_initial_call=True,
)
def update_viewport(relayout, _model, _projection_choice, current_vp):
    if ctx.triggered_id in {"model-selector", "projection-selector"}:
        return {}
    if relayout is None:
        return current_vp or {}
    if relayout.get("xaxis.autorange") or relayout.get("autosize"):
        return {}
    if "xaxis.range[0]" in relayout:
        return {
            "x0": relayout["xaxis.range[0]"],
            "x1": relayout["xaxis.range[1]"],
            "y0": relayout["yaxis.range[0]"],
            "y1": relayout["yaxis.range[1]"],
        }
    return current_vp or {}


# ─────────────────────────────────────────────────────────────
# Callback: track legend toggles → hidden-labels-store
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output("hidden-labels-store", "data"),
    Input("main-graph", "restyleData"),
    Input("model-selector", "value"),
    State("layer-slider", "value"),       # read current layer but do NOT trigger on changes
    State("hidden-labels-store", "data"),
    prevent_initial_call=True,
)
def update_hidden_labels(restyle_data, model_name, layer_1based, current_hidden):
    # Reset whenever the user switches runs (label sets can differ between runs)
    if ctx.triggered_id == "model-selector":
        return []

    if not restyle_data or not isinstance(restyle_data, list) or len(restyle_data) < 2:
        return current_hidden or []

    prop_dict, trace_indices = restyle_data[0], restyle_data[1]
    if "visible" not in prop_dict:
        return current_hidden or []

    # Map trace index → label name using the same sort order as update_figure
    data  = all_data[model_name]
    idx0  = int(layer_1based or 1) - 1
    Lkey  = data["sorted_layers"][idx0]
    ld    = data["layer_data"][Lkey]
    unique_labels = sorted(set(ld.get("labels", [])))

    hidden = set(current_hidden or [])
    for trace_idx, vis_value in zip(trace_indices, prop_dict["visible"]):
        if trace_idx >= len(unique_labels):
            continue  # highlight trace has showlegend=False; ignore it
        label = unique_labels[trace_idx]
        if vis_value == "legendonly" or vis_value is False:
            hidden.add(label)
        else:
            hidden.discard(label)

    return list(hidden)


# ─────────────────────────────────────────────────────────────
# Colour map for interestingness labels
# ─────────────────────────────────────────────────────────────
LABEL_COLORS = {
    "Not Interesting":       "#607d8b",
    "Slightly Interesting":  "#4285f4",
    "Moderately Interesting":"#34a853",
    "Very Interesting":      "#fbbc04",
    "Extremely Interesting": "#ea4335",
}
FALLBACK_COLORS = ["#9c27b0", "#00bcd4", "#ff9800", "#795548", "#e91e63"]


# ─────────────────────────────────────────────────────────────
# Callback: main scatter figure
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output("main-graph", "figure"),
    Input("layer-slider", "value"),
    Input("model-selector", "value"),
    Input("axis-mode", "value"),
    Input("projection-selector", "value"),
    Input("selected-point-store", "data"),
    Input("viewport-store", "data"),
    prevent_initial_call=False,
)
def update_figure(
    layer_1based, model_name, axis_mode, projection_choice, selected_point, viewport
):
    if layer_1based is None:
        layer_1based = 1

    data = all_data[model_name]
    idx0 = int(layer_1based) - 1
    Lkey = data["sorted_layers"][idx0]
    ld   = data["layer_data"][Lkey]

    x, y, x_title, y_title = get_projection_xy(ld, projection_choice)
    labels = list(ld.get("labels", ["?"] * len(x)))
    texts  = list(ld.get("texts",  [""]  * len(x)))
    fnames = data["filenames"]   # shared across layers

    # customdata: [0] short text, [1] fname, [2] point-index
    short_texts = [textwrap.shorten(str(t), 120, placeholder="…") for t in texts]
    custom = np.column_stack([
        np.array(short_texts, dtype=object),
        np.array(fnames[:len(x)] if fnames else [""] * len(x), dtype=object),
        np.arange(len(x), dtype=object),
    ])

    unique_labels = sorted(set(labels))
    fig = go.Figure()
    for i, lbl in enumerate(unique_labels):
        mask  = np.array([l == lbl for l in labels])
        color = LABEL_COLORS.get(lbl, FALLBACK_COLORS[i % len(FALLBACK_COLORS)])
        fig.add_trace(go.Scatter(
            x=x[mask], y=y[mask],
            mode="markers",
            marker=dict(size=8, color=color, opacity=0.85),
            name=lbl,
            customdata=custom[mask],
            hovertemplate="<b>%{customdata[1]}</b><br>%{customdata[0]}<extra></extra>",
        ))

    # Highlight selected point
    if selected_point is not None:
        si = int(selected_point)
        if 0 <= si < len(x):
            fig.add_trace(go.Scatter(
                x=[float(x[si])], y=[float(y[si])],
                mode="markers",
                marker=dict(
                    size=22, color="rgba(0,0,0,0)",
                    line=dict(color="#ff5722", width=3),
                ),
                showlegend=False, hoverinfo="skip",
            ))

    fig.update_layout(
        margin=dict(l=50, r=20, t=30, b=50),
        autosize=True,
        plot_bgcolor="#f8f8f8",
        paper_bgcolor="white",
        legend=dict(
            title="Label",
            orientation="v",
            x=1.02, y=1,
            xanchor="left", yanchor="top",
            font=dict(size=11),
        ),
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.97)",
            align="left", font_size=12, namelength=-1,
        ),
        uirevision=f"{model_name}:{projection_choice}",
        # ── Equal-scale axes ──────────────────────────────────
        xaxis=dict(
            title=x_title,
            constrain="domain",
            scaleanchor="y",   # lock x to the y axis scale
        ),
        yaxis=dict(
            title=y_title,
            constrain="domain",
        ),
    )

    if "static" in axis_mode:
        x_range, y_range = get_static_square_ranges(data, projection_choice)
    elif viewport and "x0" in viewport:
        x_range, y_range = square_ranges_from_viewport(viewport)
    else:
        x_range, y_range = square_ranges(x, y)

    fig.update_xaxes(range=x_range)
    fig.update_yaxes(range=y_range)

    return fig


# ─────────────────────────────────────────────────────────────
# Callback: image gallery (updates on zoom)
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output("image-gallery", "children"),
    Output("image-gallery", "style"),
    Output("gallery-header", "children"),
    Input("layer-slider", "value"),
    Input("model-selector", "value"),
    Input("projection-selector", "value"),
    Input("viewport-store", "data"),
    Input("selected-point-store", "data"),
    Input("hidden-labels-store", "data"),
    prevent_initial_call=False,
)
def update_gallery(layer_1based, model_name, projection_choice, viewport,
                   selected_point, hidden_labels):
    if layer_1based is None:
        layer_1based = 1

    data   = all_data[model_name]
    idx0   = int(layer_1based) - 1
    Lkey   = data["sorted_layers"][idx0]
    ld     = data["layer_data"][Lkey]
    fnames = data["filenames"]
    labels = list(ld.get("labels", []))

    x, y, _, _ = get_projection_xy(ld, projection_choice)
    n_total = len(x)

    hidden_set = set(hidden_labels or [])

    # Filter by viewport
    if viewport and "x0" in viewport:
        x0, x1 = float(viewport["x0"]), float(viewport["x1"])
        y0, y1 = float(viewport["y0"]), float(viewport["y1"])
        vis_mask = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    else:
        vis_mask = np.ones(n_total, bool)

    # Also exclude points whose label is hidden via legend toggle
    if hidden_set:
        label_mask = np.array([labels[i] not in hidden_set for i in range(n_total)])
        vis_mask = vis_mask & label_mask

    vis_idx   = np.where(vis_mask)[0]
    n_visible = len(vis_idx)

    # Dynamic thumb size: fewer visible → bigger thumbnails
    thumb_min = max(55, min(200, int(500 / max(1, n_visible ** 0.5))))

    gallery_style = {
        "display": "grid",
        "gridTemplateColumns": f"repeat(auto-fill, minmax({thumb_min}px, 1fr))",
        "gap": "4px",
        "overflowY": "auto",
        "maxHeight": "min(52vw, 580px)",
        "padding": "2px",
    }

    imgs = []
    for i in vis_idx:
        fname      = fnames[i] if i < len(fnames) else ""
        url        = image_url(fname)
        is_sel     = (selected_point is not None and int(selected_point) == int(i))

        imgs.append(html.Div([
            html.Img(
                src=url, title=fname,
                style={
                    "width": "100%", "aspectRatio": "1 / 1",
                    "objectFit": "cover", "borderRadius": "3px", "display": "block",
                },
            ),
            html.Div(fname, style={
                "fontSize": "0.62em", "textAlign": "center",
                "overflow": "hidden", "textOverflow": "ellipsis",
                "whiteSpace": "nowrap", "color": "#666", "marginTop": "2px",
            }),
        ],
        id={"type": "thumb", "index": int(i)},
        n_clicks=0,
        style={
            "border":          "3px solid #ff5722" if is_sel else "2px solid transparent",
            "borderRadius":    "5px",
            "padding":         "2px",
            "cursor":          "pointer",
            "backgroundColor": "#fff4f2" if is_sel else "transparent",
        }))

    n_label_hidden = sum(1 for i in range(n_total) if labels[i] in hidden_set)
    n_after_labels = n_total - n_label_hidden
    is_zoomed      = n_visible < n_after_labels

    parts = []
    if hidden_set:
        parts.append(f"{len(hidden_set)} label(s) hidden")
    if is_zoomed:
        parts.append("zoomed")
    hint   = "  •  " + ",  ".join(parts) if parts else "  •  zoom or click legend to filter"
    header = f"{n_visible} of {n_total} images{hint}"
    return imgs, gallery_style, header


# ─────────────────────────────────────────────────────────────
# Callback: click thumbnail → highlight corresponding point
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output("selected-point-store", "data"),
    Input({"type": "thumb", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_point(n_clicks_list):
    if not ctx.triggered or not any(n for n in n_clicks_list if n):
        return no_update
    tid = ctx.triggered_id
    if isinstance(tid, dict) and "index" in tid:
        return tid["index"]
    return no_update


# ─────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(IMAGES_ROOT, exist_ok=True)
    print(f"Loaded {len(all_data)} runs: {list(all_data.keys())}")
    print(f"Serving images from: {IMAGES_ROOT}")
    app.run_server(debug=True)
