#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob, csv, pickle, textwrap
from typing import Any, Dict, List, Tuple, Optional

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(BASE_DIR, "res")

# ---------- safe unpickler (numpy._core remap) ----------
class RemappingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)

def load_pickle_remap(path: str) -> Any:
    with open(path, "rb") as f:
        return RemappingUnpickler(f).load()

# ---------- load runs ----------
RUNS: Dict[str, Dict[str, Any]] = {}
for pkl in sorted(glob.glob(os.path.join(RESULTS_ROOT, "gdv_*", "gdv.pkl"))):
    run = os.path.basename(os.path.dirname(pkl))
    obj = load_pickle_remap(pkl)
    if not all(k in obj for k in ["sorted_layers","layer_data","gdv_per_layer"]):
        continue
    # group layers by modality
    by_mod = {"vision": [], "language": []}
    for k in obj["sorted_layers"]:
        m = str(k).split("_", 1)[0]
        if m in by_mod: by_mod[m].append(k)
    obj["_layers_by_mod"] = by_mod
    obj["_plots_root"] = os.path.join(os.path.dirname(pkl), "plots")
    RUNS[run] = obj

if not RUNS:
    raise FileNotFoundError("No runs found under results/gdv_*/gdv.pkl")

RUN_NAMES = list(RUNS.keys())

# ---------- helpers ----------
def layer_dir_for_key(plots_root: str, modality: str, key: str) -> Optional[str]:
    # key: language_42_D5120 -> dir name like language_42__D=5120_
    try:
        base, d = key.split("_D")
        mod, num = base.split("_", 1)
        num = int(num); D = int(d)
    except Exception:
        return None
    mod_dir = os.path.join(plots_root, modality)
    patt = re.compile(rf"^{re.escape(mod)}[_-]{num}.*D[=_]{D}\b", re.IGNORECASE)
    for dpath in glob.glob(os.path.join(mod_dir, "*")):
        if patt.search(os.path.basename(dpath)):
            return dpath
    return None

def read_algo_summary(layer_dir: str, algo: str) -> List[Dict[str, str]]:
    """
    Returns list of runs for UMAP/TSNE:
      [{"tag": "...", "coords": "/abs/path/to_coords.csv"} ...]
    Falls back to empty if no summary/coords exist.
    """
    sub = "UMAP" if algo.lower()=="umap" else "TSNE"
    subdir = os.path.join(layer_dir, sub)
    summary = os.path.join(subdir, "_summary.csv")
    runs = []
    if os.path.exists(summary):
        with open(summary, "r", newline="") as f:
            rows = list(csv.reader(f))
        if rows:
            header = rows[0]
            idx_tag = header.index("Setup_Tag") if "Setup_Tag" in header else None
            idx_coords = header.index("Coords_CSV") if "Coords_CSV" in header else None
            for r in rows[1:]:
                tag = r[idx_tag] if idx_tag is not None else ""
                coords = r[idx_coords] if idx_coords is not None else ""
                if coords:
                    coords_abs = coords if os.path.isabs(coords) else os.path.join(BASE_DIR, coords)
                    runs.append({"tag": tag, "coords": coords_abs})
    return runs

def read_coords_csv(path: str) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path): return None
    try:
        arr = np.genfromtxt(path, delimiter=",", names=True)
        # try common column names
        for c1, c2 in [("x","y"), ("Dim1","Dim2"), ("PC1","PC2")]:
            if c1 in arr.dtype.names and c2 in arr.dtype.names:
                return np.column_stack([arr[c1], arr[c2]]).astype(float)
    except Exception:
        return None
    return None

def get_labels_texts(layer_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    labels = layer_data.get("labels", [])
    texts  = layer_data.get("texts", [])
    # pretty labels (short)
    lab_map = {
        "Not Interesting":"Not",
        "Slightly Interesting":"Slightly",
        "Moderately Interesting":"Moderate",
        "Very Interesting":"Very",
        "Extremely Interesting":"Extreme",
    }
    labels_pretty = [lab_map.get(str(x), str(x)) for x in labels]
    return labels_pretty, [str(t) for t in texts]

# if later you add PCA coords CSVs, list their filenames here (the app auto-detects)
PCA_FILES = [
    ("PCA2 (PC1–PC2)",      "PCA2__coords.csv",       ("PC1","PC2")),
    ("PCA4 (PC1–PC2)",      "PCA4_PC1_PC2__coords.csv", ("PC1","PC2")),
    ("PCA4 (PC2–PC3)",      "PCA4_PC2_PC3__coords.csv", ("PC2","PC3")),
    ("PCA4 (PC3–PC4)",      "PCA4_PC3_PC4__coords.csv", ("PC3","PC4")),
]

# ---------- UI ----------
def side(mod: str, title: str, slots: List[str]) -> html.Div:
    return html.Div([
        html.H3(title, style={"textAlign":"center","margin":"6px 0 10px"}),
        html.Div([
            html.Div("Layer", style={"fontWeight":"600"}),
            dcc.Slider(id=f"{mod}-layer", min=1, max=1, step=1, value=1,
                       tooltip={"placement":"bottom"})
        ], style={"marginBottom":"10px"}),
        *[
            html.Div([
                dcc.Dropdown(id=f"{mod}-choice-{s}", options=[], value=None,
                             clearable=False, style={"marginBottom":"6px"}),
                dcc.Graph(id=f"{mod}-plot-{s}", config={"displayModeBar": False})
            ], style={"marginBottom":"14px"})
            for s in slots
        ]
    ], style={"border":"1px solid #ddd","borderRadius":"10px","padding":"10px","background":"#fafafa"})

LEFT_SLOTS  = ["a","b","c"]   # 3 vision rows
RIGHT_SLOTS = ["a","b","c"]   # 3 language rows

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Interactive Embedding Viewer (GDV/PCA • UMAP • t-SNE)", style={"textAlign":"center"}),
    html.Div([
        html.Div("Run", style={"fontWeight":"600"}),
        dcc.Dropdown(id="run", options=[{"label":n,"value":n} for n in RUN_NAMES],
                     value=RUN_NAMES[0], clearable=False, style={"minWidth":"320px"})
    ], style={"display":"flex","justifyContent":"center","marginBottom":"12px"}),
    html.Div([
        side("vision", "Vision", LEFT_SLOTS),
        side("language", "Language", RIGHT_SLOTS),
    ], style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"14px","padding":"0 8px"})
])

# ---------- logic ----------
def build_options_for_layer(run_name: str, modality: str, layer_1b: int) -> Tuple[List[Dict[str,str]], str]:
    run = RUNS[run_name]
    keys = run["_layers_by_mod"].get(modality, [])
    if not keys:
        return [], None
    idx0 = max(0, min(layer_1b-1, len(keys)-1))
    k = keys[idx0]
    ld = run["layer_data"][k]
    plots_root = run["_plots_root"]
    ldir = layer_dir_for_key(plots_root, modality, k)

    options: List[Dict[str,str]] = []

    # PCA options
    # 1) Always available: PC1–PC2 from gdv.pkl ('x','y')
    options.append({"label": "PCA2 (PC1–PC2)", "value": f"pca|gdv|{k}"})
    # 2) Optional: if you saved PCA coords CSVs, expose them
    if ldir and os.path.isdir(ldir):
        for label, rel, _cols in PCA_FILES[1:]:
            p = os.path.join(ldir, rel)
            if os.path.exists(p):
                options.append({"label": label, "value": f"pca|csv|{p}"})

    # UMAP / TSNE options (from per-layer summaries)
    if ldir and os.path.isdir(ldir):
        umap_runs = read_algo_summary(ldir, "umap")
        tsne_runs = read_algo_summary(ldir, "tsne")
        for i, r in enumerate(umap_runs):
            tag = r["tag"] or f"UMAP #{i+1}"
            options.append({"label": f"UMAP — {tag}", "value": f"umap|{r['coords']}"})
        for i, r in enumerate(tsne_runs):
            tag = r["tag"] or f"t-SNE #{i+1}"
            options.append({"label": f"t-SNE — {tag}", "value": f"tsne|{r['coords']}"})

    default = options[0]["value"] if options else None
    return options, default

def plot_from_choice(run_name: str, modality: str, layer_1b: int, choice: str) -> go.Figure:
    run = RUNS[run_name]
    keys = run["_layers_by_mod"].get(modality, [])
    idx0 = max(0, min(layer_1b-1, len(keys)-1))
    k = keys[idx0]
    ld = run["layer_data"][k]
    labels, texts = get_labels_texts(ld)

    algo, src, path = choice.split("|", 2)

    if algo == "pca" and src == "gdv":
        x = np.array(ld.get("x", []), float)
        y = np.array(ld.get("y", []), float)
        coords = np.column_stack([x, y])
        title = f"{layer_key_nice(k)} • PCA(PC1–PC2) • GDV={ld.get('gdv_euclidean', float('nan')):.4f}"
    elif algo == "pca" and src == "csv":
        coords = read_coords_csv(path)
        title = f"{layer_key_nice(k)} • PCA (from CSV)"
    else:
        coords = read_coords_csv(path)
        title = f"{layer_key_nice(k)} • {algo.upper()}"

    if coords is None or coords.shape[0] != len(labels):
        fig = go.Figure()
        fig.update_layout(title="No coordinates available for this selection.",
                          xaxis_title="Dim 1", yaxis_title="Dim 2", height=420)
        return fig

    # colors per label
    uniq = sorted(set(labels))
    color_map = {lab: i for i, lab in enumerate(uniq)}
    palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f"]

    fig = go.Figure()
    for lab in uniq:
        m = np.array([l == lab for l in labels])
        fig.add_trace(go.Scatter(
            x=coords[m, 0], y=coords[m, 1], mode="markers",
            marker=dict(size=9, color=palette[color_map[lab] % len(palette)]),
            name=str(lab),
            text=[textwrap.shorten(t, 120, placeholder="…") for t in np.array(texts, dtype=object)[m]],
            hovertemplate="<b>%{text}</b><extra></extra>",
        ))
    fig.update_layout(
        title=title, xaxis_title="Dim 1", yaxis_title="Dim 2",
        legend_title="Label", height=420, margin=dict(l=40,r=20,t=60,b=50),
        hoverlabel=dict(bgcolor="rgba(255,255,255,0.95)", font_size=12)
    )
    return fig

def layer_key_nice(k: str) -> str:
    try:
        base, d = k.split("_D")
        mod, num = base.split("_", 1)
        return f"{mod} {int(num)} (D={int(d)})"
    except Exception:
        return k

def register_callbacks(mod: str, slots: List[str]):
    # init slider bounds
    @app.callback(
        Output(f"{mod}-layer", "min"),
        Output(f"{mod}-layer", "max"),
        Output(f"{mod}-layer", "value"),
        Input("run", "value"),
        prevent_initial_call=False
    )
    def _init_slider(run_name):
        keys = RUNS[run_name]["_layers_by_mod"].get(mod, [])
        n = max(1, len(keys))
        return 1, n, 1

    # fill dropdowns
    for s in slots:
        @app.callback(
            Output(f"{mod}-choice-{s}", "options"),
            Output(f"{mod}-choice-{s}", "value"),
            Input("run", "value"),
            Input(f"{mod}-layer", "value"),
            prevent_initial_call=False
        )
        def _fill(run_name, layer_1b, _s=s):
            opts, default = build_options_for_layer(run_name, mod, int(layer_1b or 1))
            return opts, default

        @app.callback(
            Output(f"{mod}-plot-{s}", "figure"),
            Input("run", "value"),
            Input(f"{mod}-layer", "value"),
            Input(f"{mod}-choice-{s}", "value"),
            prevent_initial_call=False
        )
        def _draw(run_name, layer_1b, choice, _s=s):
            if not choice:
                return go.Figure()
            return plot_from_choice(run_name, mod, int(layer_1b or 1), choice)

register_callbacks("vision", LEFT_SLOTS)
register_callbacks("language", RIGHT_SLOTS)

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
