#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob, pickle, textwrap
from typing import Any, Dict, List, Tuple, Optional

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np

from flask import send_from_directory, abort, jsonify
from werkzeug.utils import safe_join
from werkzeug.middleware.proxy_fix import ProxyFix

# ─────────────────────────────────────────────────────────────
# Paths (env-driven so gunicorn finds your data)
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.environ.get("DASH_RESULTS_ROOT", os.path.join(BASE_DIR, "res"))
print(f"[app] Using RESULTS_ROOT={RESULTS_ROOT}")

# ─────────────────────────────────────────────────────────────
# Safe unpickler (numpy._core → numpy.core)
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
# Load runs ONCE
# ─────────────────────────────────────────────────────────────
RUNS: Dict[str, Dict[str, Any]] = {}
pkl_paths = sorted(glob.glob(os.path.join(RESULTS_ROOT, "gdv_*", "gdv.pkl")))
print(f"[app] scanning for runs in {RESULTS_ROOT!r}")
print(f"[app] found {len(pkl_paths)} candidate pkls")
for pkl in pkl_paths:
    try:
        run = os.path.basename(os.path.dirname(pkl))
        print(f"[app] loading: {pkl}")
        obj = load_pickle_remap(pkl)
        if not all(k in obj for k in ("sorted_layers","layer_data","gdv_per_layer")):
            print(f"[app][WARN] {pkl}: missing keys; skipping")
            continue
        by_mod = {"vision": [], "language": []}
        for key in obj["sorted_layers"]:
            mod = str(key).split("_", 1)[0]
            if mod in by_mod:
                by_mod[mod].append(key)
        obj["_layers_by_mod"] = by_mod
        print(f"[app]   layers: vision={len(by_mod['vision'])}, language={len(by_mod['language'])}")
        RUNS[run] = obj
    except Exception as e:
        print(f"[app][ERROR] failed to load {pkl}: {e!r}")

print(f"[app] RUNS loaded: {sorted(RUNS.keys())}")
# after: print(f"[app] RUNS loaded: {sorted(RUNS.keys())}")
RUN_NAMES = sorted(RUNS.keys())
DEFAULT_RUN = RUN_NAMES[0] if RUN_NAMES else None


# ─────────────────────────────────────────────────────────────
# Single Dash app init (CDN assets to avoid Cloudflare truncation)
# ─────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    serve_locally=False,  # use CDNs for plotly & component bundles
    external_scripts=["https://cdn.plot.ly/plotly-2.30.0.min.js"],
    suppress_callback_exceptions=True,
)
server = app.server

# Make Flask trust Cloudflare/X-Forwarded-* so URLs render correctly
server.wsgi_app = ProxyFix(server.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1)

# Optional: static files from RESULTS_ROOT (PNGs/CSVs if you link them)
@server.route("/files/<path:relpath>")
def serve_res(relpath: str):
    target = safe_join(RESULTS_ROOT, relpath)
    if target is None or not os.path.isfile(target):
        return abort(404)
    root, fname = os.path.dirname(target), os.path.basename(target)
    return send_from_directory(root, fname)

# Healthcheck (quick sanity that backend sees data)
@server.route("/health")
def health():
    return jsonify({
        "runs": sorted(RUNS.keys()),
        "counts": {
            r: {m: len(RUNS[r]["_layers_by_mod"].get(m, [])) for m in ("vision", "language")}
            for r in RUNS
        }
    })


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def layer_key_nice(k: str) -> str:
    try:
        base, d = k.split("_D")
        mod, num = base.split("_", 1)
        return f"{mod} {int(num)} (D={int(d)})"
    except Exception:
        return k

def get_layer_info(run_name: str, modality: str, layer_1b: int) -> Tuple[str, Dict[str, Any]]:
    run = RUNS[run_name]
    keys = run["_layers_by_mod"].get(modality, [])
    if not keys:
        raise ValueError(f"No layers for modality '{modality}' in run '{run_name}'.")
    idx0 = max(0, min(int(layer_1b) - 1, len(keys) - 1))
    key = keys[idx0]
    return key, run["layer_data"][key]

def get_labels_texts_samples(ld: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    if "samples" in ld:
        labels = ld["samples"].get("labels", ld.get("labels", []))
        texts  = ld["samples"].get("texts", ld.get("texts", []))
        fnames = ld["samples"].get("filenames", [])
    else:
        labels = ld.get("labels", [])
        texts  = ld.get("texts", [])
        fnames = []

    lab_map = {
        "Not Interesting":"Not",
        "Slightly Interesting":"Slightly",
        "Moderately Interesting":"Moderate",
        "Very Interesting":"Very",
        "Extremely Interesting":"Extreme",
    }
    labels_pretty = [lab_map.get(str(x), str(x)) for x in labels]
    return labels_pretty, [str(t) for t in texts], [str(f) for f in fnames]

def available_choices(ld: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Build dropdown options from projections stored in gdv.pkl:
      - PCA views present in ld['projections']['pca'] (PC12/PC23/PC34)
      - All UMAP tags in ld['projections']['umap']
      - All TSNE tags in ld['projections']['tsne']
    Value schema: "<algo>|<tag>"
    """
    opts: List[Dict[str, str]] = []
    proj = ld.get("projections", {})

    for tag in ["PC12", "PC23", "PC34"]:
        if "pca" in proj and tag in proj["pca"] and isinstance(proj["pca"][tag], dict):
            opts.append({"label": f"PCA — {tag.replace('PC', 'PC ')}", "value": f"pca|{tag}"})
    for tag in sorted(proj.get("umap", {}).keys()):
        opts.append({"label": f"UMAP — {tag}", "value": f"umap|{tag}"})
    for tag in sorted(proj.get("tsne", {}).keys()):
        opts.append({"label": f"t-SNE — {tag}", "value": f"tsne|{tag}"})

    return opts

def coords_from_choice(ld: Dict[str, Any], choice: str) -> Optional[np.ndarray]:
    algo, tag = choice.split("|", 1)
    proj = ld.get("projections", {})
    if algo == "pca":
        d = proj.get("pca", {}).get(tag, {})
        arr = d.get("coords")
    elif algo == "umap":
        d = proj.get("umap", {}).get(tag, {})
        arr = d.get("coords")
    elif algo == "tsne":
        d = proj.get("tsne", {}).get(tag, {})
        arr = d.get("coords")
    else:
        arr = None

    if arr is None:
        return None
    arr = np.asarray(arr, float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return None
    return arr

def figure_for_layer(run_name: str, modality: str, layer_1b: int, choice: str) -> go.Figure:
    key, ld = get_layer_info(run_name, modality, layer_1b)
    labels, texts, fnames = get_labels_texts_samples(ld)
    coords = coords_from_choice(ld, choice)

    if coords is None or len(labels) == 0 or coords.shape[0] != len(labels):
        msg = "No coordinates available for this selection." if coords is None else \
              "Coordinates/labels length mismatch — cannot plot."
        fig = go.Figure()
        fig.update_layout(
            title=msg, xaxis_title="Dim 1", yaxis_title="Dim 2",
            height=420, margin=dict(l=40, r=20, t=60, b=50)
        )
        return fig

    algo, tag = choice.split("|", 1)
    algo_label = "PCA" if algo == "pca" else ("UMAP" if algo == "umap" else "t-SNE")
    title = f"{layer_key_nice(key)} • {algo_label} — {tag}"
    if "gdv_euclidean" in ld:
        title += f" • GDV={ld['gdv_euclidean']:.4f}"

    uniq = sorted(set(labels))
    cmap = {lab: i for i, lab in enumerate(uniq)}
    palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728",
               "#9467bd","#8c564b","#e377c2","#7f7f7f",
               "#bcbd22","#17becf"]

    fig = go.Figure()
    labels_arr = np.array(labels, dtype=object)
    texts_arr  = np.array(texts, dtype=object) if texts else np.array([""]*len(labels), dtype=object)
    fnames_arr = np.array(fnames, dtype=object) if fnames else np.array([""]*len(labels), dtype=object)

    for lab in uniq:
        m = (labels_arr == lab)
        hover_txt = []
        for tx, fn in zip(texts_arr[m], fnames_arr[m]):
            short = textwrap.shorten(str(tx), 140, placeholder="…")
            if fn and fn != "":
                hover_txt.append(f"<b>{fn}</b><br>{short}")
            else:
                hover_txt.append(short)

        fig.add_trace(go.Scatter(
            x=coords[m, 0], y=coords[m, 1], mode="markers",
            marker=dict(size=9, color=palette[cmap[lab] % len(palette)]),
            name=str(lab),
            text=hover_txt,
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        title=title, xaxis_title="Dim 1", yaxis_title="Dim 2",
        legend_title="Label", height=420,
        margin=dict(l=40, r=20, t=60, b=50),
        hoverlabel=dict(bgcolor="rgba(255,255,255,0.95)", font_size=12),
    )
    return fig

# ─────────────────────────────────────────────────────────────
# UI (each slot has its own layer slider + dropdown + graph)
# ─────────────────────────────────────────────────────────────
LEFT_SLOTS  = ["a","b","c"]   # Vision rows
RIGHT_SLOTS = ["a","b","c"]   # Language rows

def slot_block(mod: str, slot_id: str) -> html.Div:
    return html.Div([
        html.Div([
            html.Div("Layer", style={"fontWeight":"600", "marginBottom":"4px"}),
            dcc.Slider(
                id=f"{mod}-layer-{slot_id}", min=1, max=1, step=1, value=1,
                tooltip={"placement":"bottom"}
            ),
        ], style={"marginBottom":"8px"}),

        dcc.Dropdown(
            id=f"{mod}-choice-{slot_id}",
            options=[], value=None, clearable=False,
            style={"marginBottom":"6px"}
        ),
        dcc.Graph(id=f"{mod}-plot-{slot_id}", config={"displayModeBar": False})
    ], style={"marginBottom":"14px"})

def side(mod: str, title: str, slots: List[str]) -> html.Div:
    return html.Div([
        html.H3(title, style={"textAlign":"center","margin":"6px 0 10px"}),
        *[slot_block(mod, s) for s in slots]
    ], style={"border":"1px solid #ddd","borderRadius":"10px","padding":"10px","background":"#fafafa"})

app.layout = html.Div([
    html.H1("Interactive Embedding Viewer (PCA • UMAP • t-SNE)", style={"textAlign":"center"}),

    html.Div([
        html.Div("Run", style={"fontWeight":"600"}),
        dcc.Dropdown(
            id="run",
            options=[{"label": n, "value": n} for n in RUN_NAMES],
            value=RUN_NAMES[0], clearable=False, style={"minWidth":"320px"}
        )
    ], style={"display":"flex","justifyContent":"center","marginBottom":"12px"}),

    html.Div([
        side("vision",   "Vision",   LEFT_SLOTS),
        side("language", "Language", RIGHT_SLOTS),
    ], style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"14px","padding":"0 8px"})
])

# ─────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────
def register_callbacks(mod: str, slots: List[str]):
    # For each slot: init its slider bounds based on the selected run
    for s in slots:
        @app.callback(
            Output(f"{mod}-layer-{s}", "min"),
            Output(f"{mod}-layer-{s}", "max"),
            Output(f"{mod}-layer-{s}", "value"),
            Input("run", "value"),
            prevent_initial_call=False
        )
        def _init_slot_slider(run_name, _s=s):
            keys = RUNS[run_name]["_layers_by_mod"].get(mod, [])
            n = max(1, len(keys))
            return 1, n, 1

        # Populate the choices for this slot's dropdown; keep last selection if possible
        @app.callback(
            Output(f"{mod}-choice-{s}", "options"),
            Output(f"{mod}-choice-{s}", "value"),
            Input("run", "value"),
            Input(f"{mod}-layer-{s}", "value"),
            State(f"{mod}-choice-{s}", "value"),
            prevent_initial_call=False
        )
        def _fill(run_name, layer_1b, current_value, _s=s):
            try:
                _, ld = get_layer_info(run_name, mod, int(layer_1b or 1))
                opts = available_choices(ld)
                if not opts:
                    return [], None
                values = {o["value"] for o in opts}

                # keep exact selection if still valid
                if current_value in values:
                    return opts, current_value

                # else keep same algo family (pca/umap/tsne) if available
                if current_value:
                    try:
                        cur_algo = current_value.split("|", 1)[0]
                        for o in opts:
                            if o["value"].startswith(cur_algo + "|"):
                                return opts, o["value"]
                    except Exception:
                        pass

                # fallback to first option
                return opts, opts[0]["value"]
            except Exception:
                return [], None

        # Draw this slot's figure
        @app.callback(
            Output(f"{mod}-plot-{s}", "figure"),
            Input("run", "value"),
            Input(f"{mod}-layer-{s}", "value"),
            Input(f"{mod}-choice-{s}", "value"),
            prevent_initial_call=False
        )
        def _draw(run_name, layer_1b, choice, _s=s):
            if not choice:
                return go.Figure()
            try:
                return figure_for_layer(run_name, mod, int(layer_1b or 1), choice)
            except Exception as e:
                fig = go.Figure()
                fig.update_layout(
                    title=f"Error: {e}", xaxis_title="Dim 1", yaxis_title="Dim 2",
                    height=420, margin=dict(l=40, r=20, t=60, b=50)
                )
                return fig

register_callbacks("vision", LEFT_SLOTS)
register_callbacks("language", RIGHT_SLOTS)

# ─────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
