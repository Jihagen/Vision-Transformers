#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, pickle, textwrap
from typing import Any, Dict, List, Tuple

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
from flask import send_from_directory

# ─────────────────────────────────────────────────────────────
# Absolute paths (robust no matter your cwd)
# ─────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(BASE_DIR, "results")
IMAGES_ROOT  = os.path.join(BASE_DIR, "data", "imagesDemographics")
PERSONAS_PKL = os.path.join(BASE_DIR, "data", "df_generated-personas-sample.pkl")

# ─────────────────────────────────────────────────────────────
# Fallback filename order (applies to ALL runs)
# ─────────────────────────────────────────────────────────────
GLOBAL_IMAGE_ORDER: List[str] = [
    "0211.jpg", "0881.jpg", "0462.jpg",
    "0754.jpg", "0044.jpg", "0047.jpg", "0129.jpg", "0602.jpg",
    "0149.jpg", "0733.jpg", "0793.jpg", "0858.jpg", "0523.jpg",
    "0852.jpg", "0928.jpg", "0564.jpg", "0842.jpg", "0818.jpg",
    "0821.jpg", "0839.jpg"
]

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
# Personas by row position (0→first row)
# ─────────────────────────────────────────────────────────────
PROMPT_COLS = ["prompt","persona_prompt","instruction","system_prompt","persona_text","text"]

def load_persona_prompts(personas_pkl: str) -> Dict[int, str]:
    if not os.path.exists(personas_pkl):
        return {}
    obj = load_pickle_remap(personas_pkl)
    if hasattr(obj, "columns") and hasattr(obj, "iloc"):
        col = next((c for c in PROMPT_COLS if c in obj.columns), None)
        if col is None:
            return {i: str(obj.iloc[i].to_dict()) for i in range(len(obj))}
        return {i: str(obj.iloc[i][col]) for i in range(len(obj))}
    if isinstance(obj, dict):
        return {i: str(v) for i, v in enumerate(obj.values())}
    try:
        return {i: str(r) for i, r in enumerate(list(obj))}
    except Exception:
        return {}

PERSONA_PROMPTS = load_persona_prompts(PERSONAS_PKL)

# ─────────────────────────────────────────────────────────────
# Discover runs and load gdv.pkl
# ─────────────────────────────────────────────────────────────
def persona_idx_from_name(name: str):
    try: return int(name.split("_")[-1])
    except: return None

MODEL_PKLS: Dict[str,str] = {}
for pkl_path in sorted(glob.glob(os.path.join(RESULTS_ROOT, "gdv_*", "gdv.pkl"))):
    MODEL_PKLS[os.path.basename(os.path.dirname(pkl_path))] = pkl_path
if not MODEL_PKLS:
    raise FileNotFoundError(f"No gdv.pkl under {RESULTS_ROOT}/gdv_*/")

all_data: Dict[str,Dict[str,Any]] = {}
for name, pth in MODEL_PKLS.items():
    data = load_pickle_remap(pth)
    for k in ["sorted_layers","layer_data","gdv_per_layer"]:
        if k not in data: raise KeyError(f"{name}: missing '{k}' in {pth}")
    pid = persona_idx_from_name(name)
    data["persona_idx_from_name"] = pid
    data["persona_prompt_from_name"] = PERSONA_PROMPTS.get(pid, "")
    gdv_vals = np.array([data["gdv_per_layer"][L] for L in data["sorted_layers"]], float)
    best = int(np.argmin(gdv_vals))
    data["best_layer_idx"] = best
    data["best_layer_key"] = data["sorted_layers"][best]
    coords = []
    for L in data["sorted_layers"]:
        ld = data["layer_data"][L]; coords += list(ld["x"]) + list(ld["y"])
    arr = np.array(coords, float)
    data["static_min"], data["static_max"] = (float(np.nanmin(arr)), float(np.nanmax(arr))) if arr.size else (-1.0,1.0)
    data["fallback_filenames_by_order"] = GLOBAL_IMAGE_ORDER
    all_data[name] = data

# ─────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────
IMAGE_NAME_CANDS = ["filename","file","img_file","image_file","img_name","image_name","basename","name"]
IMAGE_PATH_CANDS = ["img_path","image_path","path","filepath"]
IMAGE_ID_CANDS   = ["image_id","img_id","id","index"]

def padded_name(n:int)->str: return f"{n:04d}"

def pick_image_label(ld: Dict[str,Any], i:int, fallback:List[str]) -> Tuple[str,Any,bool]:
    for k in IMAGE_PATH_CANDS:
        if k in ld and i < len(ld[k]) and ld[k][i]:
            v = str(ld[k][i]); return os.path.basename(v), v, False
    for k in IMAGE_NAME_CANDS:
        if k in ld and i < len(ld[k]) and ld[k][i]:
            v = str(ld[k][i]); return v, v, False
    for k in IMAGE_ID_CANDS:
        if k in ld and i < len(ld[k]) and ld[k][i] is not None:
            try: ii = int(ld[k][i]); return padded_name(ii), ii, True
            except: v = str(ld[k][i]); return v, v, False
    if fallback and i < len(fallback):
        v = fallback[i]; return v, v, False
    return "", None, False

def resolve_served_filename(label_or_id: Any) -> Tuple[str,bool]:
    """Return (filename under IMAGES_ROOT, exists)."""
    if label_or_id in [None,""]: return "", False
    if isinstance(label_or_id,(int,np.integer)) or (isinstance(label_or_id,str) and str(label_or_id).isdigit()):
        stem = padded_name(int(label_or_id))
        for ext in (".jpg",".jpeg",".png"):
            f = stem+ext
            if os.path.exists(os.path.join(IMAGES_ROOT,f)): return f, True
        return stem+".jpg", False
    fname = os.path.basename(str(label_or_id))
    abs_p = os.path.join(IMAGES_ROOT,fname)
    if os.path.exists(abs_p): return fname, True
    base, ext = os.path.splitext(abs_p)
    if ext=="":
        for e in (".jpg",".jpeg",".png"):
            cand = os.path.basename(base+e)
            if os.path.exists(os.path.join(IMAGES_ROOT,cand)): return cand, True
    return fname, False

# ─────────────────────────────────────────────────────────────
# Dash app + route to serve images
# ─────────────────────────────────────────────────────────────
app = dash.Dash(__name__)
server = app.server

@server.route("/_images/<path:filename>")
def _serve_image(filename):
    return send_from_directory(IMAGES_ROOT, filename)

# ─────────────────────────────────────────────────────────────
# Build layout
# ─────────────────────────────────────────────────────────────
panels = []
for name, data in all_data.items():
    n_layers = len(data["sorted_layers"])
    pid = data["persona_idx_from_name"]
    ptxt = (data["persona_prompt_from_name"] or "").strip().replace("\n"," ")
    if len(ptxt)>140: ptxt = ptxt[:140]+"…"

    panels.append(html.Div([
        html.H2(name, style={"textAlign":"center","marginBottom":"0.4rem"}),
        html.Div([html.Span(f"Persona {pid if pid is not None else '–'}"),
                  html.Span(" • "),
                  html.Span(ptxt or "(no persona prompt found)")],
                 style={"textAlign":"center","margin":"0.4em"}),
        dcc.Graph(id=f"graph-{name}", config={"displayModeBar": False}),
        html.Div([
            html.Label("Layer:"),
            dcc.Slider(
                id=f"slider-{name}",
                min=1, max=n_layers, step=1, value=1,
                marks={i+1:str(i+1) for i in range(n_layers)},
                tooltip={"placement":"bottom"}
            )
        ], style={"marginTop":"0.8em","padding":"0 1em"}),
        # Hovercard below: image + full explanation (updated on hover)
        html.Div(id=f"hovercard-{name}", style={
            "marginTop":"0.6rem","border":"1px solid #eee","borderRadius":"8px",
            "padding":"0.6rem","background":"white","position":"relative",
            "zIndex":10,"width":"100%","boxShadow":"0 2px 10px rgba(0,0,0,0.06)"
        })
    ], style={"border":"1px solid #ddd","borderRadius":"10px","padding":"0.8em","background":"#fafafa","overflow":"visible"}))

app.layout = html.Div([
    html.H1("GDV Dashboard", style={"textAlign":"center"}),
    html.Div(
        dcc.Checklist(
            id="axis-mode",
            options=[{"label":" Use static per-model axes","value":"static"}],
            value=[],
            labelStyle={"display":"inline-block","margin":"0 1em"}
        ),
        style={"textAlign":"center","marginBottom":"20px"}
    ),
    html.Div(panels, id="grid",
             style={"display":"grid","gridTemplateColumns":"repeat(auto-fit, minmax(600px, 1fr))","gap":"1rem","padding":"1rem"})
])

# ─────────────────────────────────────────────────────────────
# Callbacks: figure + hovercard (image below)
# ─────────────────────────────────────────────────────────────
for name, data in all_data.items():
    pid = data["persona_idx_from_name"]
    fallback = data.get("fallback_filenames_by_order", GLOBAL_IMAGE_ORDER)

    @app.callback(
        Output(f"graph-{name}", "figure"),
        Output(f"hovercard-{name}", "children"),
        Input(f"slider-{name}", "value"),
        Input("axis-mode", "value"),
        Input(f"graph-{name}", "hoverData"),
        prevent_initial_call=False
    )
    def update_panel(layer_1based, axis_mode, hoverData, name=name, data=data, pid=pid, fallback=fallback):
        idx0 = int(layer_1based)-1
        Lkey = data["sorted_layers"][idx0]
        ld = data["layer_data"][Lkey]

        x = np.array(ld["x"], float)
        y = np.array(ld["y"], float)
        grp = np.array(ld.get("group", np.zeros(len(x), int)))
        sents = np.array(ld.get("sentence", [""]*len(x)), dtype=object)

        # Build per-point label + served URL, + short text for purple hover
        expl_full, expl_short, fname_list, url_list = [], [], [], []
        for i in range(len(x)):
            full = str(sents[i]) if i < len(sents) else ""
            short = textwrap.shorten(full, width=120, placeholder="…")  # keep purple hover compact
            label, raw, _ = pick_image_label(ld, i, fallback)
            fname, exists = resolve_served_filename(raw if raw else label)
            url = f"/_images/{fname}" if fname else ""
            expl_full.append(full); expl_short.append(short)
            fname_list.append(fname or label); url_list.append(url)

        custom_all = np.column_stack([
            np.array(expl_short, dtype=object),   # [0] short text for purple hover
            np.array(fname_list, dtype=object),   # [1] filename label
            np.array(url_list,   dtype=object),   # [2] served image URL
            np.full(len(x), pid if pid is not None else "", dtype=object),  # [3] persona idx
            np.array(expl_full,  dtype=object),   # [4] FULL explanation  ← add this
        ])


        colors = ["blue","red","green","orange","purple","brown","pink","gray"]
        fig = go.Figure()
        for i, g in enumerate(np.unique(grp)):
            mask = (grp == g)
            fig.add_trace(go.Scatter(
                x=x[mask], y=y[mask], mode="markers",
                marker=dict(size=9, color=colors[i % len(colors)]),
                name=f"Group {g}",
                customdata=custom_all[mask],
                hovertemplate="<b>Image:</b> %{customdata[1]}<br>%{customdata[0]}<br><b>Persona:</b> %{customdata[3]}<extra></extra>",
                hoverinfo="all"
            ))

        gdv_here = float(data["gdv_per_layer"][Lkey])
        fig.update_layout(
            title=f"{name} — Layer {layer_1based}: GDV = {gdv_here:.4f}",
            xaxis_title="PC 1", yaxis_title="PC 2", legend_title="Group",
            margin=dict(l=40, r=20, t=60, b=80), height=500,
            hoverlabel=dict(bgcolor="rgba(255,255,255,0.98)", align="left", font_size=13, namelength=-1)
        )
        if "static" in axis_mode:
            fig.update_xaxes(range=[data["static_min"], data["static_max"]])
            fig.update_yaxes(range=[data["static_min"], data["static_max"]])

        # Hover card with actual image + full explanation
        if hoverData and "points" in hoverData and hoverData["points"]:
            p = hoverData["points"][0]
            fname = p["customdata"][1]; url = p["customdata"][2]
            full_text = p["customdata"][4] if p.get("customdata") else ""
            persona_prompt = PERSONA_PROMPTS.get(pid, "")
            children = [
                html.Img(src=url, style={"maxWidth":"100%","borderRadius":"6px"}) if url else html.Div(),
                html.Div([html.B(f"Image: {fname}")], style={"marginTop":"0.4rem"}),
                html.Div(full_text, style={"marginTop":"0.3rem"}),
                html.Div([
                    html.Div(html.B(f"Persona {pid} prompt")),
                    html.Div(persona_prompt or "(no prompt found)", style={"whiteSpace":"pre-wrap"})
                ], style={"marginTop":"0.6rem","fontSize":"0.95em","background":"#fafafa","padding":"0.5rem","borderRadius":"6px"})
            ]
        else:
            persona_prompt = PERSONA_PROMPTS.get(pid, "")
            children = [
                html.Div("(hover a point to preview its image + explanation)"),
                html.Div(html.B(f"Persona {pid} prompt")),
                html.Div(persona_prompt or "(no prompt found)", style={"whiteSpace":"pre-wrap","marginTop":"0.3rem"})
            ]

        return fig, children

# ─────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(IMAGES_ROOT, exist_ok=True)
    print("Serving images from:", IMAGES_ROOT)
    app.run_server(debug=True)

