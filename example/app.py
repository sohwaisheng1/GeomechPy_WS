"""
Quick MEM Calculator - Streamlit front end.

A one-page Mechanical Earth Model (MEM) builder on top of the geomechpy
library: dynamic elastic properties -> static conversion -> rock strength
(selectable methods) -> overburden & pore pressure -> horizontal stresses ->
vertical-well wellbore stability, with QC flagging, tornado sensitivity
analysis and interactive Plotly displays. Supports Oilfield and Metric
unit systems for both input and display.

Run locally:   streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils import mem_calculator as mc

# ---------------------------------------------------------------------------
# Page setup & session state
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Quick MEM Calculator",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Persist data across reruns
for key, default in {
    "raw_df": None,        # uploaded / sample input data
    "results_df": None,    # full workflow output (canonical units)
    "qc_summary": None,
    "qc_flags": None,      # per-sample QC flags (canonical column names)
    "data_source": None,   # label shown in the sidebar
    "load_messages": [],   # info messages from the data cleaner
    "unit_warnings": [],   # unit sanity warnings from the last run
    "tornado": None,       # last tornado analysis (fig, table, meta)
}.items():
    st.session_state.setdefault(key, default)

FLAG_COLORS = {
    "OK": "background-color: #1e7e34; color: white",
    "LOW": "background-color: #d39e00; color: black",
    "HIGH": "background-color: #c82333; color: white",
    "MISSING": "background-color: #6c757d; color: white",
}


def style_flags(table: pd.DataFrame, flags: pd.DataFrame, columns: list[str]) -> "pd.io.formats.style.Styler":
    """Color-code table cells according to their QC flag (matching column names)."""
    shown = [c for c in columns if c in table.columns]

    def _color(row_df: pd.DataFrame) -> pd.DataFrame:
        css = pd.DataFrame("", index=row_df.index, columns=row_df.columns)
        for col in row_df.columns:
            if col in flags.columns:
                css[col] = flags.loc[row_df.index, col].map(FLAG_COLORS).fillna("")
        return css

    return table[shown].style.apply(_color, axis=None).format(precision=3)


def depth_track_figure(df: pd.DataFrame, depth_col: str, tracks: list[tuple[str, list[tuple[str, str]]]], height: int = 750) -> go.Figure:
    """Build a multi-track log plot (property vs depth, depth increasing downwards).

    tracks: list of (track_title, [(column, legend_name), ...])
    """
    fig = make_subplots(
        rows=1,
        cols=len(tracks),
        shared_yaxes=True,
        horizontal_spacing=0.03,
        subplot_titles=[t[0] for t in tracks],
    )
    for i, (_, curves) in enumerate(tracks, start=1):
        for col, name in curves:
            if col not in df.columns:
                continue
            fig.add_trace(
                go.Scatter(
                    x=df[col],
                    y=df[depth_col],
                    mode="lines",
                    name=name,
                    hovertemplate=f"{name}: %{{x:.3f}}<br>Depth: %{{y:.1f}}<extra></extra>",
                ),
                row=1,
                col=i,
            )
    fig.update_yaxes(autorange="reversed", title_text=depth_col, col=1)
    fig.update_layout(
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.06),
        margin=dict(t=90, b=40),
    )
    return fig


def mud_window_figure(disp: pd.DataFrame, depth_col: str, N: dict[str, str], unit: str, height: int = 700) -> go.Figure:
    """NEW: mud weight window plot — safe window shaded between the breakout
    (shear failure) and breakdown (fracture) mud weight limits."""
    fig = go.Figure()
    bo, bd = N.get("MW_BREAKOUT_GCC"), N.get("MW_BREAKDOWN_GCC")
    if bo in disp.columns and bd in disp.columns:
        fig.add_trace(
            go.Scatter(
                x=disp[bo], y=disp[depth_col], mode="lines", name="Breakout limit (min MW)",
                line=dict(color="#c82333"),
                hovertemplate=f"Breakout: %{{x:.3f}} {unit}<br>Depth: %{{y:.1f}}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=disp[bd], y=disp[depth_col], mode="lines", name="Breakdown limit (max MW)",
                line=dict(color="#1f77b4"), fill="tonextx", fillcolor="rgba(40, 167, 69, 0.18)",
                hovertemplate=f"Breakdown: %{{x:.3f}} {unit}<br>Depth: %{{y:.1f}}<extra></extra>",
            )
        )
    extra_curves = [
        ("MW_PP_GCC", "Pore pressure EMW", "dot", "gray"),
        ("MW_SV_GCC", "Overburden EMW", "dash", "gray"),
        # NEW: loss gradient = min principal stress (Sv, SHmax, Shmin) as EMW
        ("MW_LOSS_GCC", "Loss gradient (min principal stress)", "dashdot", "#6f42c1"),
    ]
    for canonical, label, dash, color in extra_curves:
        col = N.get(canonical)
        if col in disp.columns:
            fig.add_trace(
                go.Scatter(
                    x=disp[col], y=disp[depth_col], mode="lines", name=label,
                    line=dict(dash=dash, color=color),
                    hovertemplate=f"{label}: %{{x:.3f}} {unit}<br>Depth: %{{y:.1f}}<extra></extra>",
                )
            )
    fig.update_yaxes(autorange="reversed", title_text=depth_col)
    fig.update_layout(
        height=height,
        xaxis_title=f"Equivalent mud weight ({unit})",
        legend=dict(orientation="h", yanchor="bottom", y=1.04),
        margin=dict(t=80, b=40),
        title="Mud weight window (green = safe drilling window)",
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar: units, data input & calculation settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🪨 Quick MEM Calculator")
    st.caption(
        "1D Mechanical Earth Model builder powered by "
        "[geomechpy](https://github.com/sohwaisheng1/GeomechPy_WS)."
    )
    st.divider()

    st.header("1. Units")
    unit_system = st.selectbox(
        "Input/Output Units",
        mc.UNIT_SYSTEMS,
        index=0,
        help="Controls how the uploaded data is interpreted AND how results are displayed.",
    )
    depth_unit = st.radio(
        "Depth (MD) unit",
        mc.DEPTH_UNITS,
        index=0,
        horizontal=True,
        help="Unit of the depth column. MD is assumed ≈ TVD (vertical well) for the stress calculations.",
    )
    expected = dict(mc.INPUT_UNITS[unit_system])
    expected["DEPTH"] = depth_unit
    st.caption(
        "Expected input units — "
        + " · ".join(f"{curve}: {unit}" for curve, unit in expected.items())
    )

    st.divider()
    st.header("2. Data input")
    uploaded = st.file_uploader(
        "Upload well log data (CSV / Excel)",
        type=["csv", "txt", "xls", "xlsx"],
        help="One row per depth sample. Required curves: DEPTH/MD, GR, RHOB, DTCO, DTSM. "
        "POROSITY is optional. Unit rows under the header and -999.25/-9999 nulls are handled automatically.",
    )
    if uploaded is not None:
        try:
            st.session_state.raw_df, st.session_state.load_messages = mc.load_data(uploaded)
            st.session_state.data_source = f"📄 {uploaded.name}"
        except ValueError as exc:
            st.error(str(exc))

    if st.button("🧪 Load Sample Data", use_container_width=True):
        st.session_state.raw_df = mc.generate_sample_data(unit_system=unit_system)
        st.session_state.data_source = f"🧪 Synthetic sample well (2500-3000 m, {unit_system.lower()})"
        st.session_state.load_messages = []

    st.download_button(
        "⬇️ Download Example File",
        data=mc.sample_csv_bytes(unit_system=unit_system),
        file_name="mem_example_data.csv",
        mime="text/csv",
        use_container_width=True,
        help=f"Clean sample CSV (MD, GR, RHOB, DTCO, DTSM, POROSITY) in {unit_system.lower()}.",
    )

    if st.session_state.data_source:
        st.success(f"Loaded: {st.session_state.data_source}")
    for msg in st.session_state.load_messages:
        st.info(msg)
    if st.session_state.raw_df is not None:
        undetected = mc.missing_required_curves(list(st.session_state.raw_df.columns))
        if undetected:
            st.warning(
                "Could not auto-detect column(s) for: "
                + ", ".join(undetected)
                + ". Map them manually below or check your file."
            )

    st.divider()
    st.header("3. Column mapping")
    column_map: dict[str, str] = {}
    if st.session_state.raw_df is not None:
        options = ["-- not mapped --"] + list(st.session_state.raw_df.columns)
        for curve in mc.ALL_CURVES:
            required = curve in mc.REQUIRED_CURVES
            label = f"{curve} [{expected[curve]}] {'(required)' if required else '(optional)'}"
            choice = st.selectbox(
                label,
                options,
                index=mc.guess_column(curve, list(st.session_state.raw_df.columns)),
                key=f"map_{curve}",
            )
            column_map[curve] = "" if choice == "-- not mapped --" else choice
    else:
        st.info("Load data first to map columns.")

    st.divider()
    st.header("4. Static properties")
    method_label = st.selectbox(
        "Dynamic → static YME correlation",
        list(mc.STATIC_YME_METHODS.keys()),
        help="Correlations from geomechpy.static_elastic_properties. "
        "Morales additionally requires a mapped POROSITY column.",
    )
    calibration_multiplier = st.slider(
        "Static YME calibration multiplier",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.05,
        help="Scales the correlation output — use it to calibrate against core test data.",
    )
    pr_multiplier = st.slider(
        "Static Poisson's ratio multiplier",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.05,
    )
    custom_a, custom_b = 0.5, 1.0
    if "power" in method_label:
        custom_a = st.number_input("Custom multiplier a", value=0.5, format="%.4f")
        custom_b = st.number_input("Custom exponent b", value=1.0, format="%.4f")
    elif "linear" in method_label:
        custom_a = st.number_input("Custom slope a", value=0.8, format="%.4f")
        custom_b = st.number_input("Custom intercept b (Mpsi)", value=0.0, format="%.4f")

    st.divider()
    st.header("5. Rock strength")
    # NEW: selectable UCS / FANG methods
    ucs_method_label = st.selectbox(
        "UCS method",
        list(mc.UCS_METHODS.keys()),
        help="All UCS correlations available in geomechpy.rock_strength (currently Plumb 1994), "
        "plus a constant-value fallback for calibration.",
    )
    ucs_method = mc.UCS_METHODS[ucs_method_label]
    ucs_constant_mpa = 50.0
    if ucs_method == "constant":
        ucs_constant_mpa = st.number_input("Constant UCS (MPa)", value=50.0, min_value=0.1, format="%.1f")
    fang_method_label = st.selectbox(
        "Friction angle method",
        list(mc.FANG_METHODS.keys()),
        help="All FANG correlations available in geomechpy.rock_strength (currently Lal 1999), "
        "plus a constant-value fallback for calibration.",
    )
    fang_method = mc.FANG_METHODS[fang_method_label]
    fang_constant_deg = 30.0
    fang_gr_min, fang_gr_max = 15.0, 120.0
    if fang_method == "constant":
        fang_constant_deg = st.number_input("Constant friction angle (deg)", value=30.0, min_value=1.0, max_value=60.0, format="%.1f")
    elif fang_method == "gr_linear":
        cgr1, cgr2 = st.columns(2)
        fang_gr_min = cgr1.number_input("GR min (clean sand, gAPI)", value=15.0, format="%.1f",
                                        help="GR at clean sand → FANG = 45°.")
        fang_gr_max = cgr2.number_input("GR max (pure shale, gAPI)", value=120.0, format="%.1f",
                                        help="GR at pure shale → FANG = 15°. Must differ from GR min.")
    ucs_multiplier = st.slider(
        "UCS calibration multiplier",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.05,
        help="Scales the UCS output (and TSTR derived from it) — calibrate against core test data, "
        "just like the static YME multiplier.",
    )
    tstr_multiplier = st.slider(
        "Tensile strength / UCS ratio",
        min_value=0.05,
        max_value=0.30,
        value=0.15,
        step=0.01,
        help="TSTR = ratio × UCS (geomechpy default is 0.15).",
    )

    st.divider()
    st.header("6. Mechanical stratigraphy")
    # NEW: GR-based lithology flag with user-defined cutoffs
    compute_litho = st.checkbox(
        "Flag lithology from GR",
        value=True,
        help="Classify each sample into sandstone (0), shale (1), limestone (2) or coal (6) "
        "using GR windows. The flag is shown on the stratigraphy tab and alongside the "
        "computation tabs.",
    )
    litho_config = None
    if compute_litho:
        st.caption(
            "Define the GR window [min, max) gAPI for each lithology. First matching row wins "
            "(priority = top to bottom). Samples matching none are 'Undefined'."
        )
        litho_editor = st.data_editor(
            pd.DataFrame(mc.default_litho_config()),
            column_config={
                "name": st.column_config.TextColumn("Lithology", disabled=True),
                "code": st.column_config.NumberColumn("Code", disabled=True),
                "gr_min": st.column_config.NumberColumn("GR min", min_value=0.0, step=1.0, format="%.1f"),
                "gr_max": st.column_config.NumberColumn("GR max", min_value=0.0, step=1.0, format="%.1f"),
            },
            hide_index=True,
            num_rows="fixed",
            use_container_width=True,
            key="litho_editor",
        )
        litho_config = litho_editor.to_dict("records")

    st.divider()
    st.header("7. Stress & wellbore stability")
    # NEW: overburden / pore pressure / horizontal stress / stability inputs
    compute_stress = st.checkbox(
        "Compute stresses & mud weight window",
        value=True,
        help="Adds overburden, pore pressure, horizontal stresses and the vertical-well "
        "mud weight window (geomechpy gradient-based + poroelastic methods).",
    )
    stress_params = None
    if compute_stress:
        setting = st.radio("Well setting", mc.WELL_SETTINGS, horizontal=True)
        air_gap = st.number_input(
            f"Air gap / KB elevation ({depth_unit})", value=0.0, min_value=0.0, format="%.1f",
            help="Drill floor to ground level (onshore) or to mean sea level (offshore).",
        )
        water_depth, sea_gradient = 0.0, 0.47
        if setting == "Offshore":
            water_depth = st.number_input(f"Water depth ({depth_unit})", value=0.0, min_value=0.0, format="%.1f")
            sea_gradient = st.number_input("Sea water gradient (psi/ft)", value=0.47, min_value=0.30, max_value=0.60, format="%.3f")
        ovb_source_label = st.selectbox(
            "Overburden gradient source",
            list(mc.OVB_GRADIENT_SOURCES.keys()),
            help="Constant lithostatic gradient, or a gradient derived from the mean of the "
            "mapped RHOB log (mean g/cc × 0.4335 psi/ft).",
        )
        ovb_source = mc.OVB_GRADIENT_SOURCES[ovb_source_label]
        ovb_gradient = st.number_input(
            "Lithostatic gradient (psi/ft)", value=1.05, min_value=0.5, max_value=1.5, format="%.3f",
            disabled=(ovb_source == "density"),
            help="Typical 1.0–1.1 psi/ft. Ignored when the gradient is derived from RHOB.",
        )
        pp_gradient = st.number_input(
            "Pore pressure gradient (psi/ft)", value=0.47, min_value=0.30, max_value=1.0, format="%.3f",
            help="Hydrostatic ≈ 0.433–0.47 psi/ft; higher = overpressure.",
        )
        shmax_method_label = st.selectbox("SHmax method", list(mc.SHMAX_METHODS.keys()))
        shmax_method = mc.SHMAX_METHODS[shmax_method_label]
        shmax_multiplier = 1.1
        if shmax_method == "multiplier":
            shmax_multiplier = st.slider("SHmax / Shmin multiplier", 1.0, 2.0, 1.1, 0.05)
        biot = st.slider("Biot coefficient", 0.5, 1.0, 1.0, 0.05)
        c_ex, c_ey = st.columns(2)
        ex = c_ex.number_input("Tectonic strain EX", value=0.0001, format="%.5f",
                               help="Poroelastic tectonic strain term (Shmin direction).")
        ey = c_ey.number_input("Tectonic strain EY", value=0.009, format="%.5f",
                               help="Poroelastic tectonic strain term (SHmax direction). Keep EY ≥ EX.")
        stress_params = {
            "setting": setting,
            "depth_unit": depth_unit,
            "air_gap": air_gap,
            "water_depth": water_depth,
            "sea_gradient_psift": sea_gradient,
            "ovb_source": ovb_source,
            "ovb_gradient_psift": ovb_gradient,
            "pp_gradient_psift": pp_gradient,
            "shmax_method": shmax_method,
            "shmax_multiplier": shmax_multiplier,
            "biot": biot,
            "ex": ex,
            "ey": ey,
        }

    st.divider()
    run_clicked = st.button(
        "🚀 Run MEM Calculation",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.raw_df is None,
    )

# Bundle the workflow settings once — used by the run button and the tornado tab.
workflow_settings = dict(
    method_label=method_label,
    calibration_multiplier=calibration_multiplier,
    pr_multiplier=pr_multiplier,
    tstr_multiplier=tstr_multiplier,
    custom_a=custom_a,
    custom_b=custom_b,
    unit_system=unit_system,
    ucs_method=ucs_method,
    fang_method=fang_method,
    ucs_constant_mpa=ucs_constant_mpa,
    fang_constant_deg=fang_constant_deg,
    fang_gr_min=fang_gr_min,
    fang_gr_max=fang_gr_max,
    ucs_multiplier=ucs_multiplier,
    litho_config=litho_config,
    stress_params=stress_params,
)

# ---------------------------------------------------------------------------
# Run the workflow
# ---------------------------------------------------------------------------

if run_clicked:
    try:
        st.session_state.unit_warnings = mc.check_unit_sanity(
            st.session_state.raw_df, column_map, unit_system
        )
        with st.spinner("Computing properties, stresses and stability..."):
            results = mc.run_full_workflow(
                data=st.session_state.raw_df,
                column_map=column_map,
                **workflow_settings,
            )
        st.session_state.results_df = results
        st.session_state.qc_summary, st.session_state.qc_flags = mc.run_qc(results)
        st.toast("MEM calculation complete ✅")
    except ValueError as exc:
        st.error(f"⚠️ {exc}")
    except Exception as exc:  # keep the app alive on unexpected input
        st.error(f"Unexpected error during calculation: {exc}")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Quick MEM Calculator")
st.markdown(
    "Build a quick-look **Mechanical Earth Model** from standard well logs: "
    "dynamic → static properties → rock strength → stresses → mud weight window, with built-in QC."
)

for warning in st.session_state.unit_warnings:
    st.warning(f"⚠️ Unit check: {warning}")

results = st.session_state.results_df
flags = st.session_state.qc_flags

# Convert canonical results into the selected display unit system.
# N maps canonical column names -> display names (e.g. YME_DYN_GPA -> 'YME_DYN [Mpsi]').
# has_stress requires the FULL current stress column set (incl. the loss gradient)
# so results from an older run/app version prompt a re-run instead of KeyError.
STRESS_COLUMNS = ["SV_MPA", "PP_MPA", "SHMIN_MPA", "SHMAX_MPA", "LOSS_P_MPA",
                  "MW_BREAKOUT_GCC", "MW_BREAKDOWN_GCC", "MW_LOSS_GCC"]
STRESS_INFO = (
    "Enable **Compute stresses & mud weight window** in the sidebar and click "
    "**🚀 Run MEM Calculation** — the results currently in memory don't include "
    "the full stress profile (they may be from an older run or app version)."
)
if results is not None:
    disp, N = mc.display_results(results, unit_system)
    flags_disp = flags.rename(columns=N) if flags is not None else None
    DEPTH = N["DEPTH"]
    has_stress = all(c in results.columns for c in STRESS_COLUMNS)
    has_litho = "LITHO_CODE" in results.columns and results["LITHO_CODE"].notna().any()
else:
    disp, N, flags_disp, DEPTH, has_stress, has_litho = None, {}, None, None, False, False


def lithology_figure(depth_series, code_series, height: int = 650, title: str = "Lithology (from GR)") -> go.Figure:
    """Colored lithology strip vs depth (contiguous runs shaded by code)."""
    depth = pd.to_numeric(depth_series, errors="coerce").to_numpy(dtype=float)
    codes = pd.to_numeric(code_series, errors="coerce").to_numpy(dtype=float)
    filled = np.where(np.isfinite(codes), codes, -1.0)
    n = len(filled)
    fig = go.Figure()
    present = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and filled[j + 1] == filled[i]:
            j += 1
        code = int(filled[i])
        y0 = depth[i]
        y1 = depth[j + 1] if (j + 1) < n else depth[j]
        fig.add_shape(type="rect", xref="x", yref="y", x0=0.0, x1=1.0, y0=y0, y1=y1,
                      fillcolor=mc.LITHO_COLORS.get(code, "#ecf0f1"), line_width=0, layer="below")
        present.append(code)
        i = j + 1
    for code in sorted(set(present)):
        label = mc.LITHO_NAME_BY_CODE.get(code, "Undefined")
        name = f"{label} ({code})" if code >= 0 else "Undefined"
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=12, color=mc.LITHO_COLORS.get(code, "#ecf0f1")),
                                 name=name))
    fig.update_xaxes(visible=False, range=[0.0, 1.0])
    fig.update_yaxes(autorange="reversed", title_text=DEPTH if DEPTH else "Depth")
    fig.update_layout(height=height, title=title,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      margin=dict(t=70, b=40))
    return fig


def litho_expander(key: str, label: str = "🪨 Lithology flag (from GR)") -> None:
    """Show the lithology strip in a collapsed expander (for computation tabs)."""
    if has_litho:
        with st.expander(label, expanded=False):
            st.plotly_chart(
                lithology_figure(disp[DEPTH], disp[N["LITHO_CODE"]], height=430),
                use_container_width=True,
                key=f"litho_{key}",
            )


st.title("Quick MEM Calculator")
st.markdown(
    "Build a quick-look **Mechanical Earth Model** from standard well logs: "
    "mechanical stratigraphy → stresses → rock properties → wellbore stability, with built-in QC."
)

for warning in st.session_state.unit_warnings:
    st.warning(f"⚠️ Unit check: {warning}")

(
    tab_input,
    tab_strat,
    tab_ovb,
    tab_rock,
    tab_hstress,
    tab_wbs,
    tab_qc,
    tab_tornado,
) = st.tabs(
    [
        "📥 Data Input",
        "🪨 Mechanical Stratigraphy",
        "🏔️ Overburden & Pore Pressure",
        "🧱 Rock Properties",
        "↔️ Horizontal Stress",
        "🛢️ Wellbore Stability",
        "✅ QC & Results",
        "🌪️ Sensitivity (Tornado)",
    ]
)

# --- Tab 1: Data input ------------------------------------------------------
with tab_input:
    st.subheader("Data input")
    # Clear expected-columns reference as a table, in the selected unit system.
    curve_meta = [
        ("DEPTH", "MD", "Measured depth (assumed ≈ TVD, vertical well)", "Required", "2500.0"),
        ("GR", "Gamma ray", "Shale/lithology indicator; drives mechanical stratigraphy", "Required", "75.0"),
        ("RHOB", "Bulk density", "Formation bulk density (elastic properties, Sv option)", "Required", "2.45"),
        ("DTCO", "Compressional slowness", "P-sonic transit time (Vp, moduli, McNally UCS, Lal FANG)", "Required", "85.0"),
        ("DTSM", "Shear slowness", "S-sonic transit time (Vs, moduli)", "Required", "150.0"),
        ("POROSITY", "Porosity", "Total/effective porosity (only for the Morales static method)", "Optional", "0.18"),
    ]
    ref = pd.DataFrame(
        [
            {
                "Curve": c,
                "Description": f"{name} — {desc}",
                f"Unit ({unit_system})": expected[c],
                "Requirement": req,
                "Example value": ex,
            }
            for c, name, desc, req, ex in curve_meta
        ]
    )
    st.markdown(
        f"**Expected input columns** — one row per depth sample. Column *names* can be anything; "
        f"map them to these curves in the sidebar. Units below follow the selected **{unit_system}** system."
    )
    st.dataframe(ref, use_container_width=True, hide_index=True)
    st.caption(
        "Tip: use **⬇️ Download Example File** in the sidebar for a correctly formatted CSV template, "
        "or **🧪 Load Sample Data** to try the app immediately. Unit rows under the header and "
        "-999.25 / -9999 null flags are handled automatically on upload."
    )

    if st.session_state.raw_df is None:
        st.info("👈 Upload a CSV/Excel file or click **Load Sample Data** in the sidebar to get started.")
    else:
        df_in = st.session_state.raw_df
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df_in):,}")
        c2.metric("Columns", df_in.shape[1])
        depth_col = column_map.get("DEPTH") if column_map else None
        if depth_col:
            c3.metric("Depth range", f"{df_in[depth_col].min():.0f} – {df_in[depth_col].max():.0f} {depth_unit}")
        st.subheader("Uploaded data preview")
        st.dataframe(df_in, use_container_width=True, height=380)
        with st.expander("Basic statistics"):
            st.dataframe(df_in.describe().T, use_container_width=True)

# --- Tab 2: Mechanical Stratigraphy -----------------------------------------
with tab_strat:
    st.subheader("Mechanical Stratigraphy")
    st.markdown(
        "Each depth is flagged into a lithology from **GR** using the cutoff windows defined in the "
        "sidebar (**6. Mechanical stratigraphy**): sandstone (code 0), shale (1), limestone (2), coal (6). "
        "The flag is carried through and shown alongside the computation tabs."
    )
    if results is None:
        st.info("Run the calculation from the sidebar to generate the lithology flag.")
    elif not has_litho:
        st.info("Enable **Flag lithology from GR** in the sidebar (section 6) and re-run.")
    else:
        counts = mc.lithology_counts(results)
        cols = st.columns(len(mc.LITHO_NAME_BY_CODE))
        for col, (code, name) in zip(cols, mc.LITHO_NAME_BY_CODE.items()):
            row = counts[counts["Code"] == code]
            pct = float(row["Fraction %"].iloc[0]) if not row.empty else 0.0
            col.metric(f"{name} ({code})", f"{pct:.1f}%")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Lithology fractions**")
            st.dataframe(counts, use_container_width=True, hide_index=True)
        with c2:
            st.plotly_chart(
                depth_track_figure(
                    disp, DEPTH,
                    [("GR (gAPI)", [(N["GR"], "GR")])],
                    height=620,
                ),
                use_container_width=True,
            )
        st.plotly_chart(
            lithology_figure(disp[DEPTH], disp[N["LITHO_CODE"]], height=620),
            use_container_width=True,
            key="litho_strat",
        )

# --- Tab 3: Overburden & Pore Pressure --------------------------------------
with tab_ovb:
    if results is None:
        st.info("Run the calculation from the sidebar to see the overburden and pore pressure profiles.")
    elif not has_stress:
        st.info(STRESS_INFO)
    else:
        st.subheader(f"Overburden stress & pore pressure ({unit_system})")
        sp = stress_params or {}
        st.caption(
            f"Setting: **{sp.get('setting', '-')}** · air gap {sp.get('air_gap', 0):g} {depth_unit}"
            + (f" · water depth {sp.get('water_depth', 0):g} {depth_unit}" if sp.get("setting") == "Offshore" else "")
            + f" · Sv gradient source: {sp.get('ovb_source', '-')}"
            + f" · Pp gradient {sp.get('pp_gradient_psift', 0):.3f} psi/ft. MD assumed ≈ TVD (vertical well)."
        )
        litho_expander("ovb")
        ovb_cols = [DEPTH] + [N[c] for c in ["SV_MPA", "PP_MPA", "MW_SV_GCC", "MW_PP_GCC"]]
        st.dataframe(style_flags(disp, flags_disp, ovb_cols), use_container_width=True, height=380)
        mw_unit = mc.display_unit("MW_PP_GCC", unit_system)
        st.plotly_chart(
            depth_track_figure(
                disp, DEPTH,
                [
                    (f"Pressure ({mc.display_unit('SV_MPA', unit_system)})", [(N["SV_MPA"], "Sv"), (N["PP_MPA"], "Pp")]),
                    (f"Equivalent gradients ({mw_unit})", [(N["MW_SV_GCC"], "Sv EMW"), (N["MW_PP_GCC"], "Pp EMW")]),
                ],
                height=650,
            ),
            use_container_width=True,
        )

# --- Tab 4: Rock Properties (dynamic + static + strength) -------------------
with tab_rock:
    if results is None:
        st.info("Run the calculation from the sidebar to see rock properties.")
    else:
        st.subheader(f"Rock properties ({unit_system})")
        st.caption(
            f"Static YME: **{method_label}** ×{calibration_multiplier:.2f} · PR ×{pr_multiplier:.2f} · "
            f"UCS: **{ucs_method_label}** ×{ucs_multiplier:.2f} · FANG: **{fang_method_label}** · "
            f"TSTR = {tstr_multiplier:.2f} × UCS."
        )
        litho_expander("rock")

        st.markdown("**Dynamic elastic properties** (from DTCO / DTSM / RHOB)")
        dyn_cols = [DEPTH] + [N[c] for c in ["VP_MS", "VS_MS", "VPVS", "YME_DYN_GPA", "PR_DYN", "K_DYN_GPA", "G_DYN_GPA", "LAME_DYN_GPA", "M_DYN_GPA"]]
        st.dataframe(style_flags(disp, flags_disp, dyn_cols), use_container_width=True, height=300)

        st.markdown("**Static elastic + rock strength**")
        sta_cols = [DEPTH] + [N[c] for c in ["YME_DYN_GPA", "YME_STA_GPA", "PR_DYN", "PR_STA", "UCS_MPA", "TSTR_MPA", "FANG_DEG"]]
        st.dataframe(style_flags(disp, flags_disp, sta_cols), use_container_width=True, height=300)

        st.plotly_chart(
            depth_track_figure(
                disp, DEPTH,
                [
                    (f"Velocities ({mc.display_unit('VP_MS', unit_system)})", [(N["VP_MS"], "Vp"), (N["VS_MS"], "Vs")]),
                    (f"Young's mod. ({mc.display_unit('YME_DYN_GPA', unit_system)})", [(N["YME_DYN_GPA"], "E dyn"), (N["YME_STA_GPA"], "E sta")]),
                    ("Poisson's ratio", [(N["PR_DYN"], "ν dyn"), (N["PR_STA"], "ν sta")]),
                    (f"UCS / TSTR ({mc.display_unit('UCS_MPA', unit_system)})", [(N["UCS_MPA"], "UCS"), (N["TSTR_MPA"], "TSTR")]),
                    ("Friction angle (°)", [(N["FANG_DEG"], "FANG")]),
                ],
                height=680,
            ),
            use_container_width=True,
        )

# --- Tab 5: Horizontal Stress -----------------------------------------------
with tab_hstress:
    if results is None:
        st.info("Run the calculation from the sidebar to see horizontal stresses.")
    elif not has_stress:
        st.info(STRESS_INFO)
    else:
        st.subheader(f"Horizontal stresses ({unit_system})")
        sp = stress_params or {}
        method_txt = [k for k, v in mc.SHMAX_METHODS.items() if v == sp.get("shmax_method")]
        st.caption(
            f"Method: **{method_txt[0] if method_txt else '-'}** · Biot {sp.get('biot', 1.0):.2f} · "
            f"EX {sp.get('ex', 0):g} · EY {sp.get('ey', 0):g}"
            + (f" · SHmax multiplier ×{sp.get('shmax_multiplier', 1.1):.2f}" if sp.get("shmax_method") == "multiplier" else "")
            + ". Shmin from the poroelastic equation (static PR & YME, Biot, tectonic strains)."
        )
        litho_expander("hstress")
        hs_cols = [DEPTH] + [N[c] for c in ["SV_MPA", "PP_MPA", "SHMIN_MPA", "SHMAX_MPA", "Q_FACTOR", "SH_RATIO"]]
        st.dataframe(style_flags(disp, flags_disp, hs_cols), use_container_width=True, height=380)

        q_med = pd.to_numeric(results["Q_FACTOR"], errors="coerce").median()
        if pd.notna(q_med):
            regime = "Normal" if q_med < 1 else ("Strike-slip" if q_med < 2 else "Reverse")
            st.metric("Median stress regime q-factor", f"{q_med:.2f}", help="q<1 normal · 1–2 strike-slip · 2–3 reverse")
            st.caption(f"Dominant stress regime over the interval: **{regime} faulting**.")

        st.plotly_chart(
            depth_track_figure(
                disp, DEPTH,
                [
                    (
                        f"Stresses ({mc.display_unit('SV_MPA', unit_system)})",
                        [(N["SV_MPA"], "Sv"), (N["SHMAX_MPA"], "SHmax"), (N["SHMIN_MPA"], "Shmin"), (N["PP_MPA"], "Pp")],
                    ),
                    ("q-factor (-)", [(N["Q_FACTOR"], "q")]),
                    ("SHmax/Shmin (-)", [(N["SH_RATIO"], "ratio")]),
                ],
                height=650,
            ),
            use_container_width=True,
        )

# --- Tab 6: Wellbore Stability ----------------------------------------------
with tab_wbs:
    if results is None:
        st.info("Run the calculation from the sidebar to see the wellbore stability results.")
    elif not has_stress:
        st.info(STRESS_INFO)
    else:
        st.subheader(f"Wellbore stability — vertical well ({unit_system})")
        st.caption(
            "Breakout limit: Mohr-Coulomb shear failure (Kirsch, analytical) — drilling below it risks breakouts. "
            "Breakdown limit: fracture initiation (Hubbert & Willis) — drilling above it risks losses. "
            "Loss gradient: minimum principal stress among Sv, SHmax and Shmin. "
            "The green band is the safe mud weight window."
        )
        litho_expander("wbs")

        mw_unit = mc.display_unit("MW_BREAKOUT_GCC", unit_system)
        window = results["MW_BREAKDOWN_GCC"] - results["MW_BREAKOUT_GCC"]
        n_valid = int(window.notna().sum())
        n_closed = int((window < 0).sum())
        c1, c2, c3 = st.columns(3)
        factor = mc.PPG_PER_GCC if unit_system == mc.OILFIELD else 1.0
        c1.metric(f"Median breakout limit ({mw_unit})", f"{results['MW_BREAKOUT_GCC'].median() * factor:.2f}")
        c2.metric(f"Median breakdown limit ({mw_unit})", f"{results['MW_BREAKDOWN_GCC'].median() * factor:.2f}")
        c3.metric(f"Median window width ({mw_unit})", f"{window.median() * factor:.2f}")
        if n_closed:
            st.warning(
                f"⚠️ The mud weight window is closed (breakout limit above breakdown limit) at "
                f"{n_closed} of {n_valid} depth samples — no safe static mud weight exists there."
            )

        wbs_cols = [DEPTH] + [N[c] for c in ["PW_BREAKOUT_MPA", "PW_BREAKDOWN_MPA", "LOSS_P_MPA", "MW_BREAKOUT_GCC", "MW_BREAKDOWN_GCC", "MW_LOSS_GCC", "MW_PP_GCC", "MW_SV_GCC"]]
        st.dataframe(style_flags(disp, flags_disp, wbs_cols), use_container_width=True, height=380)
        st.plotly_chart(mud_window_figure(disp, DEPTH, N, mw_unit), use_container_width=True)

# --- Tab 7: QC & Results ----------------------------------------------------
with tab_qc:
    if results is None or st.session_state.qc_summary is None:
        st.info("Run the calculation from the sidebar to generate the QC report.")
    else:
        qc = st.session_state.qc_summary
        status = mc.qc_status(qc)
        badge = {"PASS": "🟢 PASS", "WARNING": "🟡 WARNING", "FAIL": "🔴 FAIL"}[status]
        st.subheader(f"QC report — overall status: {badge}")
        st.caption(
            "Each curve is checked against standard geomechanical ranges "
            "(units shown in the Unit column). LOW/HIGH = outside range, MISSING = null/non-numeric."
        )

        def _pct_color(v):
            if v >= 95:
                return "background-color: #1e7e34; color: white"
            if v >= 70:
                return "background-color: #d39e00; color: black"
            return "background-color: #c82333; color: white"

        st.dataframe(
            qc.style.map(_pct_color, subset=["% in range"]).format({"% in range": "{:.1f}"}),
            use_container_width=True,
            hide_index=True,
        )

        flagged_canonical = [c for c in flags.columns if (flags[c] != "OK").any()]
        if flagged_canonical:
            st.markdown("**Flagged samples** (rows where at least one curve is out of range or missing):")
            bad_rows = flags[flagged_canonical].ne("OK").any(axis=1)
            show_cols = [DEPTH] + [N[c] for c in flagged_canonical if c in N]
            st.dataframe(
                style_flags(disp.loc[bad_rows], flags_disp.loc[bad_rows], show_cols),
                use_container_width=True,
                height=300,
            )
        else:
            st.success("All samples passed QC — no flags raised. 🎉")

        st.divider()
        st.subheader(f"Composite MEM display ({unit_system})")
        composite_tracks = [
            ("GR (gAPI)", [(N["GR"], "GR")]),
            (f"Slowness ({mc.display_unit('DTCO', unit_system)})", [(N["DTCO"], "DTCO"), (N["DTSM"], "DTSM")]),
            (f"E ({mc.display_unit('YME_DYN_GPA', unit_system)})", [(N["YME_DYN_GPA"], "E dyn"), (N["YME_STA_GPA"], "E sta")]),
            (f"UCS / TSTR ({mc.display_unit('UCS_MPA', unit_system)})", [(N["UCS_MPA"], "UCS"), (N["TSTR_MPA"], "TSTR")]),
        ]
        if has_stress:
            mw_unit = mc.display_unit("MW_PP_GCC", unit_system)
            composite_tracks += [
                (
                    f"Stresses ({mc.display_unit('SV_MPA', unit_system)})",
                    [(N["SV_MPA"], "Sv"), (N["SHMAX_MPA"], "SHmax"), (N["SHMIN_MPA"], "Shmin"), (N["PP_MPA"], "Pp")],
                ),
                (
                    f"Mud window ({mw_unit})",
                    [(N["MW_BREAKOUT_GCC"], "MW min"), (N["MW_BREAKDOWN_GCC"], "MW max"), (N["MW_PP_GCC"], "Pp EMW")],
                ),
            ]
        st.plotly_chart(depth_track_figure(disp, DEPTH, composite_tracks, height=800), use_container_width=True)

        if has_litho:
            st.plotly_chart(
                lithology_figure(disp[DEPTH], disp[N["LITHO_CODE"]], height=500),
                use_container_width=True,
                key="litho_qc",
            )

        with st.expander("Crossplot explorer"):
            numeric_cols = [c for c in disp.columns if pd.api.types.is_numeric_dtype(disp[c])]
            c1, c2, c3 = st.columns(3)
            x_col = c1.selectbox("X axis", numeric_cols, index=numeric_cols.index(N["YME_DYN_GPA"]) if N.get("YME_DYN_GPA") in numeric_cols else 0)
            y_col = c2.selectbox("Y axis", numeric_cols, index=numeric_cols.index(N["YME_STA_GPA"]) if N.get("YME_STA_GPA") in numeric_cols else 0)
            color_col = c3.selectbox("Color by", numeric_cols, index=numeric_cols.index(N["GR"]) if N.get("GR") in numeric_cols else 0)
            xfig = go.Figure(
                go.Scatter(
                    x=disp[x_col], y=disp[y_col], mode="markers",
                    marker=dict(color=disp[color_col], colorscale="Viridis", showscale=True, colorbar_title=color_col),
                    hovertemplate=f"{x_col}: %{{x:.3f}}<br>{y_col}: %{{y:.3f}}<extra></extra>",
                )
            )
            xfig.update_layout(xaxis_title=x_col, yaxis_title=y_col, height=520)
            st.plotly_chart(xfig, use_container_width=True)

        st.divider()
        st.download_button(
            f"⬇️ Download results as CSV ({unit_system})",
            data=mc.results_to_csv_bytes(disp),
            file_name="mem_results.csv",
            mime="text/csv",
            type="primary",
        )

# --- Tab 8: Sensitivity analysis (Tornado plot) -----------------------------
with tab_tornado:
    st.subheader("Sensitivity Analysis (Tornado Plot)")
    st.markdown(
        "Using the loaded data as the *base case*, each input (GR, RHOB, DTCO, DTSM, POROSITY and "
        "the static YME multiplier) is varied one at a time by the selected percentage while everything "
        "else is held fixed, and the workflow is recomputed. Bars show how the depth-averaged target "
        "moves — longer bar = more sensitive. GR has no bar unless the target depends on it "
        "(e.g. GR-linear FANG); POROSITY only matters for the Morales static method."
    )
    if st.session_state.raw_df is None:
        st.info("👈 Load data in the sidebar first — the tornado plot needs a base case.")
    else:
        c1, c2 = st.columns(2)
        tornado_targets = list(mc.TORNADO_TARGETS)
        if stress_params is not None:
            tornado_targets += mc.TORNADO_STRESS_TARGETS
        target_options = [mc.display_name(t, unit_system) for t in tornado_targets]
        target_label_sel = c1.selectbox("Target Output", target_options, index=0, key="tornado_target",
                                        help="Result whose sensitivity is analysed (depth-averaged mean).")
        target_canonical = tornado_targets[target_options.index(target_label_sel)]
        variation_pct = c2.select_slider("Variation Range", options=[5, 10, 20], value=10,
                                         format_func=lambda v: f"±{v}%", key="tornado_pct")

        if st.button("🌪️ Generate Tornado Plot", type="primary", key="tornado_btn"):
            try:
                with st.spinner("Recomputing the workflow for each input variation..."):
                    fig, table, base_disp, skipped = mc.generate_tornado_plot(
                        st.session_state.raw_df, column_map, target_canonical, variation_pct,
                        **workflow_settings,
                    )
                st.session_state.tornado = {
                    "fig": fig, "table": table, "base": base_disp, "skipped": skipped,
                    "target_label": mc.display_name(target_canonical, unit_system),
                    "pct": variation_pct, "units": unit_system, "method": method_label,
                }
            except ValueError as exc:
                st.session_state.tornado = None
                st.error(f"⚠️ {exc}")
            except Exception as exc:  # keep the app alive on unexpected input
                st.session_state.tornado = None
                st.error(f"Unexpected error during sensitivity analysis: {exc}")

        tornado = st.session_state.tornado
        if tornado is not None:
            st.metric(f"Base case — depth-averaged {tornado['target_label']}", f"{tornado['base']:.3f}")
            st.plotly_chart(tornado["fig"], use_container_width=True)
            st.markdown("**Per-parameter results** (sorted by impact):")
            st.dataframe(tornado["table"].style.format(precision=3), use_container_width=True, hide_index=True)
            if tornado["skipped"]:
                st.info("Skipped (not mapped or could not be recomputed): " + ", ".join(tornado["skipped"]))
            st.caption(
                f"Generated with ±{tornado['pct']}% variation · {tornado['units']} · "
                f"static method: {tornado['method']}. Re-generate after changing data or settings."
            )

st.divider()
st.caption(
    "Quick MEM Calculator · built with [Streamlit](https://streamlit.io) + "
    "[geomechpy](https://github.com/sohwaisheng1/GeomechPy_WS) · "
    "correlations: Bradford (1998), Najibi (2015), Fuller, Morales (1993), Plumb (1994), Lal (1999), "
    "Thiercelin & Plumb (1994), Hubbert & Willis (1957), Al-Ajmi & Zimmerman (2006)."
)
