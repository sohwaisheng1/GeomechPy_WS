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


def _add_lithology_column(fig: go.Figure, depth_series, code_series, col: int) -> None:
    """Draw the lithology flag as a colored track (contiguous runs shaded by code)
    into subplot column `col`, plus one legend entry per lithology present."""
    depth = pd.to_numeric(depth_series, errors="coerce").to_numpy(dtype=float)
    codes = pd.to_numeric(code_series, errors="coerce").to_numpy(dtype=float)
    filled = np.where(np.isfinite(codes), codes, -1.0)
    n = len(filled)
    present = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and filled[j + 1] == filled[i]:
            j += 1
        c = int(filled[i])
        y0 = depth[i]
        y1 = depth[j + 1] if (j + 1) < n else depth[j]
        fig.add_shape(type="rect", x0=0.0, x1=1.0, y0=y0, y1=y1,
                      fillcolor=mc.LITHO_COLORS.get(c, "#ecf0f1"), line_width=0, layer="below",
                      row=1, col=col)
        present.append(c)
        i = j + 1
    for c in sorted(set(present)):
        label = mc.LITHO_NAME_BY_CODE.get(c, "Undefined")
        name = f"{label} ({c})" if c >= 0 else "Undefined"
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode="markers",
                       marker=dict(size=11, color=mc.LITHO_COLORS.get(c, "#ecf0f1")),
                       name=name, legendgroup="lithology"),
            row=1, col=col,
        )
    fig.update_xaxes(visible=False, range=[0.0, 1.0], row=1, col=col)


def depth_track_figure(df: pd.DataFrame, depth_col: str, tracks: list[tuple[str, list[tuple[str, str]]]], height: int = 750, litho_codes=None) -> go.Figure:
    """Build a multi-track log plot (property vs depth, depth increasing downwards).

    tracks: list of (track_title, [(column, legend_name), ...])
    litho_codes: optional series (aligned to df) — when given, a colored
    lithology flag track is added as the first (leftmost) column.
    """
    has_litho_track = litho_codes is not None
    n_prop = len(tracks)
    ncols = n_prop + (1 if has_litho_track else 0)
    titles = (["Litho"] if has_litho_track else []) + [t[0] for t in tracks]
    widths = None
    if has_litho_track:
        raw = [0.5] + [2.2] * n_prop  # narrow lithology strip, wider property tracks
        total = sum(raw)
        widths = [w / total for w in raw]
    fig = make_subplots(
        rows=1,
        cols=ncols,
        shared_yaxes=True,
        horizontal_spacing=0.03,
        subplot_titles=titles,
        column_widths=widths,
    )
    offset = 1 if has_litho_track else 0
    if has_litho_track:
        _add_lithology_column(fig, df[depth_col], litho_codes, col=1)
    for i, (_, curves) in enumerate(tracks, start=1 + offset):
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
        # legend well above the subplot titles so the two never overlap
        legend=dict(orientation="h", yanchor="bottom", y=1.14),
        margin=dict(t=120, b=40),
    )
    return fig


def mud_window_figure(disp: pd.DataFrame, depth_col: str, N: dict[str, str], unit: str, height: int = 720,
                      litho_codes=None, mw_line=None, ecd_line=None, casing_depths=None) -> go.Figure:
    """NEW: mud weight window plot.

    Safe window (green) is shaded between the breakout limit (min MW, shear
    failure) and the LOSS GRADIENT (max MW = minimum principal stress among
    Sv/SHmax/Shmin) — exceeding the loss gradient risks losses into
    natural/reopened fractures. Breakdown (fracture initiation), Pp and Sv
    are shown as reference lines. A lithology flag track is prepended when
    litho_codes is provided.
    """
    has_litho_track = litho_codes is not None
    if has_litho_track:
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.03,
                            subplot_titles=["Litho", "Mud weight window"],
                            column_widths=[0.12, 0.88])
        _add_lithology_column(fig, disp[depth_col], litho_codes, col=1)
        mw_col = 2
    else:
        fig = make_subplots(rows=1, cols=1, subplot_titles=["Mud weight window"])
        mw_col = 1

    def _line(canonical, label, color, dash=None, fill=None):
        c = N.get(canonical)
        if c in disp.columns:
            fig.add_trace(
                go.Scatter(
                    x=disp[c], y=disp[depth_col], mode="lines", name=label,
                    line=dict(color=color, dash=dash), fill=fill,
                    fillcolor="rgba(40, 167, 69, 0.18)" if fill else None,
                    hovertemplate=f"{label}: %{{x:.3f}} {unit}<br>Depth: %{{y:.1f}}<extra></extra>",
                ),
                row=1, col=mw_col,
            )

    # safe window: breakout (lower) -> loss gradient (upper, filled green)
    _line("MW_BREAKOUT_GCC", "Breakout limit (min MW)", "#c82333")
    _line("MW_LOSS_GCC", "Loss gradient / min σ (max MW)", "#6f42c1", fill="tonextx")
    # reference lines
    _line("MW_BREAKDOWN_GCC", "Breakdown limit (fracture)", "#1f77b4", dash="dashdot")
    _line("MW_PP_GCC", "Pore pressure EMW", "gray", dash="dot")
    _line("MW_SV_GCC", "Overburden EMW", "gray", dash="dash")

    # (5) user-planned MW and ECD as vertical lines (move via the tab sliders)
    depth_vals = pd.to_numeric(disp[depth_col], errors="coerce")
    y_top, y_bot = float(depth_vals.min()), float(depth_vals.max())
    if mw_line is not None:
        fig.add_trace(
            go.Scatter(x=[mw_line, mw_line], y=[y_top, y_bot], mode="lines",
                       name=f"MW = {mw_line:.2f} {unit}", line=dict(color="#0b6e4f", width=3)),
            row=1, col=mw_col,
        )
    if ecd_line is not None:
        fig.add_trace(
            go.Scatter(x=[ecd_line, ecd_line], y=[y_top, y_bot], mode="lines",
                       name=f"ECD = {ecd_line:.2f} {unit}", line=dict(color="#e67e22", width=3, dash="dash")),
            row=1, col=mw_col,
        )
    # (5) casing setting depths as horizontal lines
    for k, cd in enumerate(casing_depths or []):
        fig.add_hline(y=cd, line=dict(color="#111", width=1.5, dash="dot"), row=1, col=mw_col)
        fig.add_annotation(x=1.0, xref="x domain", y=cd, yref="y", showarrow=False,
                           text=f"Casing {k + 1}: {cd:g}", font=dict(size=10, color="#111"),
                           bgcolor="rgba(255,255,255,0.6)", xanchor="right", yanchor="bottom",
                           row=1, col=mw_col)

    fig.update_yaxes(autorange="reversed", title_text=depth_col, col=1)
    fig.update_xaxes(title_text=f"Equivalent mud weight ({unit})", row=1, col=mw_col)
    fig.update_layout(
        height=height,
        # title at the very top, legend BELOW the plot -> no overlap
        title=dict(text="Mud weight window (green = safe window: breakout → loss gradient)",
                   y=0.98, yanchor="top"),
        legend=dict(orientation="h", yanchor="top", y=-0.12),
        margin=dict(t=60, b=90),
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

    # (4) Sidebar sections 1-7 are collapsible via st.expander.
    with st.expander("1. Units", expanded=True):
        unit_system = st.selectbox(
            "Input/Output Units",
            mc.UNIT_SYSTEMS,
            index=0,
            help="Controls how the uploaded data is interpreted AND how results are displayed. "
            "Note: YME, UCS, TSTR, Sv, Pp, Shmin and SHmax are always shown in psi.",
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

    with st.expander("2. Data input", expanded=True):
        uploaded = st.file_uploader(
            "Upload well log data (LAS / CSV / Excel)",
            type=["las", "csv", "txt", "xls", "xlsx"],
            help="One row per depth sample. LAS files are read with lasio and their curves "
            "auto-extracted. Required curves: DEPTH/MD, GR, RHOB, DTCO, DTSM. POROSITY is "
            "optional. Unit rows and -999.25/-9999 nulls are handled automatically.",
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

    with st.expander("3. Column mapping", expanded=True):
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

    with st.expander("5. Static properties", expanded=False):
        method_label = st.selectbox(
            "Dynamic → static YME correlation",
            list(mc.STATIC_YME_METHODS.keys()),
            help="Correlations from geomechpy.static_elastic_properties. "
            "Morales additionally requires a mapped POROSITY column.",
        )
        calibration_multiplier = st.slider(
            "Static YME calibration multiplier",
            min_value=0.5, max_value=2.0, value=1.0, step=0.05,
            help="Scales the correlation output — use it to calibrate against core test data.",
        )
        pr_multiplier = st.slider(
            "Static Poisson's ratio multiplier",
            min_value=0.5, max_value=2.0, value=1.0, step=0.05,
        )
        custom_a, custom_b = 0.5, 1.0
        if "power" in method_label:
            custom_a = st.number_input("Custom multiplier a", value=0.5, format="%.4f")
            custom_b = st.number_input("Custom exponent b", value=1.0, format="%.4f")
        elif "linear" in method_label:
            custom_a = st.number_input("Custom slope a", value=0.8, format="%.4f")
            custom_b = st.number_input("Custom intercept b (Mpsi)", value=0.0, format="%.4f")

    with st.expander("6. Rock strength", expanded=False):
        ucs_method_label = st.selectbox(
            "UCS method",
            list(mc.UCS_METHODS.keys()),
            help="All UCS correlations available in geomechpy.rock_strength, plus a constant-value fallback.",
        )
        ucs_method = mc.UCS_METHODS[ucs_method_label]
        ucs_constant_mpa = 50.0
        if ucs_method == "constant":
            ucs_constant_mpa = st.number_input("Constant UCS (MPa)", value=50.0, min_value=0.1, format="%.1f")
        fang_method_label = st.selectbox(
            "Friction angle method",
            list(mc.FANG_METHODS.keys()),
            help="All FANG correlations available in geomechpy.rock_strength, plus a constant-value fallback.",
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
            min_value=0.5, max_value=2.0, value=1.0, step=0.05,
            help="Scales the UCS output (and TSTR derived from it) — calibrate against core, "
            "like the static YME multiplier.",
        )
        tstr_multiplier = st.slider(
            "Tensile strength / UCS ratio",
            min_value=0.05, max_value=0.30, value=0.15, step=0.01,
            help="TSTR = ratio × UCS (geomechpy default is 0.15).",
        )

    with st.expander("4. Mechanical stratigraphy", expanded=False):
        # (2) Simplified lithology: one GR cutoff -> sandstone (0) vs shale (1)
        compute_litho = st.checkbox(
            "Flag lithology from GR",
            value=True,
            help="Split each sample into sandstone (0) or shale (1) with a single GR cutoff. "
            "The flag is shown on the stratigraphy tab and as a track on every output plot.",
        )
        gr_cutoff = None
        if compute_litho:
            gr_cutoff = st.slider(
                "GR cutoff (gAPI)",
                min_value=0.0, max_value=200.0, value=mc.DEFAULT_GR_CUTOFF, step=1.0,
                help="GR below the cutoff = sandstone (0); at or above = shale (1).",
            )

    with st.expander("7. Stress & wellbore stability", expanded=False):
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
    gr_cutoff=gr_cutoff,
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


def with_litho(cols: list[str]) -> list[str]:
    """Insert the lithology display column right after DEPTH when available,
    so every results table shows lithology next to the output properties."""
    if has_litho and N.get("LITHO_CODE") and N["LITHO_CODE"] not in cols:
        return [cols[0], N["LITHO_CODE"]] + cols[1:]
    return cols


# Lithology code series passed to every depth plot so the flag renders as a track.
litho_arg = disp[N["LITHO_CODE"]] if (has_litho and disp is not None) else None


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
        "Each depth is flagged from **GR** using the single cutoff defined in the sidebar "
        "(**6. Mechanical stratigraphy**): GR below the cutoff = **sandstone (code 0)**, at or above "
        "= **shale (code 1)**. The flag is carried through and shown as a track on every output plot."
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
                    height=620, litho_codes=litho_arg,
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
        ovb_cols = [DEPTH] + [N[c] for c in ["SV_MPA", "PP_MPA", "MW_SV_GCC", "MW_PP_GCC"]]
        st.dataframe(style_flags(disp, flags_disp, with_litho(ovb_cols)), use_container_width=True, height=380)
        mw_unit = mc.display_unit("MW_PP_GCC", unit_system)
        st.plotly_chart(
            depth_track_figure(
                disp, DEPTH,
                [
                    (f"Pressure ({mc.display_unit('SV_MPA', unit_system)})", [(N["SV_MPA"], "Sv"), (N["PP_MPA"], "Pp")]),
                    (f"Equivalent gradients ({mw_unit})", [(N["MW_SV_GCC"], "Sv EMW"), (N["MW_PP_GCC"], "Pp EMW")]),
                ],
                height=650, litho_codes=litho_arg,
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

        st.markdown("**Dynamic elastic properties** (from DTCO / DTSM / RHOB)")
        dyn_cols = [DEPTH] + [N[c] for c in ["VP_MS", "VS_MS", "VPVS", "YME_DYN_GPA", "PR_DYN", "K_DYN_GPA", "G_DYN_GPA", "LAME_DYN_GPA", "M_DYN_GPA"]]
        st.dataframe(style_flags(disp, flags_disp, with_litho(dyn_cols)), use_container_width=True, height=300)

        st.markdown("**Static elastic + rock strength**")
        sta_cols = [DEPTH] + [N[c] for c in ["YME_DYN_GPA", "YME_STA_GPA", "PR_DYN", "PR_STA", "UCS_MPA", "TSTR_MPA", "FANG_DEG"]]
        st.dataframe(style_flags(disp, flags_disp, with_litho(sta_cols)), use_container_width=True, height=300)

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
                height=680, litho_codes=litho_arg,
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
        hs_cols = [DEPTH] + [N[c] for c in ["SV_MPA", "PP_MPA", "SHMIN_MPA", "SHMAX_MPA", "Q_FACTOR", "SH_RATIO"]]
        st.dataframe(style_flags(disp, flags_disp, with_litho(hs_cols)), use_container_width=True, height=380)

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
                height=650, litho_codes=litho_arg,
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
            "Loss gradient: minimum principal stress among Sv, SHmax and Shmin — drilling above it risks losses. "
            "The green band is the safe mud weight window (**breakout limit → loss gradient**). "
            "Breakdown (fracture initiation, Hubbert & Willis) is shown as a reference only."
        )

        mw_unit = mc.display_unit("MW_BREAKOUT_GCC", unit_system)
        # Safe window: breakout (lower bound) up to the loss gradient (upper bound = min principal stress)
        window = results["MW_LOSS_GCC"] - results["MW_BREAKOUT_GCC"]
        n_valid = int(window.notna().sum())
        n_closed = int((window < 0).sum())
        c1, c2, c3 = st.columns(3)
        factor = mc.PPG_PER_GCC if unit_system == mc.OILFIELD else 1.0
        c1.metric(f"Median breakout limit ({mw_unit})", f"{results['MW_BREAKOUT_GCC'].median() * factor:.2f}")
        c2.metric(f"Median loss gradient ({mw_unit})", f"{results['MW_LOSS_GCC'].median() * factor:.2f}")
        c3.metric(f"Median window width ({mw_unit})", f"{window.median() * factor:.2f}")
        if n_closed:
            st.warning(
                f"⚠️ The mud weight window is closed (breakout limit above the loss gradient) at "
                f"{n_closed} of {n_valid} depth samples — no safe static mud weight exists there."
            )

        wbs_cols = [DEPTH] + [N[c] for c in ["PW_BREAKOUT_MPA", "PW_BREAKDOWN_MPA", "LOSS_P_MPA", "MW_BREAKOUT_GCC", "MW_BREAKDOWN_GCC", "MW_LOSS_GCC", "MW_PP_GCC", "MW_SV_GCC"]]
        st.dataframe(style_flags(disp, flags_disp, with_litho(wbs_cols)), use_container_width=True, height=340)

        # (5) Interactive mud-weight planning: MW + ECD sliders and casing setting depths.
        st.markdown("#### 🛠️ Mud weight planning")
        bo_disp = results["MW_BREAKOUT_GCC"] * factor  # display MW units (ppg or g/cc)
        loss_disp = results["MW_LOSS_GCC"] * factor
        lo = float(np.nanmin(bo_disp)) if bo_disp.notna().any() else 8.0
        hi = float(np.nanmax(loss_disp)) if loss_disp.notna().any() else 18.0
        pad = max(0.5, 0.1 * (hi - lo))
        slo, shi = round(lo - pad, 1), round(hi + pad, 1)
        mid = round((lo + hi) / 2, 2)
        step = 0.05 if unit_system == mc.OILFIELD else 0.01

        cmw, cecd = st.columns(2)
        mw_value = cmw.slider(
            f"Mud weight — MW ({mw_unit})", slo, shi, value=mid, step=step, key="wbs_mw",
            help="Planned static mud weight. Slide left/right to move the green MW line on the plot; "
            "keep it inside the safe window (breakout → loss gradient).",
        )
        ecd_value = cecd.slider(
            f"Equivalent circulating density — ECD ({mw_unit})", slo, shi + pad,
            value=round(min(mw_value + (0.3 if unit_system == mc.OILFIELD else 0.04), shi + pad), 2),
            step=step, key="wbs_ecd",
            help="Dynamic (circulating) density. Slide to move the orange ECD line; keep it below the loss gradient.",
        )

        st.caption(
            "**Casing setting depths** — add one row per casing shoe. These depths split the well into "
            "sections and draw horizontal markers on the plot to help pick MW per section."
        )
        casing_default = pd.DataFrame({f"Casing shoe depth ({depth_unit})": pd.Series([], dtype=float)})
        casing_edit = st.data_editor(
            casing_default, num_rows="dynamic", hide_index=True, use_container_width=True, key="wbs_casing",
        )
        casing_depths = sorted(
            float(v) for v in casing_edit.iloc[:, 0].tolist()
            if pd.notna(v) and np.isfinite(v)
        )

        # Per-section safe MW range (breakout .. loss gradient) between casing shoes.
        depth_num = pd.to_numeric(results["DEPTH"], errors="coerce")
        edges = [float(depth_num.min())] + casing_depths + [float(depth_num.max())]
        edges = sorted(set(round(e, 3) for e in edges))
        section_rows = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (depth_num >= a) & (depth_num <= b)
            if not m.any():
                continue
            sec_lo = float((results.loc[m, "MW_BREAKOUT_GCC"] * factor).max())  # highest breakout = min safe MW
            sec_hi = float((results.loc[m, "MW_LOSS_GCC"] * factor).min())      # lowest loss = max safe MW
            ok = "✅" if (np.isfinite(sec_lo) and np.isfinite(sec_hi) and sec_lo <= mw_value <= sec_hi) else "⚠️"
            section_rows.append({
                f"Top ({depth_unit})": round(a, 1),
                f"Base ({depth_unit})": round(b, 1),
                f"Min MW ({mw_unit})": round(sec_lo, 2),
                f"Max MW ({mw_unit})": round(sec_hi, 2),
                f"Planned MW ({mw_unit})": round(mw_value, 2),
                "MW in window?": ok,
            })
        if section_rows:
            st.markdown("**Safe MW range per section** (min = highest breakout, max = lowest loss gradient):")
            st.dataframe(pd.DataFrame(section_rows), use_container_width=True, hide_index=True)

        st.plotly_chart(
            mud_window_figure(disp, DEPTH, N, mw_unit, litho_codes=litho_arg,
                              mw_line=mw_value, ecd_line=ecd_value, casing_depths=casing_depths),
            use_container_width=True,
        )

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
            show_cols = with_litho([DEPTH] + [N[c] for c in flagged_canonical if c in N])
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
                    [(N["MW_BREAKOUT_GCC"], "MW min"), (N["MW_LOSS_GCC"], "MW max"), (N["MW_PP_GCC"], "Pp EMW")],
                ),
            ]
        st.plotly_chart(depth_track_figure(disp, DEPTH, composite_tracks, height=800, litho_codes=litho_arg), use_container_width=True)

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
