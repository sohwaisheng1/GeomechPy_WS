"""
Stress Barrier Analysis & Perforation Planner — Streamlit front end.

A focused, one-page tool built on top of the geomechpy library. Upload well
log data (CSV / Excel / LAS) with rock properties (Young's modulus, Poisson's
ratio), pore pressure and overburden, define the horizontal strains manually,
and the app will:

  1. flag lithology from a GR cutoff (reservoir sand vs non-reservoir shale),
  2. compute the horizontal stresses Shmin / SHmax with
     geomechpy.stress_calculations (poroelastic equation),
  3. analyse the stress contrast between reservoir and non-reservoir sections
     to locate stress barriers, and
  4. recommend perforation zones graded Good / Moderate / Poor.

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils import stress_barrier as sb

# ---------------------------------------------------------------------------
# Page setup & session state
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Stress Barrier Analysis & Perforation Planner",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

for key, default in {
    "raw_df": None,        # uploaded / sample input data
    "results_df": None,    # stress workflow output (canonical units)
    "analysis": None,      # barrier / perforation analysis output
    "data_source": None,   # label shown in the sidebar
    "load_messages": [],   # info messages from the data cleaner
}.items():
    st.session_state.setdefault(key, default)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _add_lithology_column(fig: go.Figure, depth_series, code_series, col: int, row: int = 1) -> None:
    """Draw the lithology flag as a colored track (contiguous runs by code)."""
    depth = pd.to_numeric(depth_series, errors="coerce").to_numpy(dtype=float)
    codes = pd.to_numeric(code_series, errors="coerce").to_numpy(dtype=float)
    filled = np.where(np.isfinite(codes), codes, -1.0)
    n = len(filled)
    present: list[int] = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and filled[j + 1] == filled[i]:
            j += 1
        c = int(filled[i])
        y0 = depth[i]
        y1 = depth[j + 1] if (j + 1) < n else depth[j]
        fig.add_shape(type="rect", x0=0.0, x1=1.0, y0=y0, y1=y1,
                      fillcolor=sb.LITHO_COLORS.get(c, "#ecf0f1"), line_width=0,
                      layer="below", row=row, col=col)
        present.append(c)
        i = j + 1
    for c in sorted(set(present)):
        name = sb.LITHO_NAME_BY_CODE.get(c, "Undefined")
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode="markers",
                       marker=dict(size=11, color=sb.LITHO_COLORS.get(c, "#ecf0f1")),
                       name=name, legendgroup="litho"),
            row=row, col=col,
        )
    fig.update_xaxes(visible=False, range=[0.0, 1.0], row=row, col=col)


def _add_quality_column(fig: go.Figure, depth_series, quality_series, col: int, row: int = 1) -> None:
    """Draw the perforation-quality flag as a colored track (contiguous runs)."""
    depth = pd.to_numeric(depth_series, errors="coerce").to_numpy(dtype=float)
    qual = quality_series.astype(object).to_numpy()
    n = len(qual)
    present: list[str] = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and qual[j + 1] == qual[i]:
            j += 1
        q = str(qual[i])
        y0 = depth[i]
        y1 = depth[j + 1] if (j + 1) < n else depth[j]
        fig.add_shape(type="rect", x0=0.0, x1=1.0, y0=y0, y1=y1,
                      fillcolor=sb.PERF_COLORS.get(q, "#ecf0f1"), line_width=0,
                      layer="below", row=row, col=col)
        present.append(q)
        i = j + 1
    for q in [g for g in sb.PERF_QUALITIES if g in set(present)]:
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode="markers",
                       marker=dict(size=11, color=sb.PERF_COLORS.get(q, "#ecf0f1")),
                       name=f"{q} perf", legendgroup="perf"),
            row=row, col=col,
        )
    fig.update_xaxes(visible=False, range=[0.0, 1.0], row=row, col=col)


def composite_figure(results: pd.DataFrame, detail: pd.DataFrame, zones: pd.DataFrame,
                     height: int = 820) -> go.Figure:
    """Multi-track log plot: Litho | GR | stresses | stress contrast | perf quality."""
    depth = results["DEPTH"]
    titles = ["Litho", "GR (gAPI)", "Stresses (psi)", "Stress contrast (psi)", "Perf"]
    widths_raw = [0.6, 1.6, 2.6, 2.0, 0.6]
    total = sum(widths_raw)
    fig = make_subplots(
        rows=1, cols=5, shared_yaxes=True, horizontal_spacing=0.02,
        subplot_titles=titles, column_widths=[w / total for w in widths_raw],
    )

    # Col 1 — lithology flag
    _add_lithology_column(fig, depth, results["LITHO_CODE"], col=1)

    # Col 2 — GR with the cutoff for reference
    if "GR" in results.columns and results["GR"].notna().any():
        fig.add_trace(
            go.Scatter(x=results["GR"], y=depth, mode="lines", name="GR",
                       line=dict(color="#2c3e50"),
                       hovertemplate="GR: %{x:.1f}<br>Depth: %{y:.1f}<extra></extra>"),
            row=1, col=2,
        )
    fig.update_xaxes(title_text="gAPI", row=1, col=2)

    # Col 3 — Pp, Sv, Shmin, SHmax
    stress_curves = [
        ("SV_PSI", "Sv", "#8e44ad"),
        ("SHMAX_PSI", "SHmax", "#c0392b"),
        ("SHMIN_PSI", "Shmin", "#2980b9"),
        ("PP_PSI", "Pp", "#7f8c8d"),
    ]
    for canonical, name, color in stress_curves:
        if canonical in results.columns:
            fig.add_trace(
                go.Scatter(x=results[canonical], y=depth, mode="lines", name=name,
                           line=dict(color=color),
                           hovertemplate=f"{name}: %{{x:.0f}} psi<br>Depth: %{{y:.1f}}<extra></extra>"),
                row=1, col=3,
            )
    # Shade the recommended perforation zones across the stress track.
    for _, z in zones.iterrows():
        fig.add_shape(type="rect", xref="x domain", x0=0.0, x1=1.0,
                      y0=z["Top"], y1=z["Base"],
                      fillcolor=sb.PERF_COLORS.get(z["Quality"], "#ecf0f1"),
                      opacity=0.18, line_width=0, layer="below", row=1, col=3)
    fig.update_xaxes(title_text="psi", row=1, col=3)

    # Col 4 — stress contrast (Shmin minus trend), zero reference line
    fig.add_trace(
        go.Scatter(x=detail["CONTRAST_PSI"], y=detail["DEPTH"], mode="lines",
                   name="Contrast", line=dict(color="#16a085"),
                   hovertemplate="Contrast: %{x:.0f} psi<br>Depth: %{y:.1f}<extra></extra>"),
        row=1, col=4,
    )
    fig.add_vline(x=0.0, line=dict(color="gray", dash="dot"), row=1, col=4)
    fig.update_xaxes(title_text="psi", row=1, col=4)

    # Col 5 — perforation quality flag
    _add_quality_column(fig, detail["DEPTH"], detail["PERF_QUALITY"], col=5)

    fig.update_yaxes(autorange="reversed", title_text="MD", col=1)
    fig.update_layout(
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.06),
        margin=dict(t=110, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar — inputs & parameters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🎯 Perforation Planner")
    st.caption(
        "Stress barrier analysis & perforation planning powered by "
        "[geomechpy](https://github.com/sohwaisheng1/GeomechPy_WS)."
    )

    with st.expander("1. Data input", expanded=True):
        uploaded = st.file_uploader(
            "Upload well log data (CSV / Excel / LAS)",
            type=["csv", "txt", "xls", "xlsx", "las"],
            help="One row per depth sample. Expected curves: DEPTH, GR, PP (pore "
            "pressure), SV/OVB (overburden), YME (Young's modulus), PR (Poisson's "
            "ratio). Unit rows and -999.25/-9999 nulls are handled automatically.",
        )
        if uploaded is not None:
            try:
                st.session_state.raw_df, st.session_state.load_messages = sb.load_data(uploaded)
                st.session_state.data_source = f"📄 {uploaded.name}"
                st.session_state.results_df = None
                st.session_state.analysis = None
            except ValueError as exc:
                st.error(str(exc))

        c_sample, c_dl = st.columns(2)
        if c_sample.button("🧪 Load Sample", use_container_width=True):
            st.session_state.raw_df = sb.generate_sample_data()
            st.session_state.data_source = "🧪 Synthetic sand/shale well (3000–3600 m)"
            st.session_state.load_messages = []
            st.session_state.results_df = None
            st.session_state.analysis = None
        c_dl.download_button(
            "⬇️ Template", data=sb.sample_csv_bytes(),
            file_name="perforation_planner_example.csv", mime="text/csv",
            use_container_width=True, help="Correctly formatted example CSV.",
        )

        if st.session_state.data_source:
            st.success(f"Loaded: {st.session_state.data_source}")
        for msg in st.session_state.load_messages:
            st.info(msg)

    with st.expander("2. Column mapping", expanded=True):
        column_map: dict[str, str] = {}
        if st.session_state.raw_df is not None:
            cols = list(st.session_state.raw_df.columns)
            options = ["-- not mapped --"] + cols
            for curve in sb.ALL_CURVES:
                required = curve in sb.REQUIRED_CURVES
                label = f"{sb.CURVE_LABELS[curve]} {'(required)' if required else '(optional)'}"
                choice = st.selectbox(
                    label, options, index=sb.guess_column(curve, cols), key=f"map_{curve}",
                )
                column_map[curve] = "" if choice == "-- not mapped --" else choice
        else:
            st.info("Load data first to map columns.")

    with st.expander("3. Input units", expanded=False):
        yme_unit = st.selectbox("Young's modulus unit", list(sb.YME_INPUT_UNITS.keys()), index=0,
                                help="Converted internally to Mpsi for the poroelastic equation.")
        pressure_unit = st.selectbox("Pressure unit (PP / Sv)", list(sb.PRESSURE_INPUT_UNITS.keys()),
                                     index=0, help="Applies to the PP and Sv log columns. Output is always psi.")
        depth_unit = st.radio("Depth (MD) unit", sb.DEPTH_UNITS, index=0, horizontal=True,
                              help="Used to convert MD to TVD (ft) for gradient-based Pp/Sv.")

    with st.expander("4. Pore pressure & overburden source", expanded=False):
        pp_source = sb.PP_SV_SOURCES[st.selectbox(
            "Pore pressure (PP)", list(sb.PP_SV_SOURCES.keys()), index=0,
            help="Use the mapped PP column, or derive Pp from a gradient with geomechpy.",
        )]
        pp_gradient = st.number_input("Pp gradient (psi/ft)", value=0.465, min_value=0.30,
                                      max_value=1.10, step=0.005, format="%.3f",
                                      disabled=(pp_source == "column"))
        sv_source = sb.PP_SV_SOURCES[st.selectbox(
            "Overburden (Sv)", list(sb.PP_SV_SOURCES.keys()), index=0,
            help="Use the mapped Sv/OVB column, or derive Sv from a lithostatic gradient with geomechpy.",
        )]
        ovb_gradient = st.number_input("Sv gradient (psi/ft)", value=1.02, min_value=0.60,
                                       max_value=1.30, step=0.005, format="%.3f",
                                       disabled=(sv_source == "column"))
        air_gap = st.number_input(f"Air gap / KB ({depth_unit})", value=0.0, min_value=0.0,
                                  format="%.1f", help="Only used when Pp/Sv are derived from a gradient.")

    with st.expander("5. Horizontal strains & stress model", expanded=True):
        st.caption("Manually define the horizontal strains used by the poroelastic equation.")
        eps_h = st.number_input("Minimum horizontal strain εh", value=0.0001, step=0.0001,
                                format="%.5f", help="Strain in the Shmin direction (EX in geomechpy).")
        eps_H = st.number_input("Maximum horizontal strain εH", value=0.0009, step=0.0001,
                                format="%.5f", help="Strain in the SHmax direction (EY). Keep εH ≥ εh.")
        if eps_H < eps_h:
            st.warning("εH < εh — SHmax may fall below Shmin. Set εH ≥ εh for a physical result.")
        biot = st.slider("Biot coefficient", 0.5, 1.0, 1.0, 0.05)
        shmax_method = sb.SHMAX_METHODS[st.selectbox("SHmax method", list(sb.SHMAX_METHODS.keys()), index=0)]
        shmax_multiplier = st.slider("SHmax / Shmin multiplier", 1.0, 2.0, 1.1, 0.05,
                                     disabled=(shmax_method != "multiplier"))

    with st.expander("6. Lithology (GR cutoff)", expanded=True):
        gr_cutoff = st.slider("GR cutoff (gAPI)", 0.0, 200.0, sb.DEFAULT_GR_CUTOFF, 1.0,
                              help="GR < cutoff → reservoir sand (0); GR ≥ cutoff → non-reservoir shale (1).")

    with st.expander("7. Barrier & perforation screening", expanded=True):
        contrast_threshold = st.slider(
            "Stress contrast threshold (psi)", 50.0, 2000.0, 300.0, 50.0,
            help="Minimum Shmin increase in the adjacent shale for it to count as a stress barrier.",
        )
        min_zone_thickness = st.number_input(
            f"Minimum reservoir thickness ({depth_unit})", value=5.0, min_value=0.0, step=1.0,
            format="%.1f", help="Reservoir intervals thinner than this are graded Poor.",
        )
        trend_window = st.slider("Contrast trend window (samples)", 5, 75, 25, 5,
                                 help="Window for the rolling-median Shmin trend used by the contrast curve.")

    st.divider()
    run_clicked = st.button("🚀 Run Analysis", type="primary", use_container_width=True,
                            disabled=st.session_state.raw_df is None)

# Bundle the configuration once.
stress_config = dict(
    yme_unit=yme_unit if st.session_state.raw_df is not None else "Mpsi",
    pressure_unit=pressure_unit if st.session_state.raw_df is not None else "psi",
    depth_unit=depth_unit if st.session_state.raw_df is not None else "m",
    pp_source=pp_source if st.session_state.raw_df is not None else "column",
    sv_source=sv_source if st.session_state.raw_df is not None else "column",
    pp_gradient_psift=pp_gradient if st.session_state.raw_df is not None else 0.465,
    ovb_gradient_psift=ovb_gradient if st.session_state.raw_df is not None else 1.02,
    air_gap=air_gap if st.session_state.raw_df is not None else 0.0,
    eps_h=eps_h if st.session_state.raw_df is not None else 0.0001,
    eps_H=eps_H if st.session_state.raw_df is not None else 0.0009,
    biot=biot if st.session_state.raw_df is not None else 1.0,
    shmax_method=shmax_method if st.session_state.raw_df is not None else "poroelastic",
    shmax_multiplier=shmax_multiplier if st.session_state.raw_df is not None else 1.1,
    gr_cutoff=gr_cutoff if st.session_state.raw_df is not None else sb.DEFAULT_GR_CUTOFF,
)

# ---------------------------------------------------------------------------
# Run the workflow
# ---------------------------------------------------------------------------

if run_clicked:
    try:
        with st.spinner("Computing horizontal stresses and screening perforation zones..."):
            results = sb.run_stress_workflow(st.session_state.raw_df, column_map, stress_config)
            analysis = sb.analyze_stress_barriers(
                results,
                contrast_threshold_psi=contrast_threshold,
                trend_window=int(trend_window),
                min_zone_thickness=float(min_zone_thickness),
            )
        st.session_state.results_df = results
        st.session_state.analysis = analysis
        st.toast("Analysis complete ✅")
    except ValueError as exc:
        st.error(f"⚠️ {exc}")
    except Exception as exc:  # keep the app alive on unexpected input
        st.error(f"Unexpected error during analysis: {exc}")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Stress Barrier Analysis & Perforation Planner")
st.markdown(
    "Compute horizontal stresses from rock properties with **geomechpy**, then locate "
    "**stress barriers** and recommend **perforation zones**. Configure inputs in the "
    "sidebar and click **🚀 Run Analysis**."
)

results = st.session_state.results_df
analysis = st.session_state.analysis

tab_input, tab_litho, tab_stress, tab_barrier, tab_perf = st.tabs(
    ["📥 Data Input", "🪨 Lithology", "↔️ Horizontal Stress", "🧱 Stress Barriers", "🎯 Perforation Zones"]
)

# --- Tab 1: Data input ------------------------------------------------------
with tab_input:
    st.subheader("Data input")
    ref = pd.DataFrame(
        [
            {"Curve": "DEPTH", "Description": sb.CURVE_LABELS["DEPTH"], "Requirement": "Required", "Example": "3000.0"},
            {"Curve": "YME", "Description": sb.CURVE_LABELS["YME"], "Requirement": "Required", "Example": "3.5 (Mpsi)"},
            {"Curve": "PR", "Description": sb.CURVE_LABELS["PR"], "Requirement": "Required", "Example": "0.25"},
            {"Curve": "GR", "Description": sb.CURVE_LABELS["GR"], "Requirement": "Optional (needed for lithology)", "Example": "75.0"},
            {"Curve": "PP", "Description": sb.CURVE_LABELS["PP"], "Requirement": "Optional (or from gradient)", "Example": "4600 (psi)"},
            {"Curve": "SV", "Description": sb.CURVE_LABELS["SV"], "Requirement": "Optional (or from gradient)", "Example": "10000 (psi)"},
        ]
    )
    st.markdown(
        "**Expected input columns** — one row per depth sample. Column *names* can be anything; "
        "map them to these curves in the sidebar (**2. Column mapping**). PP and Sv can either "
        "come from a log column or be derived from a gradient (**4. Pore pressure & overburden source**)."
    )
    st.dataframe(ref, use_container_width=True, hide_index=True)
    st.caption(
        "Tip: use **⬇️ Template** in the sidebar for a correctly formatted CSV, or **🧪 Load Sample** "
        "to try the app immediately. Horizontal strains εh / εH are defined manually in the sidebar."
    )

    if st.session_state.raw_df is None:
        st.info("👈 Upload a file or click **Load Sample** in the sidebar to get started.")
    else:
        df_in = st.session_state.raw_df
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df_in):,}")
        c2.metric("Columns", df_in.shape[1])
        depth_col = column_map.get("DEPTH") if column_map else None
        if depth_col and depth_col in df_in.columns:
            c3.metric("Depth range", f"{df_in[depth_col].min():.0f} – {df_in[depth_col].max():.0f} {depth_unit}")
        st.subheader("Uploaded data preview")
        st.dataframe(df_in, use_container_width=True, height=360)
        with st.expander("Basic statistics"):
            st.dataframe(df_in.describe().T, use_container_width=True)

# --- Tab 2: Lithology -------------------------------------------------------
with tab_litho:
    st.subheader("Lithology flag (from GR)")
    st.markdown(
        f"Each depth is flagged from **GR** using the cutoff **{gr_cutoff:g} gAPI**: "
        "GR below the cutoff = **reservoir sand (0)**, at or above = **non-reservoir shale (1)**."
    )
    if results is None:
        st.info("Run the analysis from the sidebar to generate the lithology flag.")
    elif not results["LITHO_CODE"].notna().any():
        st.warning("No lithology flag was produced — map a **GR** column in the sidebar and re-run.")
    else:
        counts = sb.lithology_counts(results)
        cols = st.columns(len(sb.LITHO_NAME_BY_CODE))
        for col, (code, name) in zip(cols, sb.LITHO_NAME_BY_CODE.items()):
            row = counts[counts["Code"] == code]
            pct = float(row["Fraction %"].iloc[0]) if not row.empty else 0.0
            col.metric(name, f"{pct:.1f}%")
        st.dataframe(counts, use_container_width=True, hide_index=True)

        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.03,
                            subplot_titles=["Litho", "GR (gAPI)"], column_widths=[0.25, 0.75])
        _add_lithology_column(fig, results["DEPTH"], results["LITHO_CODE"], col=1)
        fig.add_trace(go.Scatter(x=results["GR"], y=results["DEPTH"], mode="lines", name="GR",
                                 line=dict(color="#2c3e50")), row=1, col=2)
        fig.add_vline(x=gr_cutoff, line=dict(color="#e74c3c", dash="dash"), row=1, col=2)
        fig.update_yaxes(autorange="reversed", title_text="MD", col=1)
        fig.update_xaxes(title_text="gAPI", row=1, col=2)
        fig.update_layout(height=620, legend=dict(orientation="h", yanchor="bottom", y=1.05),
                          margin=dict(t=90, b=40))
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 3: Horizontal stress -----------------------------------------------
with tab_stress:
    st.subheader("Horizontal stresses (psi)")
    if results is None:
        st.info("Run the analysis from the sidebar to compute horizontal stresses.")
    elif not results["SHMIN_PSI"].notna().any():
        st.warning(
            "No horizontal stresses were produced. Check that YME, PR, PP and Sv are mapped "
            "(or Pp/Sv gradients are enabled) and that PR is between 0 and 0.5."
        )
    else:
        method_txt = [k for k, v in sb.SHMAX_METHODS.items() if v == shmax_method]
        st.caption(
            f"Poroelastic equation (Thiercelin & Plumb, 1994) · εh {eps_h:g} · εH {eps_H:g} · "
            f"Biot {biot:.2f} · SHmax: {method_txt[0] if method_txt else '-'}"
            + (f" (×{shmax_multiplier:.2f})" if shmax_method == "multiplier" else "")
        )
        q_med = pd.to_numeric(results["Q_FACTOR"], errors="coerce").median()
        c1, c2, c3 = st.columns(3)
        c1.metric("Median Shmin (psi)", f"{results['SHMIN_PSI'].median():,.0f}")
        c2.metric("Median SHmax (psi)", f"{results['SHMAX_PSI'].median():,.0f}")
        if pd.notna(q_med):
            regime = "Normal" if q_med < 1 else ("Strike-slip" if q_med < 2 else "Reverse")
            c3.metric("Stress regime", regime, help=f"median q-factor = {q_med:.2f}")

        show = sb.display_frame(results[["DEPTH", "GR", "YME_MPSI", "PR", "SV_PSI", "PP_PSI",
                                         "SHMIN_PSI", "SHMAX_PSI", "Q_FACTOR", "SH_RATIO"]])
        st.dataframe(show.style.format(precision=2), use_container_width=True, height=340)

        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.03,
                            subplot_titles=["Litho", "Stresses (psi)"], column_widths=[0.15, 0.85])
        _add_lithology_column(fig, results["DEPTH"], results["LITHO_CODE"], col=1)
        for canonical, name, color in [("SV_PSI", "Sv", "#8e44ad"), ("SHMAX_PSI", "SHmax", "#c0392b"),
                                       ("SHMIN_PSI", "Shmin", "#2980b9"), ("PP_PSI", "Pp", "#7f8c8d")]:
            fig.add_trace(go.Scatter(x=results[canonical], y=results["DEPTH"], mode="lines",
                                     name=name, line=dict(color=color)), row=1, col=2)
        fig.update_yaxes(autorange="reversed", title_text="MD", col=1)
        fig.update_xaxes(title_text="psi", row=1, col=2)
        fig.update_layout(height=650, legend=dict(orientation="h", yanchor="bottom", y=1.05),
                          margin=dict(t=90, b=40))
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 4: Stress barriers -------------------------------------------------
with tab_barrier:
    st.subheader("Stress barrier analysis")
    st.markdown(
        "The **stress contrast** is the difference between each sample's Shmin and its depth "
        "trend. Reservoir intervals bounded by higher-stress shale (Shmin contrast ≥ the "
        "threshold set in the sidebar) are contained by **stress barriers**."
    )
    if analysis is None:
        st.info("Run the analysis from the sidebar to identify stress barriers.")
    else:
        detail = analysis["detail"]
        barriers = analysis["barriers"]
        c1, c2 = st.columns(2)
        c1.metric("Stress barrier intervals", f"{len(barriers)}")
        c2.metric("Contrast threshold (psi)", f"{contrast_threshold:,.0f}")

        st.markdown("**Identified stress barriers** (non-reservoir intervals containing a neighbouring reservoir):")
        if barriers.empty:
            st.warning("No stress barriers met the contrast threshold — try lowering it in the sidebar.")
        else:
            st.dataframe(barriers.style.format(precision=1), use_container_width=True, hide_index=True)

        fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.03,
                            subplot_titles=["Litho", "Shmin & trend (psi)", "Contrast (psi)"],
                            column_widths=[0.15, 0.45, 0.40])
        _add_lithology_column(fig, results["DEPTH"], results["LITHO_CODE"], col=1)
        fig.add_trace(go.Scatter(x=detail["SHMIN_PSI"], y=detail["DEPTH"], mode="lines",
                                 name="Shmin", line=dict(color="#2980b9")), row=1, col=2)
        fig.add_trace(go.Scatter(x=detail["TREND_PSI"], y=detail["DEPTH"], mode="lines",
                                 name="Shmin trend", line=dict(color="#e67e22", dash="dash")), row=1, col=2)
        fig.add_trace(go.Scatter(x=detail["CONTRAST_PSI"], y=detail["DEPTH"], mode="lines",
                                 name="Contrast", line=dict(color="#16a085")), row=1, col=3)
        fig.add_vline(x=0.0, line=dict(color="gray", dash="dot"), row=1, col=3)
        for _, b in barriers.iterrows():
            fig.add_shape(type="rect", xref="x domain", x0=0.0, x1=1.0, y0=b["Top"], y1=b["Base"],
                          fillcolor="#7f8c8d", opacity=0.20, line_width=0, layer="below", row=1, col=2)
        fig.update_yaxes(autorange="reversed", title_text="MD", col=1)
        fig.update_xaxes(title_text="psi", row=1, col=2)
        fig.update_xaxes(title_text="psi", row=1, col=3)
        fig.update_layout(height=680, legend=dict(orientation="h", yanchor="bottom", y=1.05),
                          margin=dict(t=90, b=40))
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 5: Perforation zones -----------------------------------------------
with tab_perf:
    st.subheader("Recommended perforation zones")
    st.markdown(
        "Reservoir intervals are graded from their stress-barrier containment: "
        "**Good** = barrier above *and* below · **Moderate** = barrier on one side · "
        "**Poor** = uncontained or too thin."
    )
    if analysis is None:
        st.info("Run the analysis from the sidebar to generate perforation recommendations.")
    else:
        zones = analysis["zones"]
        detail = analysis["detail"]
        good = int((zones["Quality"] == "Good").sum()) if not zones.empty else 0
        mod = int((zones["Quality"] == "Moderate").sum()) if not zones.empty else 0
        poor = int((zones["Quality"] == "Poor").sum()) if not zones.empty else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 Good zones", good)
        c2.metric("🟠 Moderate zones", mod)
        c3.metric("🔴 Poor zones", poor)

        if zones.empty:
            st.warning("No reservoir intervals were found — check the GR cutoff and lithology flag.")
        else:
            def _color_quality(val):
                return f"background-color: {sb.PERF_COLORS.get(val, '#ecf0f1')}; color: black"

            st.markdown("**Perforation zone recommendations** (reservoir intervals, ranked by depth):")
            st.dataframe(
                zones.style.map(_color_quality, subset=["Quality"]).format(precision=1),
                use_container_width=True, hide_index=True,
            )
            recommended = zones[zones["Quality"].isin(["Good", "Moderate"])]
            if not recommended.empty:
                st.success(
                    f"✅ {len(recommended)} recommended perforation interval(s) "
                    f"(Good/Moderate). Best candidates are the **Good** zones with the "
                    "highest stress contrast above and below."
                )

        st.plotly_chart(
            composite_figure(results, detail, zones, height=820),
            use_container_width=True,
        )

        st.divider()
        st.subheader("Download results")
        dcol1, dcol2 = st.columns(2)
        full = results.copy()
        full["PERF_QUALITY"] = detail["PERF_QUALITY"].to_numpy()
        full["CONTRAST_PSI"] = detail["CONTRAST_PSI"].to_numpy()
        full["LITHOLOGY"] = sb.lithology_label_column(full).to_numpy()
        dcol1.download_button(
            "⬇️ Full results (per-depth) CSV",
            data=sb.results_to_csv_bytes(sb.display_frame(full)),
            file_name="perforation_planner_results.csv", mime="text/csv",
            type="primary", use_container_width=True,
        )
        dcol2.download_button(
            "⬇️ Perforation zones CSV",
            data=sb.results_to_csv_bytes(zones),
            file_name="perforation_zones.csv", mime="text/csv",
            use_container_width=True,
        )

st.divider()
st.caption(
    "Stress Barrier Analysis & Perforation Planner · built with "
    "[Streamlit](https://streamlit.io) + [geomechpy](https://github.com/sohwaisheng1/GeomechPy_WS) · "
    "horizontal stresses: Thiercelin & Plumb (1994)."
)

