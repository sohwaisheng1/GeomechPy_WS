"""
Quick MEM Calculator - Streamlit front end.

A one-page Mechanical Earth Model (MEM) starter built on the geomechpy
library: dynamic elastic properties, dynamic-to-static conversion and
rock strength, with QC flagging and interactive Plotly log displays.

Run locally:   streamlit run app.py
"""

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
    "results_df": None,    # full workflow output
    "qc_summary": None,
    "qc_flags": None,
    "data_source": None,   # label shown in the sidebar
}.items():
    st.session_state.setdefault(key, default)

FLAG_COLORS = {
    "OK": "background-color: #1e7e34; color: white",
    "LOW": "background-color: #d39e00; color: black",
    "HIGH": "background-color: #c82333; color: white",
    "MISSING": "background-color: #6c757d; color: white",
}


def style_flags(results: pd.DataFrame, flags: pd.DataFrame, columns: list[str]) -> "pd.io.formats.style.Styler":
    """Color-code table cells according to their QC flag."""
    shown = [c for c in columns if c in results.columns]

    def _color(row_df: pd.DataFrame) -> pd.DataFrame:
        css = pd.DataFrame("", index=row_df.index, columns=row_df.columns)
        for col in row_df.columns:
            if col in flags.columns:
                css[col] = flags.loc[row_df.index, col].map(FLAG_COLORS).fillna("")
        return css

    return results[shown].style.apply(_color, axis=None).format(precision=3)


def depth_track_figure(df: pd.DataFrame, tracks: list[tuple[str, list[tuple[str, str]]]], height: int = 750) -> go.Figure:
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
                    y=df["DEPTH"],
                    mode="lines",
                    name=name,
                    hovertemplate=f"{name}: %{{x:.3f}}<br>Depth: %{{y:.1f}}<extra></extra>",
                ),
                row=1,
                col=i,
            )
    fig.update_yaxes(autorange="reversed", title_text="Depth", col=1)
    fig.update_layout(
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.06),
        margin=dict(t=90, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar: data input & calculation settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🪨 Quick MEM Calculator")
    st.caption(
        "1D Mechanical Earth Model starter kit powered by "
        "[geomechpy](https://github.com/sohwaisheng1/GeomechPy_WS)."
    )
    st.divider()

    st.header("1. Data input")
    uploaded = st.file_uploader(
        "Upload well log data (CSV / Excel)",
        type=["csv", "txt", "xls", "xlsx"],
        help="One row per depth sample. Required curves: DEPTH, GR, RHOB, DTCO, DTSM. POROSITY is optional.",
    )
    if uploaded is not None:
        try:
            st.session_state.raw_df = mc.load_data(uploaded)
            st.session_state.data_source = f"📄 {uploaded.name}"
        except ValueError as exc:
            st.error(str(exc))

    if st.button("🧪 Load Sample Data", use_container_width=True):
        st.session_state.raw_df = mc.generate_sample_data()
        st.session_state.data_source = "🧪 Synthetic sample well (2500-3000 m)"

    if st.session_state.data_source:
        st.success(f"Loaded: {st.session_state.data_source}")

    st.divider()
    st.header("2. Column mapping")
    column_map: dict[str, str] = {}
    if st.session_state.raw_df is not None:
        options = ["-- not mapped --"] + list(st.session_state.raw_df.columns)
        for curve in mc.ALL_CURVES:
            required = curve in mc.REQUIRED_CURVES
            label = f"{curve} {'(required)' if required else '(optional)'}"
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
    st.header("3. Static properties")
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
    st.header("4. Rock strength")
    tstr_multiplier = st.slider(
        "Tensile strength / UCS ratio",
        min_value=0.05,
        max_value=0.30,
        value=0.15,
        step=0.01,
        help="TSTR = ratio × UCS (geomechpy default is 0.15).",
    )

    st.divider()
    run_clicked = st.button(
        "🚀 Run MEM Calculation",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.raw_df is None,
    )

# ---------------------------------------------------------------------------
# Run the workflow
# ---------------------------------------------------------------------------

if run_clicked:
    try:
        with st.spinner("Computing dynamic, static and strength properties..."):
            results = mc.run_full_workflow(
                data=st.session_state.raw_df,
                column_map=column_map,
                method_label=method_label,
                calibration_multiplier=calibration_multiplier,
                pr_multiplier=pr_multiplier,
                tstr_multiplier=tstr_multiplier,
                custom_a=custom_a,
                custom_b=custom_b,
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
    "dynamic elastic properties → static conversion → rock strength, with built-in QC."
)

results = st.session_state.results_df
flags = st.session_state.qc_flags

tab_input, tab_dyn, tab_sta, tab_strength, tab_qc, tab_viz = st.tabs(
    ["📥 Data Input", "🌊 Dynamic Props", "🧱 Static Props", "💪 Rock Strength", "✅ QC Report", "📊 Visualizations"]
)

# --- Tab 1: Data input ------------------------------------------------------
with tab_input:
    if st.session_state.raw_df is None:
        st.info("👈 Upload a CSV/Excel file or click **Load Sample Data** in the sidebar to get started.")
        st.markdown(
            """
            **Expected input columns** (any names — map them in the sidebar):

            | Curve | Description | Unit |
            |---|---|---|
            | DEPTH | Measured depth | m or ft |
            | GR | Gamma ray | gAPI |
            | RHOB | Bulk density | g/cc |
            | DTCO | Compressional slowness | µs/ft |
            | DTSM | Shear slowness | µs/ft |
            | POROSITY | Total/effective porosity (optional) | fraction |
            """
        )
    else:
        df_in = st.session_state.raw_df
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df_in):,}")
        c2.metric("Columns", df_in.shape[1])
        depth_col = column_map.get("DEPTH") if column_map else None
        if depth_col:
            c3.metric("Depth range", f"{df_in[depth_col].min():.0f} – {df_in[depth_col].max():.0f}")
        st.subheader("Input data preview")
        st.dataframe(df_in, use_container_width=True, height=420)
        with st.expander("Basic statistics"):
            st.dataframe(df_in.describe().T, use_container_width=True)

# --- Tab 2: Dynamic properties ----------------------------------------------
with tab_dyn:
    if results is None:
        st.info("Run the calculation from the sidebar to see dynamic elastic properties.")
    else:
        st.subheader("Dynamic elastic properties (from DTCO / DTSM / RHOB)")
        st.caption("Slowness converted to velocity (v = 304800/Δt), moduli computed by geomechpy in Pa and reported in GPa.")
        dyn_cols = ["DEPTH", "VP_MS", "VS_MS", "VPVS", "YME_DYN_GPA", "PR_DYN", "K_DYN_GPA", "G_DYN_GPA", "LAME_DYN_GPA", "M_DYN_GPA"]
        st.dataframe(style_flags(results, flags, dyn_cols), use_container_width=True, height=420)
        st.plotly_chart(
            depth_track_figure(
                results,
                [
                    ("Velocities (m/s)", [("VP_MS", "Vp"), ("VS_MS", "Vs")]),
                    ("Young's mod. (GPa)", [("YME_DYN_GPA", "E dyn")]),
                    ("Poisson's ratio", [("PR_DYN", "ν dyn")]),
                    ("Bulk / Shear (GPa)", [("K_DYN_GPA", "K dyn"), ("G_DYN_GPA", "G dyn")]),
                ],
                height=650,
            ),
            use_container_width=True,
        )

# --- Tab 3: Static properties -----------------------------------------------
with tab_sta:
    if results is None:
        st.info("Run the calculation from the sidebar to see static elastic properties.")
    else:
        st.subheader("Static elastic properties")
        st.caption(f"Method: **{method_label}** · calibration multiplier ×{calibration_multiplier:.2f} · PR multiplier ×{pr_multiplier:.2f}")
        sta_cols = ["DEPTH", "YME_DYN_GPA", "YME_STA_GPA", "YME_STA_MPSI", "PR_DYN", "PR_STA"]
        st.dataframe(style_flags(results, flags, sta_cols), use_container_width=True, height=420)

        fig = depth_track_figure(
            results,
            [
                ("Young's modulus (GPa)", [("YME_DYN_GPA", "E dynamic"), ("YME_STA_GPA", "E static")]),
                ("Poisson's ratio", [("PR_DYN", "ν dynamic"), ("PR_STA", "ν static")]),
            ],
            height=650,
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 4: Rock strength -----------------------------------------------------
with tab_strength:
    if results is None:
        st.info("Run the calculation from the sidebar to see rock strength results.")
    else:
        st.subheader("Rock strength")
        st.caption(
            "UCS from Plumb (1994) static-YME correlation · "
            f"TSTR = {tstr_multiplier:.2f} × UCS · friction angle from Lal (1999) shale correlation."
        )
        str_cols = ["DEPTH", "UCS_PSI", "UCS_MPA", "TSTR_PSI", "TSTR_MPA", "FANG_DEG"]
        st.dataframe(style_flags(results, flags, str_cols), use_container_width=True, height=420)
        st.plotly_chart(
            depth_track_figure(
                results,
                [
                    ("UCS (MPa)", [("UCS_MPA", "UCS")]),
                    ("Tensile strength (MPa)", [("TSTR_MPA", "TSTR")]),
                    ("Friction angle (°)", [("FANG_DEG", "FANG")]),
                ],
                height=650,
            ),
            use_container_width=True,
        )

# --- Tab 5: QC report ---------------------------------------------------------
with tab_qc:
    if results is None or st.session_state.qc_summary is None:
        st.info("Run the calculation from the sidebar to generate the QC report.")
    else:
        qc = st.session_state.qc_summary
        status = mc.qc_status(qc)
        badge = {"PASS": "🟢 PASS", "WARNING": "🟡 WARNING", "FAIL": "🔴 FAIL"}[status]
        st.subheader(f"QC report — overall status: {badge}")
        st.caption("Each curve is checked against standard geomechanical ranges. LOW/HIGH = outside range, MISSING = null/non-numeric.")

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

        flagged_cols = [c for c in flags.columns if (flags[c] != "OK").any()]
        if flagged_cols:
            st.markdown("**Flagged samples** (rows where at least one curve is out of range or missing):")
            bad_rows = flags[flagged_cols].ne("OK").any(axis=1)
            show_cols = ["DEPTH"] + flagged_cols
            st.dataframe(
                style_flags(results.loc[bad_rows], flags.loc[bad_rows], show_cols),
                use_container_width=True,
                height=350,
            )
        else:
            st.success("All samples passed QC — no flags raised. 🎉")

# --- Tab 6: Visualizations ------------------------------------------------------
with tab_viz:
    if results is None:
        st.info("Run the calculation from the sidebar to see the composite MEM plot.")
    else:
        st.subheader("Composite MEM display")
        st.plotly_chart(
            depth_track_figure(
                results,
                [
                    ("GR (gAPI)", [("GR", "GR")]),
                    ("RHOB (g/cc)", [("RHOB", "RHOB")]),
                    ("Slowness (µs/ft)", [("DTCO", "DTCO"), ("DTSM", "DTSM")]),
                    ("E (GPa)", [("YME_DYN_GPA", "E dyn"), ("YME_STA_GPA", "E sta")]),
                    ("ν (-)", [("PR_DYN", "ν dyn"), ("PR_STA", "ν sta")]),
                    ("UCS / TSTR (MPa)", [("UCS_MPA", "UCS"), ("TSTR_MPA", "TSTR")]),
                    ("FANG (°)", [("FANG_DEG", "FANG")]),
                ],
                height=800,
            ),
            use_container_width=True,
        )

        st.subheader("Crossplot explorer")
        numeric_cols = [c for c in results.columns if pd.api.types.is_numeric_dtype(results[c])]
        c1, c2, c3 = st.columns(3)
        x_col = c1.selectbox("X axis", numeric_cols, index=numeric_cols.index("YME_DYN_GPA") if "YME_DYN_GPA" in numeric_cols else 0)
        y_col = c2.selectbox("Y axis", numeric_cols, index=numeric_cols.index("YME_STA_GPA") if "YME_STA_GPA" in numeric_cols else 0)
        color_col = c3.selectbox("Color by", numeric_cols, index=numeric_cols.index("GR") if "GR" in numeric_cols else 0)
        xfig = go.Figure(
            go.Scatter(
                x=results[x_col],
                y=results[y_col],
                mode="markers",
                marker=dict(color=results[color_col], colorscale="Viridis", showscale=True, colorbar_title=color_col),
                hovertemplate=f"{x_col}: %{{x:.3f}}<br>{y_col}: %{{y:.3f}}<extra></extra>",
            )
        )
        xfig.update_layout(xaxis_title=x_col, yaxis_title=y_col, height=520)
        st.plotly_chart(xfig, use_container_width=True)

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

if results is not None:
    st.divider()
    st.download_button(
        "⬇️ Download results as CSV",
        data=mc.results_to_csv_bytes(results),
        file_name="mem_results.csv",
        mime="text/csv",
        type="primary",
    )

st.divider()
st.caption(
    "Quick MEM Calculator · built with [Streamlit](https://streamlit.io) + "
    "[geomechpy](https://github.com/sohwaisheng1/GeomechPy_WS) · "
    "correlations: Bradford (1998), Najibi (2015), Fuller, Morales (1993), Plumb (1994), Lal (1999)."
)