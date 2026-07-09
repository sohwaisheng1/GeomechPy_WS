"""
Quick MEM Calculator - Streamlit front end.

A one-page Mechanical Earth Model (MEM) starter built on the geomechpy
library: dynamic elastic properties, dynamic-to-static conversion and
rock strength, with QC flagging and interactive Plotly log displays.
Supports Oilfield and Metric unit systems for both input and display.

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
    "results_df": None,    # full workflow output (canonical units)
    "qc_summary": None,
    "qc_flags": None,      # per-sample QC flags (canonical column names)
    "data_source": None,   # label shown in the sidebar
    "load_messages": [],   # info messages from the data cleaner
    "unit_warnings": [],   # unit sanity warnings from the last run
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


# ---------------------------------------------------------------------------
# Sidebar: units, data input & calculation settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🪨 Quick MEM Calculator")
    st.caption(
        "1D Mechanical Earth Model starter kit powered by "
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
    expected = mc.INPUT_UNITS[unit_system]
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
        st.session_state.unit_warnings = mc.check_unit_sanity(
            st.session_state.raw_df, column_map, unit_system
        )
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
                unit_system=unit_system,
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

for warning in st.session_state.unit_warnings:
    st.warning(f"⚠️ Unit check: {warning}")

results = st.session_state.results_df
flags = st.session_state.qc_flags

# Convert canonical results into the selected display unit system.
# N maps canonical column names -> display names (e.g. YME_DYN_GPA -> 'YME_DYN [Mpsi]').
if results is not None:
    disp, N = mc.display_results(results, unit_system)
    flags_disp = flags.rename(columns=N) if flags is not None else None
    DEPTH = N["DEPTH"]
else:
    disp, N, flags_disp, DEPTH = None, {}, None, None

tab_input, tab_dyn, tab_sta, tab_strength, tab_qc, tab_viz = st.tabs(
    ["📥 Data Input", "🌊 Dynamic Props", "🧱 Static Props", "💪 Rock Strength", "✅ QC Report", "📊 Visualizations"]
)

# --- Tab 1: Data input ------------------------------------------------------
with tab_input:
    if st.session_state.raw_df is None:
        st.info("👈 Upload a CSV/Excel file or click **Load Sample Data** in the sidebar to get started.")
        unit_rows = "\n".join(
            f"| {curve} | {desc} | {expected[curve]} |"
            for curve, desc in [
                ("DEPTH", "Measured depth (MD)"),
                ("GR", "Gamma ray"),
                ("RHOB", "Bulk density"),
                ("DTCO", "Compressional slowness"),
                ("DTSM", "Shear slowness"),
                ("POROSITY", "Total/effective porosity (optional)"),
            ]
        )
        st.markdown(
            f"""
            **Expected input columns for {unit_system}** (any names — map them in the sidebar):

            | Curve | Description | Unit |
            |---|---|---|
            {unit_rows}

            Use **Download Example File** in the sidebar to get a correctly formatted template.
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
        st.subheader(f"Dynamic elastic properties ({unit_system})")
        st.caption("Slowness converted to velocity (v = 304800/Δt[µs/ft]), moduli computed by geomechpy.")
        dyn_cols = [DEPTH] + [N[c] for c in ["VP_MS", "VS_MS", "VPVS", "YME_DYN_GPA", "PR_DYN", "K_DYN_GPA", "G_DYN_GPA", "LAME_DYN_GPA", "M_DYN_GPA"]]
        st.dataframe(style_flags(disp, flags_disp, dyn_cols), use_container_width=True, height=420)
        st.plotly_chart(
            depth_track_figure(
                disp,
                DEPTH,
                [
                    (f"Velocities ({mc.display_unit('VP_MS', unit_system)})", [(N["VP_MS"], "Vp"), (N["VS_MS"], "Vs")]),
                    (f"Young's mod. ({mc.display_unit('YME_DYN_GPA', unit_system)})", [(N["YME_DYN_GPA"], "E dyn")]),
                    ("Poisson's ratio", [(N["PR_DYN"], "ν dyn")]),
                    (f"Bulk / Shear ({mc.display_unit('K_DYN_GPA', unit_system)})", [(N["K_DYN_GPA"], "K dyn"), (N["G_DYN_GPA"], "G dyn")]),
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
        st.subheader(f"Static elastic properties ({unit_system})")
        st.caption(f"Method: **{method_label}** · calibration multiplier ×{calibration_multiplier:.2f} · PR multiplier ×{pr_multiplier:.2f}")
        sta_cols = [DEPTH] + [N[c] for c in ["YME_DYN_GPA", "YME_STA_GPA", "PR_DYN", "PR_STA"]]
        st.dataframe(style_flags(disp, flags_disp, sta_cols), use_container_width=True, height=420)

        fig = depth_track_figure(
            disp,
            DEPTH,
            [
                (f"Young's modulus ({mc.display_unit('YME_STA_GPA', unit_system)})", [(N["YME_DYN_GPA"], "E dynamic"), (N["YME_STA_GPA"], "E static")]),
                ("Poisson's ratio", [(N["PR_DYN"], "ν dynamic"), (N["PR_STA"], "ν static")]),
            ],
            height=650,
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 4: Rock strength -----------------------------------------------------
with tab_strength:
    if results is None:
        st.info("Run the calculation from the sidebar to see rock strength results.")
    else:
        st.subheader(f"Rock strength ({unit_system})")
        st.caption(
            "UCS from Plumb (1994) static-YME correlation · "
            f"TSTR = {tstr_multiplier:.2f} × UCS · friction angle from Lal (1999) shale correlation."
        )
        str_cols = [DEPTH] + [N[c] for c in ["UCS_MPA", "TSTR_MPA", "FANG_DEG"]]
        st.dataframe(style_flags(disp, flags_disp, str_cols), use_container_width=True, height=420)
        st.plotly_chart(
            depth_track_figure(
                disp,
                DEPTH,
                [
                    (f"UCS ({mc.display_unit('UCS_MPA', unit_system)})", [(N["UCS_MPA"], "UCS")]),
                    (f"Tensile strength ({mc.display_unit('TSTR_MPA', unit_system)})", [(N["TSTR_MPA"], "TSTR")]),
                    ("Friction angle (°)", [(N["FANG_DEG"], "FANG")]),
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
        st.caption(
            "Each curve is checked against standard geomechanical ranges "
            "(expressed in the units shown in the Unit column). "
            "LOW/HIGH = outside range, MISSING = null/non-numeric."
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
                height=350,
            )
        else:
            st.success("All samples passed QC — no flags raised. 🎉")

# --- Tab 6: Visualizations ------------------------------------------------------
with tab_viz:
    if results is None:
        st.info("Run the calculation from the sidebar to see the composite MEM plot.")
    else:
        st.subheader(f"Composite MEM display ({unit_system})")
        st.plotly_chart(
            depth_track_figure(
                disp,
                DEPTH,
                [
                    ("GR (gAPI)", [(N["GR"], "GR")]),
                    (f"RHOB ({mc.display_unit('RHOB', unit_system)})", [(N["RHOB"], "RHOB")]),
                    (f"Slowness ({mc.display_unit('DTCO', unit_system)})", [(N["DTCO"], "DTCO"), (N["DTSM"], "DTSM")]),
                    (f"E ({mc.display_unit('YME_DYN_GPA', unit_system)})", [(N["YME_DYN_GPA"], "E dyn"), (N["YME_STA_GPA"], "E sta")]),
                    ("ν (-)", [(N["PR_DYN"], "ν dyn"), (N["PR_STA"], "ν sta")]),
                    (f"UCS / TSTR ({mc.display_unit('UCS_MPA', unit_system)})", [(N["UCS_MPA"], "UCS"), (N["TSTR_MPA"], "TSTR")]),
                    ("FANG (°)", [(N["FANG_DEG"], "FANG")]),
                ],
                height=800,
            ),
            use_container_width=True,
        )

        st.subheader("Crossplot explorer")
        numeric_cols = [c for c in disp.columns if pd.api.types.is_numeric_dtype(disp[c])]
        c1, c2, c3 = st.columns(3)
        x_col = c1.selectbox("X axis", numeric_cols, index=numeric_cols.index(N["YME_DYN_GPA"]) if N.get("YME_DYN_GPA") in numeric_cols else 0)
        y_col = c2.selectbox("Y axis", numeric_cols, index=numeric_cols.index(N["YME_STA_GPA"]) if N.get("YME_STA_GPA") in numeric_cols else 0)
        color_col = c3.selectbox("Color by", numeric_cols, index=numeric_cols.index(N["GR"]) if N.get("GR") in numeric_cols else 0)
        xfig = go.Figure(
            go.Scatter(
                x=disp[x_col],
                y=disp[y_col],
                mode="markers",
                marker=dict(color=disp[color_col], colorscale="Viridis", showscale=True, colorbar_title=color_col),
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
        f"⬇️ Download results as CSV ({unit_system})",
        data=mc.results_to_csv_bytes(disp),
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
