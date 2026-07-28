"""
Quick MEM Calculator - calculation engine.

All geomechanics calculations are delegated to the geomechpy library:
    - geomechpy.elastic_properties        -> dynamic elastic properties
    - geomechpy.static_elastic_properties -> dynamic-to-static conversion
    - geomechpy.rock_strength             -> UCS / TSTR / friction angle

This module only handles unit systems and conversions, dataframe plumbing,
QC flagging and sample-data generation for the Streamlit front end.

Canonical internal units (everything is converted to these before
calculation, regardless of the selected input unit system):
    DTCO/DTSM us/ft · RHOB g/cc · velocities m/s · moduli GPa ·
    strength MPa · friction angle deg
"""

from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from geomechpy.elastic_properties import ElasticPropertiesConverter
from geomechpy.overburden_stress import OverburdenStressCalculation
from geomechpy.pore_pressure import PorePressureCalculation
from geomechpy.rock_strength import RockStrengthPropertiesConverter
from geomechpy.static_elastic_properties import StaticElasticPropertiesConverter
from geomechpy.stress_calculations import HorizontalStressesCalculation
from geomechpy.wellbore_stability import WellboreStabilityCalculation

# ---------------------------------------------------------------------------
# Constants & unit conversion
# ---------------------------------------------------------------------------

FT_TO_M_US = 304800.0          # us/ft slowness -> m/s velocity: v = 304800 / dt
M_PER_FT = 0.3048              # us/ft = us/m * 0.3048 ; ft/s = m/s / 0.3048
PA_TO_GPA = 1.0e-9             # Pascal -> GigaPascal
PA_TO_PSI = 1.0 / 6894.757293  # Pascal -> psi
PA_TO_MPSI = PA_TO_PSI * 1e-6  # Pascal -> Mega-psi
PSI_TO_MPA = 6894.757293e-6    # psi -> MPa
MPA_TO_PSI = 1.0 / PSI_TO_MPA  # MPa -> psi
GPA_TO_MPSI = PA_TO_MPSI / PA_TO_GPA  # GPa -> Mpsi (~0.145)
GCC_TO_KGM3 = 1000.0           # g/cc -> kg/m3
M_TO_FT = 1.0 / M_PER_FT       # metres -> feet
PSI_FT_PER_GCC = 0.4335        # hydrostatic gradient of 1 g/cc fluid in psi/ft
PPG_PER_GCC = 8.3454           # mud weight: 1 g/cc = 8.3454 ppg

# Common well-log null sentinels replaced with NaN on load.
NULL_SENTINELS = [-999.0, -999.25, -9999.0, -9999.25, -99999.0, 9999.0]

# Curves the app can map. POROSITY is optional unless the Morales method is used.
REQUIRED_CURVES = ["DEPTH", "GR", "RHOB", "DTCO", "DTSM"]
OPTIONAL_CURVES = ["POROSITY"]
ALL_CURVES = REQUIRED_CURVES + OPTIONAL_CURVES

# ---------------------------------------------------------------------------
# Unit systems
# ---------------------------------------------------------------------------

OILFIELD = "Oilfield Units"
METRIC = "Metric Units"
UNIT_SYSTEMS = [OILFIELD, METRIC]

# Expected INPUT units per system (shown in the UI and used to convert to
# canonical units before calculation).
INPUT_UNITS = {
    OILFIELD: {"DEPTH": "m", "GR": "gAPI", "RHOB": "g/cc", "DTCO": "µs/ft", "DTSM": "µs/ft", "POROSITY": "frac"},
    METRIC: {"DEPTH": "m", "GR": "gAPI", "RHOB": "kg/m³", "DTCO": "µs/m", "DTSM": "µs/m", "POROSITY": "frac"},
}

# Display spec: canonical column -> (display name, (oilfield unit, factor),
# (metric unit, factor)). Factor converts FROM the canonical value TO the
# displayed value. Order here defines display column order.
DISPLAY_SPEC: dict[str, tuple[str, tuple[str, float], tuple[str, float]]] = {
    "DEPTH": ("MD", ("", 1.0), ("", 1.0)),  # passed through in the input depth unit
    "GR": ("GR", ("gAPI", 1.0), ("gAPI", 1.0)),
    "RHOB": ("RHOB", ("g/cc", 1.0), ("kg/m³", GCC_TO_KGM3)),
    "DTCO": ("DTCO", ("µs/ft", 1.0), ("µs/m", 1.0 / M_PER_FT)),
    "DTSM": ("DTSM", ("µs/ft", 1.0), ("µs/m", 1.0 / M_PER_FT)),
    "POROSITY": ("POROSITY", ("frac", 1.0), ("frac", 1.0)),
    "VP_MS": ("VP", ("ft/s", 1.0 / M_PER_FT), ("m/s", 1.0)),
    "VS_MS": ("VS", ("ft/s", 1.0 / M_PER_FT), ("m/s", 1.0)),
    "VPVS": ("VP/VS", ("-", 1.0), ("-", 1.0)),
    "YME_DYN_GPA": ("YME_DYN", ("Mpsi", GPA_TO_MPSI), ("GPa", 1.0)),
    "PR_DYN": ("PR_DYN", ("-", 1.0), ("-", 1.0)),
    "K_DYN_GPA": ("K_DYN", ("Mpsi", GPA_TO_MPSI), ("GPa", 1.0)),
    "G_DYN_GPA": ("G_DYN", ("Mpsi", GPA_TO_MPSI), ("GPa", 1.0)),
    "LAME_DYN_GPA": ("LAME_DYN", ("Mpsi", GPA_TO_MPSI), ("GPa", 1.0)),
    "M_DYN_GPA": ("M_DYN", ("Mpsi", GPA_TO_MPSI), ("GPa", 1.0)),
    "YME_STA_GPA": ("YME_STA", ("Mpsi", GPA_TO_MPSI), ("GPa", 1.0)),
    "PR_STA": ("PR_STA", ("-", 1.0), ("-", 1.0)),
    "UCS_MPA": ("UCS", ("psi", MPA_TO_PSI), ("MPa", 1.0)),
    "TSTR_MPA": ("TSTR", ("psi", MPA_TO_PSI), ("MPa", 1.0)),
    "FANG_DEG": ("FANG", ("deg", 1.0), ("deg", 1.0)),
    # --- NEW: stress profile & wellbore stability columns ---
    "SV_MPA": ("SV", ("psi", MPA_TO_PSI), ("MPa", 1.0)),
    "PP_MPA": ("PP", ("psi", MPA_TO_PSI), ("MPa", 1.0)),
    "SHMIN_MPA": ("SHMIN", ("psi", MPA_TO_PSI), ("MPa", 1.0)),
    "SHMAX_MPA": ("SHMAX", ("psi", MPA_TO_PSI), ("MPa", 1.0)),
    "Q_FACTOR": ("Q_FACTOR", ("-", 1.0), ("-", 1.0)),
    "SH_RATIO": ("SHMAX/SHMIN", ("-", 1.0), ("-", 1.0)),
    "PW_BREAKOUT_MPA": ("PW_BREAKOUT", ("psi", MPA_TO_PSI), ("MPa", 1.0)),
    "PW_BREAKDOWN_MPA": ("PW_BREAKDOWN", ("psi", MPA_TO_PSI), ("MPa", 1.0)),
    "LOSS_P_MPA": ("LOSS_GRAD_P", ("psi", MPA_TO_PSI), ("MPa", 1.0)),  # NEW
    "MW_LOSS_GCC": ("LOSS_GRADIENT", ("ppg", PPG_PER_GCC), ("g/cc", 1.0)),  # NEW
    "MW_PP_GCC": ("EMW_PP", ("ppg", PPG_PER_GCC), ("g/cc", 1.0)),
    "MW_BREAKOUT_GCC": ("MW_BREAKOUT", ("ppg", PPG_PER_GCC), ("g/cc", 1.0)),
    "MW_BREAKDOWN_GCC": ("MW_BREAKDOWN", ("ppg", PPG_PER_GCC), ("g/cc", 1.0)),
    "MW_SV_GCC": ("EMW_SV", ("ppg", PPG_PER_GCC), ("g/cc", 1.0)),
}


def _spec(canonical: str, unit_system: str) -> tuple[str, str, float]:
    """(display base name, unit string, factor from canonical) for a column."""
    name, oilfield, metric = DISPLAY_SPEC[canonical]
    unit, factor = oilfield if unit_system == OILFIELD else metric
    return name, unit, factor


def display_name(canonical: str, unit_system: str) -> str:
    """Display column header, e.g. 'YME_DYN [Mpsi]'."""
    name, unit, _ = _spec(canonical, unit_system)
    return f"{name} [{unit}]" if unit not in ("", "-") else name


def display_unit(canonical: str, unit_system: str) -> str:
    """Unit string for the selected system, e.g. 'GPa' or 'Mpsi'."""
    return _spec(canonical, unit_system)[1]


def display_results(df: pd.DataFrame, unit_system: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """Convert a canonical results frame into the selected unit system.

    Returns:
        disp: converted DataFrame with unit-labelled column names.
        names: mapping canonical column -> display column name (for plots
               and for renaming QC flag columns).
    """
    disp = pd.DataFrame(index=df.index)
    names: dict[str, str] = {}
    for canonical in DISPLAY_SPEC:
        if canonical not in df.columns:
            continue
        _, _, factor = _spec(canonical, unit_system)
        label = display_name(canonical, unit_system)
        disp[label] = pd.to_numeric(df[canonical], errors="coerce") * factor
        names[canonical] = label
    return disp, names


def normalize_input_units(df: pd.DataFrame, unit_system: str) -> pd.DataFrame:
    """Convert mapped input columns (canonical names) into canonical units.

    Oilfield input is already canonical. Metric input: DT µs/m -> µs/ft,
    RHOB kg/m³ -> g/cc.
    """
    out = df.copy()
    if unit_system == METRIC:
        for col in ("DTCO", "DTSM"):
            if col in out.columns:
                out[col] = out[col] * M_PER_FT
        if "RHOB" in out.columns:
            out["RHOB"] = out["RHOB"] / GCC_TO_KGM3
    return out


def check_unit_sanity(data: pd.DataFrame, column_map: dict[str, str], unit_system: str) -> list[str]:
    """Heuristic warnings when the data magnitudes contradict the selected units."""
    warnings: list[str] = []

    def _median(curve: str) -> float:
        src = column_map.get(curve)
        if not src or src not in data.columns:
            return float("nan")
        return float(pd.to_numeric(data[src], errors="coerce").median())

    dt = _median("DTCO")
    rhob = _median("RHOB")
    if unit_system == OILFIELD:
        if np.isfinite(dt) and dt > 250:
            warnings.append(
                f"Median DTCO is {dt:.0f} — that looks like µs/m, but Oilfield Units expects µs/ft. "
                "Consider switching to Metric Units."
            )
        if np.isfinite(rhob) and rhob > 100:
            warnings.append(
                f"Median RHOB is {rhob:.0f} — that looks like kg/m³, but Oilfield Units expects g/cc. "
                "Consider switching to Metric Units."
            )
    else:
        if np.isfinite(dt) and dt < 130:
            warnings.append(
                f"Median DTCO is {dt:.0f} — that looks like µs/ft, but Metric Units expects µs/m. "
                "Consider switching to Oilfield Units."
            )
        if np.isfinite(rhob) and rhob < 10:
            warnings.append(
                f"Median RHOB is {rhob:.2f} — that looks like g/cc, but Metric Units expects kg/m³. "
                "Consider switching to Oilfield Units."
            )
    return warnings


# QC validation ranges in CANONICAL units: column -> (min, max, unit).
QC_RANGES = {
    "GR": (0.0, 250.0, "gAPI"),
    "RHOB": (1.5, 3.2, "g/cc"),
    "DTCO": (40.0, 240.0, "us/ft"),
    "DTSM": (60.0, 450.0, "us/ft"),
    "POROSITY": (0.0, 0.5, "frac"),
    "VPVS": (1.4, 2.4, "ratio"),
    "PR_DYN": (0.0, 0.5, "unitless"),
    "YME_DYN_GPA": (0.5, 130.0, "GPa"),
    "PR_STA": (0.0, 0.5, "unitless"),
    "YME_STA_GPA": (0.1, 110.0, "GPa"),
    "UCS_MPA": (1.0, 400.0, "MPa"),
    "TSTR_MPA": (0.1, 60.0, "MPa"),
    "FANG_DEG": (10.0, 55.0, "deg"),
    # NEW: stress profile sanity ranges
    "SV_MPA": (1.0, 300.0, "MPa"),
    "PP_MPA": (0.5, 200.0, "MPa"),
    "SHMIN_MPA": (0.5, 300.0, "MPa"),
    "SHMAX_MPA": (0.5, 400.0, "MPa"),
}

# NEW: selectable rock strength methods. geomechpy.rock_strength currently
# ships one correlation per property (Plumb UCS, Lal FANG); a constant-value
# option is provided as a generic app-level fallback for calibration work.
UCS_METHODS = {
    "Plumb (1994) — from static YME [geomechpy]": "plumb",
    "Constant value": "constant",
}
FANG_METHODS = {
    "Lal (1999) — from DTCO, shale [geomechpy]": "lal",
    "Constant value": "constant",
}

# Static Young's modulus correlations exposed in the UI.
# input_unit tells us which unit the geomechpy function expects for dynamic YME.
STATIC_YME_METHODS = {
    "Bradford (power law, North Sea sandstone)": {"key": "bradford", "input_unit": "Mpsi"},
    "Najibi (power law, Iranian carbonates)": {"key": "najibi", "input_unit": "Mpsi"},
    "Fuller (power law, sandstone/shale)": {"key": "fuller", "input_unit": "GPa"},
    "Morales (porosity-dependent, sandstone)": {"key": "morales", "input_unit": "Mpsi"},
    "Custom power law (y = a*x^b)": {"key": "custom_power", "input_unit": "Mpsi"},
    "Custom linear law (y = a*x + b)": {"key": "custom_linear", "input_unit": "Mpsi"},
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _drop_unit_rows(df: pd.DataFrame, max_rows: int = 3) -> tuple[pd.DataFrame, int]:
    """Drop leading rows that hold unit strings instead of data.

    A leading row is treated as a unit row when it is non-numeric in at least
    half of the columns whose remaining values are mostly numeric
    (e.g. a 'M | GAPI | G/CC | US/F' line under the header).
    """
    dropped = 0
    while len(df) > 1 and dropped < max_rows:
        first = df.iloc[0]
        body = df.iloc[1:]
        checkable = 0
        non_numeric = 0
        for col in df.columns:
            body_numeric_share = pd.to_numeric(body[col], errors="coerce").notna().mean()
            if body_numeric_share >= 0.6:
                checkable += 1
                first_val = pd.to_numeric(pd.Series([first[col]]), errors="coerce").iloc[0]
                if pd.isna(first_val):
                    non_numeric += 1
        if checkable and non_numeric / checkable >= 0.5:
            df = df.iloc[1:].reset_index(drop=True)
            dropped += 1
        else:
            break
    return df, dropped


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Robust cleanup of a freshly parsed log table.

    - drops leading unit rows,
    - coerces mostly-numeric columns to numeric (bad cells -> NaN),
    - replaces well-log null sentinels (-999.25, -9999, ...) and ±inf with NaN.

    Returns the cleaned frame and a list of informational messages.
    """
    messages: list[str] = []
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    df, dropped = _drop_unit_rows(df)
    if dropped:
        messages.append(f"Skipped {dropped} leading unit/header row(s) that contained no data.")

    n_nulls = 0
    for col in df.columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().mean() >= 0.5:  # mostly numeric -> treat as a log curve
            sentinel_mask = numeric.isin(NULL_SENTINELS) | ~np.isfinite(numeric.fillna(0.0))
            n_nulls += int(sentinel_mask.sum())
            numeric[sentinel_mask] = np.nan
            df[col] = numeric
    if n_nulls:
        messages.append(f"Replaced {n_nulls} null sentinel value(s) (e.g. -999.25 / -9999) with NaN.")

    df = df.dropna(how="all").reset_index(drop=True)
    return df, messages


def load_data(uploaded_file) -> tuple[pd.DataFrame, list[str]]:
    """Read an uploaded CSV or Excel file and clean it up.

    Returns (dataframe, informational messages).
    Raises ValueError with a user-friendly message on failure.
    """
    name = uploaded_file.name.lower()
    try:
        if name.endswith((".csv", ".txt")):
            df = pd.read_csv(uploaded_file, skip_blank_lines=True)
            # Single-column result usually means a non-comma delimiter: re-sniff.
            if df.shape[1] == 1:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=None, engine="python", skip_blank_lines=True)
        elif name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file type. Please upload a .csv, .xls or .xlsx file.")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not parse file '{uploaded_file.name}': {exc}") from exc

    if df.empty:
        raise ValueError("The uploaded file contains no rows.")
    if df.shape[1] < 2:
        raise ValueError("The uploaded file needs at least a depth column and one log curve.")

    df, messages = clean_dataframe(df)
    if df.empty:
        raise ValueError("No data rows remained after cleaning the file.")
    return df, messages


def guess_column(curve: str, columns: list[str]) -> int:
    """Best-effort index of the column matching a curve mnemonic (for selectbox defaults).

    Returns 0 ('-- not mapped --') when nothing matches.
    """
    aliases = {
        "DEPTH": ["depth", "dept", "md", "tvd"],
        "GR": ["gr", "gamma", "gapi", "cgr", "sgr"],
        "RHOB": ["rhob", "den", "density", "rho", "zden"],
        "DTCO": ["dtco", "dtc", "dt_p", "dtp", "ac", "dt4p", "dtcomp"],
        "DTSM": ["dtsm", "dts", "dt_s", "dt4s", "dtshear"],
        "POROSITY": ["porosity", "phit", "phie", "nphi", "por", "phi"],
    }
    lowered = [c.lower().strip() for c in columns]
    for alias in aliases[curve]:
        for i, col in enumerate(lowered):
            if col == alias or col.startswith(alias):
                return i + 1  # +1 for the '-- not mapped --' placeholder at index 0
    return 0


def missing_required_curves(columns: list[str]) -> list[str]:
    """Required curves that could not be auto-detected in the given columns."""
    return [c for c in REQUIRED_CURVES if guess_column(c, columns) == 0]


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

def generate_sample_data(n_points: int = 201, seed: int = 42, unit_system: str = OILFIELD) -> pd.DataFrame:
    """Generate a synthetic sand/shale well-log interval (2500-3000 m MD).

    Values are geologically plausible so all downstream calculations produce
    sensible magnitudes. Output columns: MD, GR, RHOB, DTCO, DTSM, POROSITY,
    expressed in the requested unit system.
    """
    rng = np.random.default_rng(seed)
    depth = np.linspace(2500.0, 3000.0, n_points)

    # Smooth sand/shale alternation driver (0 = clean sand, 1 = shale)
    vsh = 0.5 + 0.35 * np.sin(depth / 18.0) + 0.15 * np.sin(depth / 61.0)
    vsh = np.clip(vsh + rng.normal(0, 0.05, n_points), 0.02, 0.98)

    compaction = (depth - 2500.0) / 500.0  # 0 -> 1 over the interval

    gr = 25.0 + 110.0 * vsh + rng.normal(0, 4.0, n_points)
    rhob = 2.30 + 0.25 * compaction + 0.12 * vsh + rng.normal(0, 0.02, n_points)  # g/cc
    dtco = 95.0 - 25.0 * compaction + 18.0 * vsh + rng.normal(0, 1.5, n_points)   # us/ft
    dtsm = dtco * (1.65 + 0.25 * vsh) + rng.normal(0, 3.0, n_points)              # us/ft
    porosity = np.clip(0.28 - 0.12 * compaction - 0.08 * vsh + rng.normal(0, 0.01, n_points), 0.03, 0.35)

    if unit_system == METRIC:
        dtco = dtco / M_PER_FT       # us/ft -> us/m
        dtsm = dtsm / M_PER_FT
        rhob = rhob * GCC_TO_KGM3    # g/cc -> kg/m3

    return pd.DataFrame(
        {
            "MD": np.round(depth, 2),
            "GR": np.round(gr, 2),
            "RHOB": np.round(rhob, 3),
            "DTCO": np.round(dtco, 2),
            "DTSM": np.round(dtsm, 2),
            "POROSITY": np.round(porosity, 3),
        }
    )


def sample_csv_bytes(unit_system: str = OILFIELD) -> bytes:
    """Example CSV for the 'Download Example File' button."""
    buffer = io.StringIO()
    generate_sample_data(unit_system=unit_system).to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Dynamic elastic properties (geomechpy.elastic_properties)
# ---------------------------------------------------------------------------

def compute_dynamic_properties(df: pd.DataFrame) -> pd.DataFrame:
    """Compute dynamic elastic properties from DTCO/DTSM/RHOB (canonical units).

    Unit handling:
        DTCO, DTSM : us/ft  -> Vp, Vs in m/s   (v = 304800 / dt)
        RHOB       : g/cc   -> kg/m3           (x1000)
        geomechpy returns moduli in Pa -> reported in GPa (+ Mpsi for YME)

    Invalid rows (non-positive slowness/density, NaN) yield NaN outputs
    instead of raising.
    """
    out = df.copy()

    dtco = pd.to_numeric(out["DTCO"], errors="coerce").to_numpy(dtype=float)
    dtsm = pd.to_numeric(out["DTSM"], errors="coerce").to_numpy(dtype=float)
    rhob = pd.to_numeric(out["RHOB"], errors="coerce").to_numpy(dtype=float)

    n = len(out)
    vp = np.full(n, np.nan)
    vs = np.full(n, np.nan)
    cols = {
        k: np.full(n, np.nan)
        for k in ["YME_DYN_GPA", "PR_DYN", "K_DYN_GPA", "G_DYN_GPA", "LAME_DYN_GPA", "M_DYN_GPA"]
    }

    valid = (dtco > 0) & (dtsm > 0) & (rhob > 0)
    vp[valid] = FT_TO_M_US / dtco[valid]
    vs[valid] = FT_TO_M_US / dtsm[valid]

    for i in np.flatnonzero(valid):
        try:
            # geomechpy expects slowness in us/ft and density in kg/m3, returns Pa
            props = ElasticPropertiesConverter.convert_dynamic_elastic_properties_from_slowness(
                p_wave_slowness=float(dtco[i]),
                s_wave_slowness=float(dtsm[i]),
                density=float(rhob[i]) * GCC_TO_KGM3,
            )
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        cols["YME_DYN_GPA"][i] = props.youngs_modulus * PA_TO_GPA
        cols["PR_DYN"][i] = props.poissons_ratio
        cols["K_DYN_GPA"][i] = props.bulk_modulus * PA_TO_GPA
        cols["G_DYN_GPA"][i] = props.shear_modulus * PA_TO_GPA
        cols["LAME_DYN_GPA"][i] = props.lame_parameter * PA_TO_GPA
        cols["M_DYN_GPA"][i] = props.p_wave_modulus * PA_TO_GPA

    out["VP_MS"] = vp
    out["VS_MS"] = vs
    with np.errstate(divide="ignore", invalid="ignore"):
        out["VPVS"] = np.where(vs > 0, vp / vs, np.nan)
    for k, v in cols.items():
        out[k] = v
    out["YME_DYN_MPSI"] = out["YME_DYN_GPA"] * GPA_TO_MPSI
    return out


# ---------------------------------------------------------------------------
# Static elastic properties (geomechpy.static_elastic_properties)
# ---------------------------------------------------------------------------

def compute_static_properties(
    df: pd.DataFrame,
    method_label: str,
    calibration_multiplier: float = 1.0,
    pr_multiplier: float = 1.0,
    custom_a: float = 0.5,
    custom_b: float = 1.0,
) -> pd.DataFrame:
    """Convert dynamic to static elastic properties.

    Args:
        df: DataFrame that already contains YME_DYN_GPA / YME_DYN_MPSI / PR_DYN
            (and POROSITY when using the Morales method).
        method_label: key of STATIC_YME_METHODS selected in the UI.
        calibration_multiplier: global calibration factor (0.5-2.0 slider)
            applied to the correlation output.
        pr_multiplier: static/dynamic Poisson's ratio multiplier.
        custom_a, custom_b: coefficients for the custom power/linear laws
            (a = multiplier/slope, b = exponent/intercept).
    """
    method = STATIC_YME_METHODS[method_label]
    out = df.copy()
    conv = StaticElasticPropertiesConverter

    n = len(out)
    yme_sta_native = np.full(n, np.nan)  # in the method's native unit (Mpsi or GPa)
    yme_dyn_mpsi = out["YME_DYN_MPSI"].to_numpy(dtype=float)
    yme_dyn_gpa = out["YME_DYN_GPA"].to_numpy(dtype=float)

    if method["key"] == "morales":
        if "POROSITY" not in out.columns or out["POROSITY"].isna().all():
            raise ValueError("The Morales correlation requires a mapped POROSITY column.")
        por = pd.to_numeric(out["POROSITY"], errors="coerce").to_numpy(dtype=float)

    for i in range(n):
        yd_mpsi, yd_gpa = yme_dyn_mpsi[i], yme_dyn_gpa[i]
        if not np.isfinite(yd_mpsi) or yd_mpsi <= 0:
            continue
        try:
            if method["key"] == "bradford":
                val = conv.dyn2sta_yme_bradord(yme_dyn=yd_mpsi)
            elif method["key"] == "najibi":
                val = conv.dyn2sta_yme_najib(yme_dyn=yd_mpsi)
            elif method["key"] == "fuller":
                val = conv.dyn2sta_yme_fuller(yme_dyn=yd_gpa)  # Fuller works in GPa
            elif method["key"] == "morales":
                if not np.isfinite(por[i]):
                    continue
                val = conv.dyn2sta_yme_morales(yme_dyn=yd_mpsi, porosity=float(por[i]))
                if val == -9999:  # library's low-porosity exclusion flag
                    val = np.nan
            elif method["key"] == "custom_power":
                val = conv.convert_dyn2sta_yme_custom_power_law(
                    yme_dyn=yd_mpsi, multiplier=custom_a, exponent=custom_b
                )
            else:  # custom_linear
                val = conv.dyn2sta_yme_custom_linear_law(
                    yme_dyn=yd_mpsi, slope=custom_a, intercept=custom_b
                )
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        yme_sta_native[i] = val

    # Apply the user's calibration multiplier, then normalise units.
    yme_sta_native = yme_sta_native * calibration_multiplier
    if method["input_unit"] == "GPa":
        out["YME_STA_GPA"] = yme_sta_native
        out["YME_STA_MPSI"] = yme_sta_native * GPA_TO_MPSI
    else:
        out["YME_STA_MPSI"] = yme_sta_native
        out["YME_STA_GPA"] = yme_sta_native / GPA_TO_MPSI

    # Static Poisson's ratio via geomechpy constant-multiplier law.
    pr_dyn = out["PR_DYN"].to_numpy(dtype=float)
    out["PR_STA"] = [
        conv.dyn2sta_poissons_ratio(pr_dyn=float(v), multiplier=pr_multiplier)
        if np.isfinite(v)
        else np.nan
        for v in pr_dyn
    ]

    # Negative static moduli (possible with a custom linear intercept) are unphysical.
    out.loc[out["YME_STA_GPA"] <= 0, ["YME_STA_GPA", "YME_STA_MPSI"]] = np.nan
    return out


# ---------------------------------------------------------------------------
# Rock strength (geomechpy.rock_strength)
# ---------------------------------------------------------------------------

def compute_rock_strength(
    df: pd.DataFrame,
    tstr_multiplier: float = 0.15,
    ucs_method: str = "plumb",
    fang_method: str = "lal",
    ucs_constant_mpa: float = 50.0,
    fang_constant_deg: float = 30.0,
) -> pd.DataFrame:
    """Compute UCS, tensile strength and friction angle with selectable methods.

    ucs_method:  'plumb' (geomechpy Plumb 1994 from static YME) or 'constant'.
    fang_method: 'lal' (geomechpy Lal 1999 from DTCO) or 'constant'.
    TSTR is always tstr_multiplier x UCS (geomechpy constant-multiplier law).

    Unit handling:
        UCS  : static YME passed in MPa, geomechpy returns psi -> also report MPa.
               Note: the geomechpy docstring says "Mpsi" but its coefficient
               (0.2103 psi per unit input, i.e. UCS[MPa] ~ 1.45 x E[GPa]) only
               yields physical UCS magnitudes with MPa input, so MPa is used.
        TSTR : psi -> also MPa
        FANG : from DTCO in us/ft, returned in degrees
    """
    out = df.copy()
    conv = RockStrengthPropertiesConverter

    n = len(out)
    ucs_psi = np.full(n, np.nan)
    tstr_psi = np.full(n, np.nan)
    fang = np.full(n, np.nan)

    yme_sta = out["YME_STA_GPA"].to_numpy(dtype=float) * 1000.0  # GPa -> MPa
    dtco = pd.to_numeric(out["DTCO"], errors="coerce").to_numpy(dtype=float)

    for i in range(n):
        # --- UCS ---
        if ucs_method == "constant":
            ucs_psi[i] = ucs_constant_mpa * MPA_TO_PSI
        elif np.isfinite(yme_sta[i]) and yme_sta[i] > 0:
            ucs_psi[i] = conv.convert_yme_sta_to_ucs_plumb(yme_sta=float(yme_sta[i]))
        # --- TSTR (always derived from UCS) ---
        if np.isfinite(ucs_psi[i]):
            tstr_psi[i] = conv.convert_ucs_to_tstr(ucs=float(ucs_psi[i]), multiplier=tstr_multiplier)
        # --- FANG ---
        if fang_method == "constant":
            fang[i] = fang_constant_deg
        elif np.isfinite(dtco[i]) and dtco[i] > 0:
            try:
                fang[i] = conv.convert_friction_angle_lal(dtco=float(dtco[i]))
            except (ValueError, ZeroDivisionError):  # asin domain / dt=0 guards
                pass

    out["UCS_PSI"] = ucs_psi
    out["UCS_MPA"] = ucs_psi * PSI_TO_MPA
    out["TSTR_PSI"] = tstr_psi
    out["TSTR_MPA"] = tstr_psi * PSI_TO_MPA
    out["FANG_DEG"] = fang
    return out


# ---------------------------------------------------------------------------
# QC validation
# ---------------------------------------------------------------------------

def run_qc(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate every known column against QC_RANGES (canonical units).

    Returns:
        qc_summary: one row per checked curve with counts and % in range.
        flags: per-sample flag DataFrame ('OK' / 'LOW' / 'HIGH' / 'MISSING')
               aligned with df, for color-coded display.
    """
    summary_rows = []
    flags = pd.DataFrame(index=df.index)

    for col, (lo, hi, unit) in QC_RANGES.items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        col_flags = pd.Series("OK", index=df.index)
        col_flags[values.isna()] = "MISSING"
        col_flags[values < lo] = "LOW"
        col_flags[values > hi] = "HIGH"
        flags[col] = col_flags

        n_total = len(values)
        n_missing = int(values.isna().sum())
        n_low = int((values < lo).sum())
        n_high = int((values > hi).sum())
        n_ok = n_total - n_missing - n_low - n_high
        summary_rows.append(
            {
                "Curve": col,
                "Unit": unit,
                "Valid range": f"{lo:g} - {hi:g}",
                "Samples": n_total,
                "OK": n_ok,
                "Below range": n_low,
                "Above range": n_high,
                "Missing": n_missing,
                "% in range": round(100.0 * n_ok / n_total, 1) if n_total else 0.0,
            }
        )

    return pd.DataFrame(summary_rows), flags


def qc_status(qc_summary: pd.DataFrame) -> str:
    """Overall traffic-light status: PASS / WARNING / FAIL."""
    if qc_summary.empty:
        return "FAIL"
    worst = qc_summary["% in range"].min()
    if worst >= 95.0:
        return "PASS"
    if worst >= 70.0:
        return "WARNING"
    return "FAIL"


# ---------------------------------------------------------------------------
# NEW: Overburden, pore pressure, horizontal stress & wellbore stability
# (geomechpy.overburden_stress / pore_pressure / stress_calculations /
#  wellbore_stability)
# ---------------------------------------------------------------------------

WELL_SETTINGS = ["Onshore", "Offshore"]
SHMAX_METHODS = {
    "Poroelastic (Thiercelin & Plumb, 1994) [geomechpy]": "poroelastic",
    "Shmin × anisotropy multiplier [geomechpy]": "multiplier",
}
OVB_GRADIENT_SOURCES = {
    "Constant lithostatic gradient": "constant",
    "Derived from mean RHOB log": "density",
}
DEPTH_UNITS = ["m", "ft"]


def default_stress_params() -> dict:
    """Default parameter set for compute_stress_profile (all user-adjustable)."""
    return {
        "setting": "Onshore",            # 'Onshore' | 'Offshore'
        "depth_unit": "m",               # unit of the DEPTH column ('m' | 'ft')
        "air_gap": 0.0,                  # in depth_unit
        "water_depth": 0.0,              # in depth_unit (offshore only)
        "sea_gradient_psift": 0.47,      # sea water pressure gradient [psi/ft]
        "ovb_source": "constant",        # 'constant' | 'density'
        "ovb_gradient_psift": 1.05,      # lithostatic gradient [psi/ft]
        "pp_gradient_psift": 0.47,       # formation pore pressure gradient [psi/ft]
        "shmax_method": "poroelastic",   # 'poroelastic' | 'multiplier'
        "shmax_multiplier": 1.1,         # used when shmax_method == 'multiplier'
        "biot": 1.0,                     # Biot coefficient (constant law)
        "ex": 0.0001,                    # tectonic strain term EX (poroelastic)
        "ey": 0.009,                     # tectonic strain term EY (poroelastic)
    }


def compute_stress_profile(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """NEW: full stress profile + vertical-well stability from geomechpy.

    Steps (all library calls, per depth sample; MD is assumed ~ TVD, i.e. a
    vertical well, which matches the analytical wellbore stability solution):
      1. SV  : OverburdenStressCalculation onshore/offshore gradient method.
               The lithostatic gradient is either a constant or derived from
               the mean RHOB log (mean g/cc x 0.4335 psi/ft).
      2. PP  : PorePressureCalculation onshore/offshore gradient method.
      3. SHMIN/SHMAX : poroelastic equation (static PR + static YME [Mpsi],
               Biot, tectonic strains EX/EY), or SHMAX = SHMIN x multiplier.
               q-factor and SHMAX/SHMIN ratio from the library helpers.
      4. Wellbore stability (vertical well): Mohr-Coulomb breakout pressure
               and breakdown (fracture initiation) pressure, converted to
               equivalent mud weights with the true vertical depth.

    Canonical outputs: pressures/stresses in MPa, mud weights in g/cc.
    Rows with missing prerequisites yield NaN instead of raising.
    """
    p = {**default_stress_params(), **(params or {})}
    out = df.copy()
    n = len(out)

    depth = pd.to_numeric(out["DEPTH"], errors="coerce").to_numpy(dtype=float)
    tvd_ft = depth * M_TO_FT if p["depth_unit"] == "m" else depth.copy()
    to_ft = M_TO_FT if p["depth_unit"] == "m" else 1.0
    air_gap_ft = float(p["air_gap"]) * to_ft
    water_depth_ft = float(p["water_depth"]) * to_ft

    # 1. Overburden gradient
    ovb_gradient = float(p["ovb_gradient_psift"])
    if p["ovb_source"] == "density" and "RHOB" in out.columns:
        mean_rhob = float(pd.to_numeric(out["RHOB"], errors="coerce").mean())
        if np.isfinite(mean_rhob) and mean_rhob > 0:
            ovb_gradient = mean_rhob * PSI_FT_PER_GCC

    sv_psi = np.full(n, np.nan)
    pp_psi = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(tvd_ft[i]) or tvd_ft[i] < 0:
            continue
        if p["setting"] == "Offshore":
            sv_psi[i] = OverburdenStressCalculation.calculate_overburden_stress_offshore(
                tvd=float(tvd_ft[i]),
                lithostatic_gradient=ovb_gradient,
                air_gap=air_gap_ft,
                water_depth=water_depth_ft,
                sea_water_pressure_gradient=float(p["sea_gradient_psift"]),
            )
            pp_psi[i] = PorePressureCalculation.calculate_pore_pressure_offshore(
                tvd=float(tvd_ft[i]),
                formation_pore_pressure_gradient=float(p["pp_gradient_psift"]),
                air_gap=air_gap_ft,
                water_depth=water_depth_ft,
                sea_water_pressure_gradient=float(p["sea_gradient_psift"]),
            )
        else:
            sv_psi[i] = OverburdenStressCalculation.calculate_overburden_stress_onshore(
                tvd=float(tvd_ft[i]),
                lithostatic_gradient=ovb_gradient,
                air_gap=air_gap_ft,
            )
            pp_psi[i] = PorePressureCalculation.calculate_pore_pressure_onshore(
                tvd=float(tvd_ft[i]),
                formation_pore_pressure_gradient=float(p["pp_gradient_psift"]),
                air_gap=air_gap_ft,
            )

    # 2. Horizontal stresses (poroelastic needs static PR + static YME in Mpsi)
    pr_sta = pd.to_numeric(out.get("PR_STA"), errors="coerce").to_numpy(dtype=float) if "PR_STA" in out.columns else np.full(n, np.nan)
    yme_sta_mpsi = pd.to_numeric(out.get("YME_STA_MPSI"), errors="coerce").to_numpy(dtype=float) if "YME_STA_MPSI" in out.columns else np.full(n, np.nan)

    shmin_psi = np.full(n, np.nan)
    shmax_psi = np.full(n, np.nan)
    q_factor = np.full(n, np.nan)
    sh_ratio = np.full(n, np.nan)
    for i in range(n):
        if not (np.isfinite(sv_psi[i]) and np.isfinite(pp_psi[i]) and np.isfinite(pr_sta[i]) and np.isfinite(yme_sta_mpsi[i])):
            continue
        if not (0.0 < pr_sta[i] < 0.5):
            continue
        try:
            hs = HorizontalStressesCalculation.calculate_poroelastic_horizontal_stresses(
                overburden_stress=float(sv_psi[i]),
                pore_pressure=float(pp_psi[i]),
                poisson_ratio=float(pr_sta[i]),
                youngs_modulus=float(yme_sta_mpsi[i]),
                biot_coefficient=float(p["biot"]),
                EX=float(p["ex"]),
                EY=float(p["ey"]),
            )
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        shmin_psi[i] = hs.shmin
        if p["shmax_method"] == "multiplier":
            shmax_psi[i] = HorizontalStressesCalculation.calculate_shmax_multiplier(
                shmin=float(hs.shmin), shmax_multiplier=float(p["shmax_multiplier"])
            )
        else:
            shmax_psi[i] = hs.shmax
        try:
            q_factor[i] = HorizontalStressesCalculation.calculate_stress_regime_q_factor(
                sigv=float(sv_psi[i]), shmax=float(shmax_psi[i]), shmin=float(shmin_psi[i])
            )
            sh_ratio[i] = HorizontalStressesCalculation.calculate_horizontal_stress_ratio(
                shmax=float(shmax_psi[i]), shmin=float(shmin_psi[i])
            )
        except (ValueError, ZeroDivisionError):
            pass

    # 3. Wellbore stability (vertical well, analytical)
    ucs_psi = pd.to_numeric(out.get("UCS_PSI"), errors="coerce").to_numpy(dtype=float) if "UCS_PSI" in out.columns else np.full(n, np.nan)
    tstr_psi = pd.to_numeric(out.get("TSTR_PSI"), errors="coerce").to_numpy(dtype=float) if "TSTR_PSI" in out.columns else np.full(n, np.nan)
    fang = pd.to_numeric(out.get("FANG_DEG"), errors="coerce").to_numpy(dtype=float) if "FANG_DEG" in out.columns else np.full(n, np.nan)

    pw_bo_psi = np.full(n, np.nan)   # breakout (shear failure) limit
    pw_bd_psi = np.full(n, np.nan)   # breakdown (fracture initiation) limit
    for i in range(n):
        if not (np.isfinite(shmin_psi[i]) and np.isfinite(shmax_psi[i]) and np.isfinite(pp_psi[i])):
            continue
        if np.isfinite(tstr_psi[i]):
            try:
                pw_bd_psi[i] = WellboreStabilityCalculation.calculate_breakdown_calculation_vertical_well_analytical(
                    shmax=float(shmax_psi[i]), shmin=float(shmin_psi[i]),
                    pprs=float(pp_psi[i]), tstr=float(tstr_psi[i]),
                )
            except (ValueError, ZeroDivisionError, OverflowError):
                pass
        if np.isfinite(sv_psi[i]) and np.isfinite(ucs_psi[i]) and np.isfinite(fang[i]) and np.isfinite(pr_sta[i]):
            try:
                pw_bo_psi[i] = WellboreStabilityCalculation.calculate_breakout_calculation_vertical_well_mohr_coulomb_analytical(
                    shmax=float(shmax_psi[i]), shmin=float(shmin_psi[i]),
                    pprs=float(pp_psi[i]), overburden_stress=float(sv_psi[i]),
                    ucs=float(ucs_psi[i]), fang=float(fang[i]), pr_sta=float(pr_sta[i]),
                )
            except (ValueError, ZeroDivisionError, OverflowError):
                pass

    # NEW: loss gradient = minimum principal stress among Sv, SHmax, Shmin.
    # NaN if any of the three is missing (a partial minimum would mislead).
    loss_psi = np.minimum.reduce([sv_psi, shmax_psi, shmin_psi])

    # 4. Equivalent mud weights (g/cc canonical): EMW = P / (0.4335 * TVD_ft)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = PSI_FT_PER_GCC * tvd_ft
        mw = lambda pressure_psi: np.where(denom > 0, pressure_psi / denom, np.nan)  # noqa: E731
        out["MW_PP_GCC"] = mw(pp_psi)
        out["MW_SV_GCC"] = mw(sv_psi)
        out["MW_BREAKOUT_GCC"] = mw(pw_bo_psi)
        out["MW_BREAKDOWN_GCC"] = mw(pw_bd_psi)
        out["MW_LOSS_GCC"] = mw(loss_psi)

    out["SV_MPA"] = sv_psi * PSI_TO_MPA
    out["PP_MPA"] = pp_psi * PSI_TO_MPA
    out["SHMIN_MPA"] = shmin_psi * PSI_TO_MPA
    out["SHMAX_MPA"] = shmax_psi * PSI_TO_MPA
    out["Q_FACTOR"] = q_factor
    out["SH_RATIO"] = sh_ratio
    out["PW_BREAKOUT_MPA"] = pw_bo_psi * PSI_TO_MPA
    out["PW_BREAKDOWN_MPA"] = pw_bd_psi * PSI_TO_MPA
    out["LOSS_P_MPA"] = loss_psi * PSI_TO_MPA
    return out


# ---------------------------------------------------------------------------
# NEW: Stress barrier analysis & perforation zone screening
# ---------------------------------------------------------------------------

PERF_QUALITIES = ["Good", "Moderate", "Poor"]


def analyze_stress_barriers(
    results: pd.DataFrame,
    contrast_threshold_mpa: float = 1.0,
    trend_window: int = 25,
    search_window: int = 20,
    min_zone_samples: int = 3,
) -> dict:
    """NEW: simple stress barrier analysis on the computed Shmin profile.

    Rationale: hydraulic fractures initiate where Shmin is locally LOW and
    stay contained when intervals of locally HIGH Shmin (stress barriers)
    exist above and below. The analysis therefore:

      1. Removes the depth trend from Shmin (centered rolling median over
         trend_window samples) and works with the residual stress contrast.
      2. Classifies each sample: contrast >= +threshold  -> 'Barrier',
         contrast <= -threshold -> 'Target', otherwise 'Neutral'.
      3. Rates each sample as a perforation candidate:
           Good     : Target with a Barrier within search_window samples
                      both above AND below (contained low-stress interval).
           Moderate : Target with a Barrier on one side only.
           Poor     : everything else (barriers themselves, neutral rock,
                      uncontained targets).
      4. Groups contiguous Good/Moderate samples into recommended
         perforation intervals (at least min_zone_samples thick).

    All stress values are canonical MPa; the caller converts for display.

    Returns dict with:
        detail   : per-depth DataFrame (DEPTH, SHMIN_MPA, TREND_MPA,
                   CONTRAST_MPA, CLASS, PERF_QUALITY).
        zones    : recommended perforation intervals (Top, Base, Thickness,
                   Samples, Mean Shmin, Mean contrast, Quality).
        barriers : barrier intervals (Top, Base, Thickness, Mean contrast).
    """
    if "SHMIN_MPA" not in results.columns:
        raise ValueError("Stress barrier analysis needs the stress profile — enable stress computation and re-run.")

    depth = pd.to_numeric(results["DEPTH"], errors="coerce")
    shmin = pd.to_numeric(results["SHMIN_MPA"], errors="coerce")
    valid = shmin.notna() & depth.notna()
    if int(valid.sum()) < max(10, min_zone_samples):
        raise ValueError("Not enough valid Shmin samples for a barrier analysis — check the QC report.")

    trend = shmin.rolling(int(trend_window), center=True, min_periods=1).median()
    contrast = shmin - trend

    n = len(results)
    cls = np.full(n, "N/A", dtype=object)
    t = float(contrast_threshold_mpa)
    c = contrast.to_numpy(dtype=float)
    ok = valid.to_numpy()
    cls[ok & (c >= t)] = "Barrier"
    cls[ok & (c <= -t)] = "Target"
    cls[ok & (np.abs(c) < t)] = "Neutral"

    # Barrier presence within search_window samples above/below each sample.
    is_barrier = (cls == "Barrier").astype(int)
    above = np.zeros(n, dtype=bool)
    below = np.zeros(n, dtype=bool)
    w = int(search_window)
    for i in range(n):
        above[i] = is_barrier[max(0, i - w):i].any()
        below[i] = is_barrier[i + 1:i + 1 + w].any()

    quality = np.full(n, "Poor", dtype=object)
    target = cls == "Target"
    quality[target & above & below] = "Good"
    quality[target & (above ^ below)] = "Moderate"
    quality[~ok] = "N/A"

    detail = pd.DataFrame(
        {
            "DEPTH": depth,
            "SHMIN_MPA": shmin,
            "TREND_MPA": trend,
            "CONTRAST_MPA": contrast,
            "CLASS": cls,
            "PERF_QUALITY": quality,
        }
    )

    def _intervals(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
        """Contiguous [start, end] index runs where mask is True."""
        runs, start = [], None
        for i in range(n):
            if mask[i] and start is None:
                start = i
            elif not mask[i] and start is not None:
                if i - start >= min_len:
                    runs.append((start, i - 1))
                start = None
        if start is not None and n - start >= min_len:
            runs.append((start, n - 1))
        return runs

    zone_rows = []
    for grade in ("Good", "Moderate"):
        for s, e in _intervals(quality == grade, int(min_zone_samples)):
            zone_rows.append(
                {
                    "Top": float(depth.iloc[s]),
                    "Base": float(depth.iloc[e]),
                    "Thickness": float(depth.iloc[e] - depth.iloc[s]),
                    "Samples": e - s + 1,
                    "Mean Shmin (MPa)": float(shmin.iloc[s:e + 1].mean()),
                    "Mean contrast (MPa)": float(contrast.iloc[s:e + 1].mean()),
                    "Quality": grade,
                }
            )
    zones = pd.DataFrame(zone_rows).sort_values("Top").reset_index(drop=True) if zone_rows else pd.DataFrame(
        columns=["Top", "Base", "Thickness", "Samples", "Mean Shmin (MPa)", "Mean contrast (MPa)", "Quality"]
    )

    barrier_rows = [
        {
            "Top": float(depth.iloc[s]),
            "Base": float(depth.iloc[e]),
            "Thickness": float(depth.iloc[e] - depth.iloc[s]),
            "Mean contrast (MPa)": float(contrast.iloc[s:e + 1].mean()),
        }
        for s, e in _intervals(cls == "Barrier", 2)
    ]
    barriers = pd.DataFrame(barrier_rows) if barrier_rows else pd.DataFrame(
        columns=["Top", "Base", "Thickness", "Mean contrast (MPa)"]
    )

    return {"detail": detail, "zones": zones, "barriers": barriers}


# ---------------------------------------------------------------------------
# Full pipeline + export
# ---------------------------------------------------------------------------

def run_full_workflow(
    data: pd.DataFrame,
    column_map: dict[str, str],
    method_label: str,
    calibration_multiplier: float,
    pr_multiplier: float,
    tstr_multiplier: float,
    custom_a: float = 0.5,
    custom_b: float = 1.0,
    unit_system: str = OILFIELD,
    ucs_method: str = "plumb",
    fang_method: str = "lal",
    ucs_constant_mpa: float = 50.0,
    fang_constant_deg: float = 30.0,
    stress_params: dict | None = None,
) -> pd.DataFrame:
    """Rename mapped columns to standard mnemonics, convert the input to
    canonical units and run the geomechpy modules: dynamic -> static ->
    rock strength (selectable methods) -> optionally the full stress
    profile + wellbore stability when stress_params is provided."""
    missing = [c for c in REQUIRED_CURVES if not column_map.get(c)]
    if missing:
        raise ValueError(f"Missing required column mapping(s): {', '.join(missing)}")

    rename = {src: curve for curve, src in column_map.items() if src}
    df = data[[c for c in rename]].rename(columns=rename).copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col].isin(NULL_SENTINELS), col] = np.nan  # safety net
    df = df.sort_values("DEPTH").reset_index(drop=True)

    df = normalize_input_units(df, unit_system)

    df = compute_dynamic_properties(df)
    df = compute_static_properties(
        df,
        method_label=method_label,
        calibration_multiplier=calibration_multiplier,
        pr_multiplier=pr_multiplier,
        custom_a=custom_a,
        custom_b=custom_b,
    )
    df = compute_rock_strength(
        df,
        tstr_multiplier=tstr_multiplier,
        ucs_method=ucs_method,
        fang_method=fang_method,
        ucs_constant_mpa=ucs_constant_mpa,
        fang_constant_deg=fang_constant_deg,
    )
    if stress_params is not None:
        df = compute_stress_profile(df, stress_params)
    return df


def results_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize results for the Streamlit download button."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, float_format="%.4f")
    return buffer.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Sensitivity analysis (Tornado plot)
# ---------------------------------------------------------------------------

# Canonical result columns selectable as tornado targets (display order).
TORNADO_TARGETS = [
    "YME_STA_GPA",
    "UCS_MPA",
    "TSTR_MPA",
    "PR_STA",
    "YME_DYN_GPA",
    "PR_DYN",
    "FANG_DEG",
]
# NEW: additional targets available when the stress profile is computed.
TORNADO_STRESS_TARGETS = [
    "SHMIN_MPA",
    "SHMAX_MPA",
    "MW_BREAKOUT_GCC",
    "MW_BREAKDOWN_GCC",
]

# Input curves perturbed one at a time, plus the static YME calibration multiplier.
TORNADO_INPUT_CURVES = ["GR", "RHOB", "DTCO", "DTSM", "POROSITY"]
STATIC_MULT_PARAM = "Static YME multiplier"
TORNADO_PARAMS = TORNADO_INPUT_CURVES + [STATIC_MULT_PARAM]


def run_tornado_analysis(
    data: pd.DataFrame,
    column_map: dict[str, str],
    target_output: str,
    variation_pct: float = 10.0,
    *,
    method_label: str,
    calibration_multiplier: float = 1.0,
    pr_multiplier: float = 1.0,
    tstr_multiplier: float = 0.15,
    custom_a: float = 0.5,
    custom_b: float = 1.0,
    unit_system: str = OILFIELD,
    ucs_method: str = "plumb",
    fang_method: str = "lal",
    ucs_constant_mpa: float = 50.0,
    fang_constant_deg: float = 30.0,
    stress_params: dict | None = None,
) -> tuple[pd.DataFrame, float, list[str]]:
    """One-at-a-time sensitivity of a target output to the main inputs.

    The current data is the base case. Each parameter in TORNADO_PARAMS is
    varied by ±variation_pct while everything else is held fixed, and the
    full workflow is recomputed. The compared statistic is the depth-averaged
    (NaN-ignoring mean) value of the target column, in canonical units.

    Returns:
        tornado: DataFrame with one row per varied parameter:
                 Parameter, low, high (target means at -/+ variation),
                 pct_low, pct_high (% change vs base), swing (|high - low|),
                 sorted by swing descending.
        base_value: base-case target mean (canonical units).
        skipped: parameters that could not be varied (unmapped column or
                 failed recomputation).
    """
    if target_output not in DISPLAY_SPEC:
        raise ValueError(f"Unknown target output: {target_output}")

    workflow_kwargs = dict(
        column_map=column_map,
        method_label=method_label,
        pr_multiplier=pr_multiplier,
        tstr_multiplier=tstr_multiplier,
        custom_a=custom_a,
        custom_b=custom_b,
        unit_system=unit_system,
        ucs_method=ucs_method,
        fang_method=fang_method,
        ucs_constant_mpa=ucs_constant_mpa,
        fang_constant_deg=fang_constant_deg,
        stress_params=stress_params,
    )

    def _target_mean(frame: pd.DataFrame, cal_multiplier: float) -> float:
        res = run_full_workflow(frame, calibration_multiplier=cal_multiplier, **workflow_kwargs)
        if target_output not in res.columns:
            raise ValueError(f"Target '{target_output}' was not produced by the workflow.")
        return float(np.nanmean(pd.to_numeric(res[target_output], errors="coerce")))

    base_value = _target_mean(data, calibration_multiplier)
    if not np.isfinite(base_value):
        raise ValueError(
            "The base case produced no valid values for the selected target — "
            "check the column mapping and QC report."
        )

    frac = float(variation_pct) / 100.0
    rows: list[dict] = []
    skipped: list[str] = []

    for param in TORNADO_PARAMS:
        try:
            if param == STATIC_MULT_PARAM:
                low = _target_mean(data, calibration_multiplier * (1.0 - frac))
                high = _target_mean(data, calibration_multiplier * (1.0 + frac))
            else:
                src = column_map.get(param)
                if not src or src not in data.columns:
                    skipped.append(param)
                    continue
                values = pd.to_numeric(data[src], errors="coerce")
                lo_frame = data.copy()
                lo_frame[src] = values * (1.0 - frac)
                hi_frame = data.copy()
                hi_frame[src] = values * (1.0 + frac)
                low = _target_mean(lo_frame, calibration_multiplier)
                high = _target_mean(hi_frame, calibration_multiplier)
        except (ValueError, ZeroDivisionError, OverflowError):
            skipped.append(param)
            continue

        rows.append(
            {
                "Parameter": param,
                "low": low,
                "high": high,
                "pct_low": 100.0 * (low - base_value) / base_value if base_value else np.nan,
                "pct_high": 100.0 * (high - base_value) / base_value if base_value else np.nan,
            }
        )

    tornado = pd.DataFrame(rows)
    if tornado.empty:
        raise ValueError("No input parameters could be varied — check the column mapping.")
    tornado["swing"] = (tornado["high"] - tornado["low"]).abs()
    tornado = tornado.sort_values("swing", ascending=False).reset_index(drop=True)
    return tornado, base_value, skipped


def generate_tornado_plot(
    df: pd.DataFrame,
    column_map: dict[str, str],
    target_output: str,
    variation_pct: float = 10.0,
    **settings,
) -> tuple[go.Figure, pd.DataFrame, float, list[str]]:
    """Run the tornado analysis and build the Plotly figure.

    settings are forwarded to run_tornado_analysis (method_label,
    calibration_multiplier, pr_multiplier, tstr_multiplier, custom_a,
    custom_b, unit_system). Values are converted to the selected unit
    system for display.

    Returns:
        fig: horizontal-bar tornado chart, largest swing on top.
        table: per-parameter results in display units (for st.dataframe).
        base_display: base-case target value in display units.
        skipped: parameters that could not be varied.
    """
    unit_system = settings.get("unit_system", OILFIELD)
    tornado, base_value, skipped = run_tornado_analysis(
        df, column_map, target_output, variation_pct, **settings
    )

    _, _, factor = _spec(target_output, unit_system)
    target_label = display_name(target_output, unit_system)
    base_display = base_value * factor

    t = tornado.copy()
    t["low"] = t["low"] * factor
    t["high"] = t["high"] * factor
    t["swing"] = t["swing"] * factor
    t["delta_low"] = t["low"] - base_display
    t["delta_high"] = t["high"] - base_display

    # Plotly draws category bars bottom-up: ascending swing puts the biggest on top.
    plot = t.sort_values("swing", ascending=True)
    pct = f"{variation_pct:g}"

    fig = go.Figure()
    fig.add_bar(
        y=plot["Parameter"],
        x=plot["delta_low"],
        base=base_display,
        orientation="h",
        name=f"Input -{pct}%",
        marker_color="#d95f02",
        customdata=np.stack([plot["low"], plot["pct_low"]], axis=-1),
        hovertemplate=(
            "%{y} -" + pct + "%<br>"
            + target_label + ": %{customdata[0]:.3f} (%{customdata[1]:+.2f}% vs base)"
            "<extra></extra>"
        ),
    )
    fig.add_bar(
        y=plot["Parameter"],
        x=plot["delta_high"],
        base=base_display,
        orientation="h",
        name=f"Input +{pct}%",
        marker_color="#1f77b4",
        customdata=np.stack([plot["high"], plot["pct_high"]], axis=-1),
        hovertemplate=(
            "%{y} +" + pct + "%<br>"
            + target_label + ": %{customdata[0]:.3f} (%{customdata[1]:+.2f}% vs base)"
            "<extra></extra>"
        ),
    )
    fig.add_vline(
        x=base_display,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"base = {base_display:.3f}",
        annotation_position="top",
    )
    fig.update_layout(
        barmode="overlay",
        title=f"Tornado plot — sensitivity of {target_label} to ±{pct}% input variation",
        xaxis_title=f"Depth-averaged {target_label}",
        yaxis_title="Varied input parameter",
        height=max(360, 90 * len(plot) + 140),
        legend=dict(orientation="h", yanchor="bottom", y=1.04),
        margin=dict(t=110, b=40),
    )

    table = t[["Parameter", "low", "high", "pct_low", "pct_high", "swing"]].rename(
        columns={
            "low": f"Target @ -{pct}%",
            "high": f"Target @ +{pct}%",
            "pct_low": "Δ% @ low",
            "pct_high": "Δ% @ high",
            "swing": f"Swing [{display_unit(target_output, unit_system)}]",
        }
    )
    return fig, table, base_display, skipped
