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
from geomechpy.rock_strength import RockStrengthPropertiesConverter
from geomechpy.static_elastic_properties import StaticElasticPropertiesConverter

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
    "DEPTH": ("MD", ("m", 1.0), ("m", 1.0)),
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

def compute_rock_strength(df: pd.DataFrame, tstr_multiplier: float = 0.15) -> pd.DataFrame:
    """Compute UCS (Plumb), tensile strength and friction angle (Lal).

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
        if np.isfinite(yme_sta[i]) and yme_sta[i] > 0:
            ucs_psi[i] = conv.convert_yme_sta_to_ucs_plumb(yme_sta=float(yme_sta[i]))
            tstr_psi[i] = conv.convert_ucs_to_tstr(ucs=float(ucs_psi[i]), multiplier=tstr_multiplier)
        if np.isfinite(dtco[i]) and dtco[i] > 0:
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
) -> pd.DataFrame:
    """Rename mapped columns to standard mnemonics, convert the input to
    canonical units and run all three geomechpy modules."""
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
    df = compute_rock_strength(df, tstr_multiplier=tstr_multiplier)
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
