"""
Quick MEM Calculator - calculation engine.

All geomechanics calculations are delegated to the geomechpy library:
    - geomechpy.elastic_properties        -> dynamic elastic properties
    - geomechpy.static_elastic_properties -> dynamic-to-static conversion
    - geomechpy.rock_strength             -> UCS / TSTR / friction angle

This module only handles unit conversions, dataframe plumbing, QC flagging
and sample-data generation for the Streamlit front end.
"""

from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd

from geomechpy.elastic_properties import ElasticPropertiesConverter
from geomechpy.rock_strength import RockStrengthPropertiesConverter
from geomechpy.static_elastic_properties import StaticElasticPropertiesConverter

# ---------------------------------------------------------------------------
# Constants & unit conversion
# ---------------------------------------------------------------------------

FT_TO_M_US = 304800.0          # us/ft slowness -> m/s velocity: v = 304800 / dt
PA_TO_GPA = 1.0e-9             # Pascal -> GigaPascal
PA_TO_PSI = 1.0 / 6894.757293  # Pascal -> psi
PA_TO_MPSI = PA_TO_PSI * 1e-6  # Pascal -> Mega-psi
PSI_TO_MPA = 6894.757293e-6    # psi -> MPa
GCC_TO_KGM3 = 1000.0           # g/cc -> kg/m3

# Curves the app can map. POROSITY is optional unless the Morales method is used.
REQUIRED_CURVES = ["DEPTH", "GR", "RHOB", "DTCO", "DTSM"]
OPTIONAL_CURVES = ["POROSITY"]
ALL_CURVES = REQUIRED_CURVES + OPTIONAL_CURVES

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

# QC validation ranges: column -> (min, max, unit) using typical log/MEM limits.
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(uploaded_file) -> pd.DataFrame:
    """Read an uploaded CSV or Excel file into a DataFrame.

    Raises ValueError with a user-friendly message on failure.
    """
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv") or name.endswith(".txt"):
            df = pd.read_csv(uploaded_file)
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
    return df


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


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

def generate_sample_data(n_points: int = 201, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic sand/shale well-log interval (2500-3000 m MD).

    Values are geologically plausible so all downstream calculations
    produce sensible magnitudes.
    """
    rng = np.random.default_rng(seed)
    depth = np.linspace(2500.0, 3000.0, n_points)

    # Smooth sand/shale alternation driver (0 = clean sand, 1 = shale)
    vsh = 0.5 + 0.35 * np.sin(depth / 18.0) + 0.15 * np.sin(depth / 61.0)
    vsh = np.clip(vsh + rng.normal(0, 0.05, n_points), 0.02, 0.98)

    compaction = (depth - 2500.0) / 500.0  # 0 -> 1 over the interval

    gr = 25.0 + 110.0 * vsh + rng.normal(0, 4.0, n_points)
    rhob = 2.30 + 0.25 * compaction + 0.12 * vsh + rng.normal(0, 0.02, n_points)
    dtco = 95.0 - 25.0 * compaction + 18.0 * vsh + rng.normal(0, 1.5, n_points)
    dtsm = dtco * (1.65 + 0.25 * vsh) + rng.normal(0, 3.0, n_points)
    porosity = np.clip(0.28 - 0.12 * compaction - 0.08 * vsh + rng.normal(0, 0.01, n_points), 0.03, 0.35)

    return pd.DataFrame(
        {
            "DEPTH": np.round(depth, 2),
            "GR": np.round(gr, 2),
            "RHOB": np.round(rhob, 3),
            "DTCO": np.round(dtco, 2),
            "DTSM": np.round(dtsm, 2),
            "POROSITY": np.round(porosity, 3),
        }
    )


# ---------------------------------------------------------------------------
# Dynamic elastic properties (geomechpy.elastic_properties)
# ---------------------------------------------------------------------------

def compute_dynamic_properties(df: pd.DataFrame) -> pd.DataFrame:
    """Compute dynamic elastic properties from DTCO/DTSM/RHOB.

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
    out["YME_DYN_MPSI"] = out["YME_DYN_GPA"] / (PA_TO_GPA / PA_TO_MPSI)
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
        out["YME_STA_MPSI"] = yme_sta_native * (PA_TO_MPSI / PA_TO_GPA)
    else:
        out["YME_STA_MPSI"] = yme_sta_native
        out["YME_STA_GPA"] = yme_sta_native * (PA_TO_GPA / PA_TO_MPSI)

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
    """Validate every known column against QC_RANGES.

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
) -> pd.DataFrame:
    """Rename mapped columns to standard mnemonics and run all three modules."""
    missing = [c for c in REQUIRED_CURVES if not column_map.get(c)]
    if missing:
        raise ValueError(f"Missing required column mapping(s): {', '.join(missing)}")

    rename = {src: curve for curve, src in column_map.items() if src}
    df = data[[c for c in rename]].rename(columns=rename).copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("DEPTH").reset_index(drop=True)

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
