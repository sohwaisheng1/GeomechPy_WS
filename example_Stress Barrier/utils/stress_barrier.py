"""
Stress Barrier Analysis & Perforation Planner — calculation engine.

All geomechanics calculations are delegated to the geomechpy library:
    - geomechpy.stress_calculations  -> horizontal stresses (Shmin / SHmax)
    - geomechpy.overburden_stress    -> Sv from a lithostatic gradient (fallback)
    - geomechpy.pore_pressure        -> Pp from a pressure gradient (fallback)

This module handles the data plumbing (file loading, unit conversion, sample
data), the GR-based lithology flag, the stress-barrier / stress-contrast
analysis and the perforation-zone screening for the Streamlit front end.

Canonical internal units (everything is converted to these before use):
    depth      -> the input depth unit (m or ft), kept as-is
    stresses   -> psi (Sv, Pp, Shmin, SHmax; the app always displays psi)
    YME        -> Mpsi (the unit the poroelastic equation in geomechpy expects)
    PR, strain -> unitless
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from geomechpy.overburden_stress import OverburdenStressCalculation
from geomechpy.pore_pressure import PorePressureCalculation
from geomechpy.stress_calculations import HorizontalStressesCalculation

# ---------------------------------------------------------------------------
# Unit conversion constants
# ---------------------------------------------------------------------------

PSI_TO_MPA = 6894.757293e-6      # psi -> MPa
MPA_TO_PSI = 1.0 / PSI_TO_MPA    # MPa -> psi
PSI_TO_MPSI = 1.0e-6             # psi -> Mega-psi
MPSI_TO_PSI = 1.0e6              # Mega-psi -> psi
GPA_TO_PSI = 145037.737797      # GPa -> psi
GPA_TO_MPSI = GPA_TO_PSI * 1e-6  # GPa -> Mpsi (~0.145)
M_PER_FT = 0.3048                # metres per foot
M_TO_FT = 1.0 / M_PER_FT         # metres -> feet

# Common well-log null sentinels replaced with NaN on load.
NULL_SENTINELS = [-999.0, -999.25, -9999.0, -9999.25, -99999.0, 9999.0]

# ---------------------------------------------------------------------------
# Curves the app understands
# ---------------------------------------------------------------------------

# DEPTH is always required. YME + PR are required to compute horizontal
# stresses. GR is needed for the lithology flag. PP and SV may either come
# from a log column or be derived from a gradient (see StressConfig).
REQUIRED_CURVES = ["DEPTH", "YME", "PR"]
OPTIONAL_CURVES = ["GR", "PP", "SV"]
ALL_CURVES = REQUIRED_CURVES + OPTIONAL_CURVES

CURVE_LABELS = {
    "DEPTH": "Depth (MD)",
    "GR": "Gamma Ray (GR)",
    "PP": "Pore Pressure (PP)",
    "SV": "Overburden Stress (Sv / OVB)",
    "YME": "Young's Modulus (YME)",
    "PR": "Poisson's Ratio (PR)",
}

# Input-unit options offered in the UI, with the factor that converts a value
# in that unit to the canonical unit used internally.
YME_INPUT_UNITS = {
    "Mpsi": 1.0,                 # canonical
    "psi": PSI_TO_MPSI,
    "GPa": GPA_TO_MPSI,
}
PRESSURE_INPUT_UNITS = {
    "psi": 1.0,                  # canonical
    "MPa": MPA_TO_PSI,
    "bar": 14.5037738,
}
DEPTH_UNITS = ["m", "ft"]

# Lithology flag: reservoir sand (0) vs non-reservoir shale (1).
LITHO_NAME_BY_CODE = {0: "Reservoir (Sand)", 1: "Non-reservoir (Shale)"}
LITHO_COLORS = {0: "#f4d03f", 1: "#7f8c8d", -1: "#ecf0f1"}
DEFAULT_GR_CUTOFF = 75.0

# Perforation quality grades and their display colours.
PERF_QUALITIES = ["Good", "Moderate", "Poor"]
PERF_COLORS = {
    "Good": "#2ecc71",
    "Moderate": "#f39c12",
    "Poor": "#e74c3c",
    "N/A": "#ecf0f1",
}

SHMAX_METHODS = {
    "Poroelastic (Thiercelin & Plumb, 1994) [geomechpy]": "poroelastic",
    "Shmin × anisotropy multiplier [geomechpy]": "multiplier",
}
PP_SV_SOURCES = {
    "From log column": "column",
    "From gradient (geomechpy)": "gradient",
}


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------

def _drop_unit_rows(df: pd.DataFrame, max_rows: int = 3) -> tuple[pd.DataFrame, int]:
    """Drop leading rows that hold unit strings instead of data.

    A leading row is treated as a unit row when it is non-numeric in at least
    half of the columns whose remaining values are mostly numeric.
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
    """Read an uploaded CSV, Excel or LAS file and clean it up.

    Returns (dataframe, informational messages).
    Raises ValueError with a user-friendly message on failure.
    """
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".las"):
            import lasio  # optional dependency; only needed for LAS input
            try:
                uploaded_file.seek(0)
            except Exception:
                pass
            raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
            text = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else raw
            las = lasio.read(io.StringIO(text))
            df = las.df().reset_index()  # depth curve is the index -> expose as a column
        elif name.endswith((".csv", ".txt")):
            df = pd.read_csv(uploaded_file, skip_blank_lines=True)
            if df.shape[1] == 1:  # non-comma delimiter: re-sniff
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=None, engine="python", skip_blank_lines=True)
        elif name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file type. Please upload a .las, .csv, .xls or .xlsx file.")
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
    """Best-effort index of the column matching a curve mnemonic (selectbox default).

    Returns 0 ('-- not mapped --') when nothing matches.
    """
    aliases = {
        "DEPTH": ["depth", "dept", "md", "tvd"],
        "GR": ["gr", "gamma", "gapi", "cgr", "sgr"],
        "PP": ["pp", "pore", "porepressure", "pore_pressure", "ppg", "pnorm"],
        "SV": ["sv", "ovb", "obg", "overburden", "vertical", "sigv", "sigmav"],
        "YME": ["yme", "ym", "young", "youngs", "e_sta", "estat", "emod", "e"],
        "PR": ["pr", "poisson", "nu", "poissons", "pois"],
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

def generate_sample_data(n_points: int = 241, seed: int = 7) -> pd.DataFrame:
    """Generate a synthetic sand/shale well interval (3000-3600 m MD).

    Produces a ready-to-run dataset with the exact columns this app expects:
    MD, GR, PP, SV, YME, PR. Values are geologically plausible so the stress
    calculation and the barrier / perforation screening return sensible
    results out of the box. Shale sections are given a higher Poisson's ratio
    (and thus a higher Shmin) than the sand, creating clear stress barriers.
    """
    rng = np.random.default_rng(seed)
    depth = np.linspace(3000.0, 3600.0, n_points)

    # Smooth sand/shale alternation driver (0 = clean sand, 1 = shale).
    vsh = 0.5 + 0.38 * np.sin(depth / 22.0) + 0.12 * np.sin(depth / 70.0)
    vsh = np.clip(vsh + rng.normal(0, 0.05, n_points), 0.02, 0.98)

    tvd_ft = depth * M_TO_FT
    gr = 25.0 + 110.0 * vsh + rng.normal(0, 4.0, n_points)                 # gAPI
    pp = 0.465 * tvd_ft + rng.normal(0, 30.0, n_points)                    # psi (~hydrostatic)
    sv = 1.02 * tvd_ft + rng.normal(0, 40.0, n_points)                     # psi (lithostatic)
    # Shale is softer + more ductile: lower YME, higher PR -> higher Shmin.
    yme = (4.2 - 2.4 * vsh) + rng.normal(0, 0.15, n_points)                # Mpsi
    yme = np.clip(yme, 0.8, 6.0)
    pr = (0.20 + 0.15 * vsh) + rng.normal(0, 0.01, n_points)               # unitless
    pr = np.clip(pr, 0.12, 0.42)

    return pd.DataFrame(
        {
            "MD": np.round(depth, 2),
            "GR": np.round(gr, 2),
            "PP": np.round(pp, 1),
            "SV": np.round(sv, 1),
            "YME": np.round(yme, 3),
            "PR": np.round(pr, 3),
        }
    )


def sample_csv_bytes() -> bytes:
    """Example CSV for the 'Download Example File' button."""
    buffer = io.StringIO()
    generate_sample_data().to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Lithology flag (GR cutoff)
# ---------------------------------------------------------------------------

def compute_lithology_flag(df: pd.DataFrame, gr_cutoff: float = DEFAULT_GR_CUTOFF) -> pd.DataFrame:
    """Add a LITHO_CODE column classifying each sample from GR by one cutoff.

    GR <  gr_cutoff -> reservoir sand (code 0)
    GR >= gr_cutoff -> non-reservoir shale (code 1)
    Missing GR      -> NaN
    """
    out = df.copy()
    gr = (
        pd.to_numeric(out["GR"], errors="coerce").to_numpy(dtype=float)
        if "GR" in out.columns
        else np.full(len(out), np.nan)
    )
    out["LITHO_CODE"] = np.where(np.isfinite(gr), np.where(gr < float(gr_cutoff), 0.0, 1.0), np.nan)
    return out


def lithology_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Per-lithology sample counts and fraction, for the summary display."""
    if "LITHO_CODE" not in df.columns:
        return pd.DataFrame(columns=["Lithology", "Code", "Samples", "Fraction %"])
    codes = pd.to_numeric(df["LITHO_CODE"], errors="coerce")
    total = int(codes.notna().sum())
    rows = []
    for code, name in LITHO_NAME_BY_CODE.items():
        cnt = int((codes == code).sum())
        rows.append({"Lithology": name, "Code": code, "Samples": cnt,
                     "Fraction %": round(100.0 * cnt / total, 1) if total else 0.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stress configuration & horizontal-stress calculation
# ---------------------------------------------------------------------------

def default_stress_config() -> dict:
    """Default configuration for run_stress_workflow (all user-adjustable)."""
    return {
        # input units
        "yme_unit": "Mpsi",
        "pressure_unit": "psi",
        "depth_unit": "m",
        # pore pressure / overburden source ('column' | 'gradient')
        "pp_source": "column",
        "sv_source": "column",
        "pp_gradient_psift": 0.465,
        "ovb_gradient_psift": 1.02,
        "setting": "Onshore",
        "air_gap": 0.0,
        # horizontal strains (manual) and poroelastic parameters
        "eps_h": 0.0001,     # minimum horizontal strain  -> EX in geomechpy
        "eps_H": 0.0009,     # maximum horizontal strain  -> EY in geomechpy
        "biot": 1.0,
        "shmax_method": "poroelastic",
        "shmax_multiplier": 1.1,
        "gr_cutoff": DEFAULT_GR_CUTOFF,
    }


def _pp_sv_from_gradient(tvd_ft: np.ndarray, gradient_psift: float, air_gap_ft: float,
                         kind: str) -> np.ndarray:
    """Compute a pressure profile from a gradient using geomechpy (onshore)."""
    n = len(tvd_ft)
    out = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(tvd_ft[i]) or tvd_ft[i] < 0:
            continue
        if kind == "sv":
            out[i] = OverburdenStressCalculation.calculate_overburden_stress_onshore(
                tvd=float(tvd_ft[i]), lithostatic_gradient=gradient_psift, air_gap=air_gap_ft,
            )
        else:  # pp
            out[i] = PorePressureCalculation.calculate_pore_pressure_onshore(
                tvd=float(tvd_ft[i]), formation_pore_pressure_gradient=gradient_psift, air_gap=air_gap_ft,
            )
    return out


def run_stress_workflow(data: pd.DataFrame, column_map: dict[str, str], config: dict) -> pd.DataFrame:
    """Build the canonical results frame: horizontal stresses + lithology flag.

    Steps:
      1. Rename mapped columns to standard mnemonics; convert YME to Mpsi and
         PP/SV to psi (or derive PP/SV from a gradient via geomechpy).
      2. Flag lithology from GR (reservoir sand vs non-reservoir shale).
      3. Compute Shmin/SHmax per depth with the geomechpy poroelastic equation
         using the user's YME, PR, Sv, Pp and the manually-defined horizontal
         strains (eps_h -> EX, eps_H -> EY). SHmax can instead be Shmin × a
         multiplier. q-factor and the SHmax/Shmin ratio come from the library.

    Canonical outputs (all stresses in psi): SV_PSI, PP_PSI, SHMIN_PSI,
    SHMAX_PSI, Q_FACTOR, SH_RATIO, plus DEPTH, GR, YME_MPSI, PR, LITHO_CODE.
    Rows with missing prerequisites yield NaN instead of raising.
    """
    cfg = {**default_stress_config(), **(config or {})}

    missing = [c for c in REQUIRED_CURVES if not column_map.get(c)]
    if missing:
        raise ValueError(
            "Missing required column mapping(s): "
            + ", ".join(f"{c} ({CURVE_LABELS[c]})" for c in missing)
        )

    rename = {src: curve for curve, src in column_map.items() if src}
    df = data[list(rename)].rename(columns=rename).copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col].isin(NULL_SENTINELS), col] = np.nan
    df = df.sort_values("DEPTH").reset_index(drop=True)
    n = len(df)

    # --- unit conversion to canonical ---
    yme_factor = YME_INPUT_UNITS.get(cfg["yme_unit"], 1.0)
    p_factor = PRESSURE_INPUT_UNITS.get(cfg["pressure_unit"], 1.0)

    out = pd.DataFrame(index=df.index)
    out["DEPTH"] = pd.to_numeric(df["DEPTH"], errors="coerce")
    out["GR"] = pd.to_numeric(df["GR"], errors="coerce") if "GR" in df.columns else np.nan
    out["YME_MPSI"] = pd.to_numeric(df["YME"], errors="coerce") * yme_factor
    out["PR"] = pd.to_numeric(df["PR"], errors="coerce")

    depth = out["DEPTH"].to_numpy(dtype=float)
    tvd_ft = depth * M_TO_FT if cfg["depth_unit"] == "m" else depth.copy()
    air_gap_ft = float(cfg["air_gap"]) * (M_TO_FT if cfg["depth_unit"] == "m" else 1.0)

    # --- overburden (Sv) ---
    if cfg["sv_source"] == "gradient" or "SV" not in df.columns or df["SV"].isna().all():
        sv_psi = _pp_sv_from_gradient(tvd_ft, float(cfg["ovb_gradient_psift"]), air_gap_ft, "sv")
    else:
        sv_psi = (pd.to_numeric(df["SV"], errors="coerce") * p_factor).to_numpy(dtype=float)

    # --- pore pressure (Pp) ---
    if cfg["pp_source"] == "gradient" or "PP" not in df.columns or df["PP"].isna().all():
        pp_psi = _pp_sv_from_gradient(tvd_ft, float(cfg["pp_gradient_psift"]), air_gap_ft, "pp")
    else:
        pp_psi = (pd.to_numeric(df["PP"], errors="coerce") * p_factor).to_numpy(dtype=float)

    out["SV_PSI"] = sv_psi
    out["PP_PSI"] = pp_psi

    # --- lithology flag ---
    out = compute_lithology_flag(out, cfg["gr_cutoff"])

    # --- horizontal stresses (poroelastic, per sample) ---
    pr = out["PR"].to_numpy(dtype=float)
    yme = out["YME_MPSI"].to_numpy(dtype=float)
    shmin = np.full(n, np.nan)
    shmax = np.full(n, np.nan)
    q_factor = np.full(n, np.nan)
    sh_ratio = np.full(n, np.nan)

    for i in range(n):
        if not (np.isfinite(sv_psi[i]) and np.isfinite(pp_psi[i]) and np.isfinite(pr[i]) and np.isfinite(yme[i])):
            continue
        if not (0.0 < pr[i] < 0.5) or yme[i] <= 0:
            continue
        try:
            hs = HorizontalStressesCalculation.calculate_poroelastic_horizontal_stresses(
                overburden_stress=float(sv_psi[i]),
                pore_pressure=float(pp_psi[i]),
                poisson_ratio=float(pr[i]),
                youngs_modulus=float(yme[i]),
                biot_coefficient=float(cfg["biot"]),
                EX=float(cfg["eps_h"]),   # minimum horizontal strain
                EY=float(cfg["eps_H"]),   # maximum horizontal strain
            )
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        shmin[i] = hs.shmin
        if cfg["shmax_method"] == "multiplier":
            shmax[i] = HorizontalStressesCalculation.calculate_shmax_multiplier(
                shmin=float(hs.shmin), shmax_multiplier=float(cfg["shmax_multiplier"])
            )
        else:
            shmax[i] = hs.shmax
        try:
            q_factor[i] = HorizontalStressesCalculation.calculate_stress_regime_q_factor(
                sigv=float(sv_psi[i]), shmax=float(shmax[i]), shmin=float(shmin[i])
            )
            sh_ratio[i] = HorizontalStressesCalculation.calculate_horizontal_stress_ratio(
                shmax=float(shmax[i]), shmin=float(shmin[i])
            )
        except (ValueError, ZeroDivisionError):
            pass

    out["SHMIN_PSI"] = shmin
    out["SHMAX_PSI"] = shmax
    out["Q_FACTOR"] = q_factor
    out["SH_RATIO"] = sh_ratio
    return out


# ---------------------------------------------------------------------------
# Stress barrier analysis & perforation zone screening
# ---------------------------------------------------------------------------

def _lithology_runs(codes: np.ndarray) -> list[tuple[int, int, float]]:
    """Contiguous runs of equal lithology code -> list of (start, end, code)."""
    n = len(codes)
    runs: list[tuple[int, int, float]] = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and (
            codes[j + 1] == codes[i] or (np.isnan(codes[j + 1]) and np.isnan(codes[i]))
        ):
            j += 1
        runs.append((i, j, codes[i]))
        i = j + 1
    return runs


def analyze_stress_barriers(
    results: pd.DataFrame,
    contrast_threshold_psi: float = 300.0,
    trend_window: int = 25,
    min_zone_thickness: float = 5.0,
) -> dict:
    """Stress-barrier analysis and perforation-zone screening.

    Rationale: a hydraulic fracture placed in a reservoir (sand) stays
    contained when it is bounded by higher-stress non-reservoir (shale)
    intervals — the stress barriers. This routine therefore:

      1. Computes a per-sample **stress contrast** = Shmin minus its depth
         trend (centred rolling median over trend_window samples), used as the
         contrast curve on the plot.
      2. Splits the log into contiguous lithology intervals and, for every
         reservoir interval, measures the Shmin contrast against the
         immediately adjacent non-reservoir intervals above and below:
             contrast = mean Shmin(adjacent shale) − mean Shmin(reservoir).
         A side is a **barrier** when that contrast ≥ contrast_threshold_psi.
      3. Grades each reservoir interval:
             Good     — a barrier above AND below (fully contained),
             Moderate — a barrier on one side only,
             Poor     — no adequate barrier (or interval too thin).
      4. Assigns the interval grade to every sample it contains (PERF_QUALITY)
         and returns the recommended perforation zones and barrier intervals.

    All stresses are psi. Returns dict with:
        detail   : per-sample DataFrame (DEPTH, LITHO_CODE, SHMIN_PSI,
                   TREND_PSI, CONTRAST_PSI, PERF_QUALITY).
        zones    : reservoir intervals graded Good/Moderate/Poor.
        barriers : non-reservoir intervals acting as a barrier to a neighbour.
    """
    for col in ("SHMIN_PSI", "LITHO_CODE", "DEPTH"):
        if col not in results.columns:
            raise ValueError(
                "Barrier analysis needs the stress profile and lithology flag — "
                "run the stress calculation with GR mapped first."
            )

    depth = pd.to_numeric(results["DEPTH"], errors="coerce")
    shmin = pd.to_numeric(results["SHMIN_PSI"], errors="coerce")
    codes = pd.to_numeric(results["LITHO_CODE"], errors="coerce").to_numpy(dtype=float)
    n = len(results)

    if int(shmin.notna().sum()) < 5:
        raise ValueError("Not enough valid Shmin samples for a barrier analysis — check the inputs.")

    # 1. Per-sample stress contrast vs the local Shmin trend.
    trend = shmin.rolling(int(trend_window), center=True, min_periods=1).median()
    contrast = shmin - trend

    # 2-3. Interval-based reservoir vs adjacent-shale contrast + grading.
    runs = _lithology_runs(codes)
    mean_shmin = [float(shmin.iloc[s:e + 1].mean()) for (s, e, _c) in runs]

    quality = np.full(n, "N/A", dtype=object)
    t = float(contrast_threshold_psi)
    zone_rows: list[dict] = []
    barrier_pairs: set[int] = set()  # indices (into runs) of shale acting as a barrier

    for k, (s, e, code) in enumerate(runs):
        if code != 0.0:  # only reservoir sand intervals are perforation candidates
            continue
        thickness = float(depth.iloc[e] - depth.iloc[s])
        res_shmin = mean_shmin[k]

        # nearest shale interval above / below (previous / next run that is shale)
        above_k = k - 1 if k - 1 >= 0 and runs[k - 1][2] == 1.0 else None
        below_k = k + 1 if k + 1 < len(runs) and runs[k + 1][2] == 1.0 else None

        c_above = (mean_shmin[above_k] - res_shmin) if above_k is not None else np.nan
        c_below = (mean_shmin[below_k] - res_shmin) if below_k is not None else np.nan
        barrier_above = np.isfinite(c_above) and c_above >= t
        barrier_below = np.isfinite(c_below) and c_below >= t

        if thickness < float(min_zone_thickness) or not np.isfinite(res_shmin):
            grade = "Poor"
        elif barrier_above and barrier_below:
            grade = "Good"
        elif barrier_above or barrier_below:
            grade = "Moderate"
        else:
            grade = "Poor"

        quality[s:e + 1] = grade
        if barrier_above and above_k is not None:
            barrier_pairs.add(above_k)
        if barrier_below and below_k is not None:
            barrier_pairs.add(below_k)

        zone_rows.append(
            {
                "Top": float(depth.iloc[s]),
                "Base": float(depth.iloc[e]),
                "Thickness": round(thickness, 2),
                "Samples": e - s + 1,
                "Mean Shmin (psi)": round(res_shmin, 1) if np.isfinite(res_shmin) else np.nan,
                "Contrast above (psi)": round(c_above, 1) if np.isfinite(c_above) else np.nan,
                "Contrast below (psi)": round(c_below, 1) if np.isfinite(c_below) else np.nan,
                "Barrier above": "Yes" if barrier_above else "No",
                "Barrier below": "Yes" if barrier_below else "No",
                "Quality": grade,
            }
        )

    detail = pd.DataFrame(
        {
            "DEPTH": depth,
            "LITHO_CODE": codes,
            "SHMIN_PSI": shmin,
            "TREND_PSI": trend,
            "CONTRAST_PSI": contrast,
            "PERF_QUALITY": quality,
        }
    )

    zone_cols = ["Top", "Base", "Thickness", "Samples", "Mean Shmin (psi)",
                 "Contrast above (psi)", "Contrast below (psi)",
                 "Barrier above", "Barrier below", "Quality"]
    zones = (
        pd.DataFrame(zone_rows, columns=zone_cols).sort_values("Top").reset_index(drop=True)
        if zone_rows else pd.DataFrame(columns=zone_cols)
    )

    barrier_rows = [
        {
            "Top": float(depth.iloc[runs[k][0]]),
            "Base": float(depth.iloc[runs[k][1]]),
            "Thickness": round(float(depth.iloc[runs[k][1]] - depth.iloc[runs[k][0]]), 2),
            "Mean Shmin (psi)": round(mean_shmin[k], 1) if np.isfinite(mean_shmin[k]) else np.nan,
        }
        for k in sorted(barrier_pairs)
    ]
    barriers = (
        pd.DataFrame(barrier_rows).sort_values("Top").reset_index(drop=True)
        if barrier_rows else pd.DataFrame(columns=["Top", "Base", "Thickness", "Mean Shmin (psi)"])
    )

    return {"detail": detail, "zones": zones, "barriers": barriers}


# ---------------------------------------------------------------------------
# Display helpers & export
# ---------------------------------------------------------------------------

# Canonical column -> friendly display header (all stresses shown in psi).
DISPLAY_NAMES = {
    "DEPTH": "MD",
    "LITHO_CODE": "LITHO",
    "GR": "GR [gAPI]",
    "YME_MPSI": "YME [Mpsi]",
    "PR": "PR [-]",
    "SV_PSI": "Sv [psi]",
    "PP_PSI": "Pp [psi]",
    "SHMIN_PSI": "Shmin [psi]",
    "SHMAX_PSI": "SHmax [psi]",
    "Q_FACTOR": "q-factor [-]",
    "SH_RATIO": "SHmax/Shmin [-]",
    "TREND_PSI": "Shmin trend [psi]",
    "CONTRAST_PSI": "Stress contrast [psi]",
    "PERF_QUALITY": "Perf quality",
}


def display_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rename canonical columns to friendly headers for on-screen tables."""
    return df.rename(columns={c: DISPLAY_NAMES.get(c, c) for c in df.columns})


def lithology_label_column(df: pd.DataFrame) -> pd.Series:
    """Human-readable lithology names for a LITHO_CODE column."""
    codes = pd.to_numeric(df.get("LITHO_CODE"), errors="coerce")
    return codes.map(LITHO_NAME_BY_CODE).fillna("Undefined")


def results_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a results frame for a Streamlit download button."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, float_format="%.4f")
    return buffer.getvalue().encode("utf-8")

