import math

import numpy as np
import numpy.typing as npt


def rotate_stress_to_shmax(
    shmin: float,
    shmax: float,
    svert: float,
    shmax_azimuth: float,
) -> npt.NDArray[np.float64]:
    """Rotate the principal stress tensor into the direction of the maximum horizontal stress.

    Assumes the vertical stress is not tilted.

    Reference:
        Jaeger, John Conrad, Neville GW Cook, and Robert Zimmerman. Fundamentals of rock mechanics. John Wiley & Sons, 2009, Chapter 2.3.

    Args:
        shmin (float): Minimum horizontal stress magnitude. Unit: Pressure
        shmax (float): Maximum horizontal stress magnitude. Unit: Pressure
        svert (float): Vertical stress magnitude. Unit: Pressure
        shmax_azimuth (float): Direction of the maximum horizontal stress magnitude relative to Geographic NORTH. Unit: [deg]

    Returns:
        npt.NDArray[np.float64]: Stress tensor in NEV coordinate system (North-East-Vertical). Unit: same as input pressure unit
    """
    stress_tensor = [shmax, shmin, svert] * np.identity(3)

    shmax_azimuth_rad = shmax_azimuth * (math.pi / 180)

    nev_rotation_matrix = np.array([
        [math.cos(shmax_azimuth_rad), math.sin(shmax_azimuth_rad), 0],
        [-math.sin(shmax_azimuth_rad), math.cos(shmax_azimuth_rad), 0],
        [0, 0, 1],
    ])

    stress_nev = np.matmul(np.transpose(nev_rotation_matrix), np.matmul(stress_tensor, nev_rotation_matrix))
    return stress_nev


def rotate_nev_to_toh(
    borehole_deviation: float,
    borehole_azimuth: float,
    stress_tensor_nev: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Rotate the stress tensor from the geographic reference into the borehole reference system using top of hole as reference.

    Reference:
        Fjaer, Erling, et al. Petroleum related rock mechanics. Vol. 53. Elsevier, 2008; Appendix C; eq C.58.

    Args:
        borehole_deviation (float): Borehole inclination. Unit: [deg]
        borehole_azimuth (float): Borehole azimuth. Unit: [deg]
        stress_tensor_nev (npt.NDArray[np.float64]): Stress tensor in NEV coordinate system (North-East-Vertical). Unit: Pressure

    Returns:
        npt.NDArray[np.float64]: Stress tensor in TOH coordinate system (Top of Hole). Unit: same as input pressure unit
    """
    bh_azimuth_rad = borehole_azimuth * (math.pi / 180)
    bh_deviation_rad = borehole_deviation * (math.pi / 180)

    nev_to_toh = np.array([
        [math.cos(bh_azimuth_rad) * math.cos(bh_deviation_rad), math.sin(bh_azimuth_rad) * math.cos(bh_deviation_rad), -math.sin(bh_deviation_rad)],
        [-math.sin(bh_azimuth_rad), math.cos(bh_azimuth_rad), 0],
        [math.cos(bh_azimuth_rad) * math.sin(bh_deviation_rad), math.sin(bh_azimuth_rad) * math.sin(bh_deviation_rad), math.cos(bh_deviation_rad)],
    ])

    stress_tensor_toh = np.matmul(nev_to_toh, np.matmul(stress_tensor_nev, np.transpose(nev_to_toh)))
    return stress_tensor_toh
