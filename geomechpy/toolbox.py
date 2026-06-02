import numpy as np
import math
from math import pi


def roatate2shmax(SHMIN, SHMAX, SVERT, SHMAX_AZIM):
    """Rotate the principal stress tensore into the direction of the maximum stress magnitude under the assumption that the vertical stress is not tilted
       Reference: Jaeger, John Conrad, Neville GW Cook, and Robert Zimmerman. Fundamentals of rock mechanics. John Wiley & Sons, 2009, Chapter 2.3
    Input:
        SHMIN: Magnitude of the minimum stress
        SHMAX: Magnitude of the maxium stress
        SVERT: Magnitude of the vertical stress
        SHMAX_AZIM: Direction of the maximum stress magnitude relative to Geographic NORTH Unit: degree

            Units: Pressure unit which needs to be consistent between the three stresses

    OUTPUT:
        Stress Tensor in NEV coordinate system (North-East-Vertical)
        Unit: same as input pressure unit

    """
    stress_tensor = [SHMAX, SHMIN, SVERT] * np.identity(3)  # Define the diagonal principal stress tensor

    shmax_dir = SHMAX_AZIM * (math.pi / 180)  # Convert Stress direction into Radians

    NEV_rotation_matrix = np.array([[math.cos(shmax_dir), math.sin(shmax_dir), 0], [-math.sin(shmax_dir), math.cos(shmax_dir), 0], [0, 0, 1]])  # Construct the rotation matrix  # Provide Reference

    # Perform the Matrix Multiplication Transposed(R)*S*R
    aa1 = np.matmul(np.transpose(NEV_rotation_matrix), stress_tensor)
    stress_NEV = np.matmul(aa1, (NEV_rotation_matrix))
    return stress_NEV


def rotNEV2TOH(DEVI, AZIM, stress_tensor):
    """Rotate the stress tensor from geographic reference into the borehole reference system using top of hole as reference
        Fjaer, Erling, et al. Petroleum related rock mechanics. Vol. 53. Elsevier, 2008; Appendix C; eq C.58.

    Input:
        DEVI: Borehole inclination Unit: [deg].
        AZIM: Borehole azimuth Unit: [deg].
        Stress Tensor in NEV coordinate system (North-East-Vertical) Unit: same as input pressure unit.

    OUTPUT:
        Stress Tensor in TOH coordinate system (Top of Hole)
        Unit: same as input pressure unit
    """
    bh_azi = AZIM * (math.pi / 180)
    bh_dev = DEVI * (math.pi / 180)
    NEV2TOH = np.array(
        [[math.cos(bh_azi) * math.cos(bh_dev), math.sin(bh_azi) * math.cos(bh_dev), -math.sin(bh_dev)], [-math.sin(bh_azi), math.cos(bh_azi), 0], [math.cos(bh_azi) * math.sin(bh_dev), math.sin(bh_azi) * math.sin(bh_dev), math.cos(bh_dev)]]
    )
    # Perform the Matrix Multiplication Transposed(R)*S*R
    stress1 = np.matmul(NEV2TOH, stress_tensor)
    stress_tensor_TOH = np.matmul(stress1, np.transpose(NEV2TOH))
    return stress_tensor_TOH
