import math

import numpy as np
import pytest

from geomechpy.toolbox import rotate_nev_to_toh, rotate_stress_to_shmax

TOLERANCE = 1e-6


class TestRotateStressToShmax:
    def test_returns_ndarray(self) -> None:
        result = rotate_stress_to_shmax(shmin=10000, shmax=12000, svert=13000, shmax_azimuth=0)
        assert isinstance(result, np.ndarray)

    def test_output_shape_is_3x3(self) -> None:
        result = rotate_stress_to_shmax(shmin=10000, shmax=12000, svert=13000, shmax_azimuth=0)
        assert result.shape == (3, 3)

    def test_output_is_symmetric(self) -> None:
        result = rotate_stress_to_shmax(shmin=10000, shmax=12000, svert=13000, shmax_azimuth=45)
        assert result == pytest.approx(result.T, rel=TOLERANCE)

    def test_azimuth_zero_shmax_aligned_with_north(self) -> None:
        # At azimuth=0 the rotation is identity; shmax sits on the North axis [0,0]
        result = rotate_stress_to_shmax(shmin=10000, shmax=12000, svert=13000, shmax_azimuth=0)
        assert result[0, 0] == pytest.approx(12000, rel=TOLERANCE)
        assert result[1, 1] == pytest.approx(10000, rel=TOLERANCE)
        assert result[2, 2] == pytest.approx(13000, rel=TOLERANCE)

    def test_azimuth_90_shmax_aligned_with_east(self) -> None:
        # At azimuth=90 shmax rotates onto the East axis [1,1]
        result = rotate_stress_to_shmax(shmin=10000, shmax=12000, svert=13000, shmax_azimuth=90)
        assert result[0, 0] == pytest.approx(10000, rel=TOLERANCE)
        assert result[1, 1] == pytest.approx(12000, rel=TOLERANCE)
        assert result[2, 2] == pytest.approx(13000, rel=TOLERANCE)

    def test_vertical_stress_is_invariant_to_azimuth(self) -> None:
        # Rotation around the vertical axis must not affect the vertical stress component
        result_0 = rotate_stress_to_shmax(shmin=10000, shmax=12000, svert=13000, shmax_azimuth=0)
        result_60 = rotate_stress_to_shmax(shmin=10000, shmax=12000, svert=13000, shmax_azimuth=60)
        assert result_0[2, 2] == pytest.approx(result_60[2, 2], rel=TOLERANCE)


class TestRotateNevToToh:
    def test_returns_ndarray(self) -> None:
        stress_nev = np.diag([12000.0, 10000.0, 13000.0])
        result = rotate_nev_to_toh(borehole_deviation=0, borehole_azimuth=0, stress_tensor_nev=stress_nev)
        assert isinstance(result, np.ndarray)

    def test_output_shape_is_3x3(self) -> None:
        stress_nev = np.diag([12000.0, 10000.0, 13000.0])
        result = rotate_nev_to_toh(borehole_deviation=0, borehole_azimuth=0, stress_tensor_nev=stress_nev)
        assert result.shape == (3, 3)

    def test_output_is_symmetric(self) -> None:
        stress_nev = np.diag([12000.0, 10000.0, 13000.0])
        result = rotate_nev_to_toh(borehole_deviation=30, borehole_azimuth=45, stress_tensor_nev=stress_nev)
        assert result == pytest.approx(result.T, rel=TOLERANCE)

    def test_vertical_borehole_preserves_stress_tensor(self) -> None:
        # deviation=0, azimuth=0 produces an identity rotation matrix
        stress_nev = np.diag([12000.0, 10000.0, 13000.0])
        result = rotate_nev_to_toh(borehole_deviation=0, borehole_azimuth=0, stress_tensor_nev=stress_nev)
        assert result == pytest.approx(stress_nev, rel=TOLERANCE)

    def test_known_value_diagonal_tensor_horizontal_borehole(self) -> None:
        # For deviation=90, azimuth=0: borehole pointing North, z-axis of TOH = North
        # The NEV vertical stress (index [2,2]) maps to the borehole axial direction (index [2,2] in TOH)
        # The NEV North stress (index [0,0]) maps to the radial direction (index [0,0] in TOH? Let's verify)
        # nev_to_toh at dev=90, azim=0:
        # Row 0: [cos(0)*cos(90), sin(0)*cos(90), -sin(90)] = [0, 0, -1]
        # Row 1: [-sin(0), cos(0), 0]              = [0, 1, 0]
        # Row 2: [cos(0)*sin(90), sin(0)*sin(90), cos(90)] = [1, 0, 0]
        # R * diag([shmax,shmin,svert]) * R^T:
        # Diagonal of result: [svert, shmin, shmax]
        stress_nev = np.diag([12000.0, 10000.0, 13000.0])
        result = rotate_nev_to_toh(borehole_deviation=90, borehole_azimuth=0, stress_tensor_nev=stress_nev)
        assert result[0, 0] == pytest.approx(13000, rel=TOLERANCE)
        assert result[1, 1] == pytest.approx(10000, rel=TOLERANCE)
        assert result[2, 2] == pytest.approx(12000, rel=TOLERANCE)
