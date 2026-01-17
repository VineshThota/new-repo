#!/usr/bin/env python3
"""
Unit tests for FOV and Resolution Calculator.

Tests verify:
1. Sensor specification consistency
2. FOV calculations match expected values
3. PPI calculations match expected values
4. Edge cases and error handling
5. Stitching calculations
"""

import math
import pytest
from fov_resolution_calculator import (
    # Constants
    SENSOR_WIDTH_MM,
    SENSOR_HEIGHT_MM,
    PIXEL_PITCH_MM,
    PIXEL_PITCH_UM,
    SENSOR_DIAGONAL_MM,
    PIXELS_HORIZONTAL,
    PIXELS_VERTICAL,
    TOTAL_MEGAPIXELS,
    LCTF_HALF_ANGLE_LIMIT,
    # Functions
    calculate_fov_and_ppi,
    calculate_distance_for_ppi,
    calculate_distance_for_fov,
    calculate_images_for_stitching,
    generate_lookup_table,
    verify_sensor_specifications,
    FOVResult,
)


class TestSensorSpecifications:
    """Tests for sensor specification constants."""

    def test_sensor_dimensions(self):
        """Verify Canon EOS 5D Mark III sensor dimensions."""
        assert SENSOR_WIDTH_MM == 36.0
        assert SENSOR_HEIGHT_MM == 24.0

    def test_pixel_pitch(self):
        """Verify pixel pitch is 6.25 micrometers."""
        assert PIXEL_PITCH_UM == 6.25
        assert abs(PIXEL_PITCH_MM - 0.00625) < 1e-10

    def test_sensor_diagonal(self):
        """Verify sensor diagonal calculation."""
        expected = math.sqrt(36**2 + 24**2)
        assert abs(SENSOR_DIAGONAL_MM - expected) < 1e-10
        assert abs(SENSOR_DIAGONAL_MM - 43.27) < 0.01

    def test_pixel_counts(self):
        """Verify pixel counts derived from sensor dimensions and pitch."""
        expected_h = int(36.0 / 0.00625)  # 5760
        expected_v = int(24.0 / 0.00625)  # 3840
        assert PIXELS_HORIZONTAL == expected_h
        assert PIXELS_VERTICAL == expected_v

    def test_megapixels(self):
        """Verify total megapixel count matches ~22.3 MP."""
        assert abs(TOTAL_MEGAPIXELS - 22.12) < 0.5  # 5760 * 3840 / 1M = 22.12

    def test_verify_sensor_specifications(self):
        """Test the verification function returns True for valid specs."""
        assert verify_sensor_specifications() is True


class TestFOVCalculations:
    """Tests for Field of View calculations."""

    def test_85mm_at_2m_fov(self):
        """Test 85mm lens at 2m distance - FOV calculations."""
        result = calculate_fov_and_ppi(85, 2000)

        # Expected: FOV_h = 36 * 2000 / 85 = 847.06 mm
        expected_fov_h_mm = (36.0 * 2000) / 85
        assert abs(result.fov_horizontal_mm - expected_fov_h_mm) < 0.1

        # Expected: FOV_v = 24 * 2000 / 85 = 564.71 mm
        expected_fov_v_mm = (24.0 * 2000) / 85
        assert abs(result.fov_vertical_mm - expected_fov_v_mm) < 0.1

        # Check inches conversion
        assert abs(result.fov_horizontal_in - expected_fov_h_mm / 25.4) < 0.1

    def test_300mm_at_2m_fov(self):
        """Test 300mm lens at 2m distance - FOV calculations."""
        result = calculate_fov_and_ppi(300, 2000)

        # Expected: FOV_h = 36 * 2000 / 300 = 240 mm = 9.45 inches
        assert abs(result.fov_horizontal_mm - 240.0) < 0.1
        assert abs(result.fov_horizontal_in - 9.45) < 0.1

        # Expected: FOV_v = 24 * 2000 / 300 = 160 mm = 6.30 inches
        assert abs(result.fov_vertical_mm - 160.0) < 0.1
        assert abs(result.fov_vertical_in - 6.30) < 0.1

    def test_fov_diagonal(self):
        """Test diagonal FOV calculation."""
        result = calculate_fov_and_ppi(300, 2000)

        # Diagonal = sqrt(240^2 + 160^2) = 288.44 mm
        expected_diag = math.sqrt(240**2 + 160**2)
        assert abs(result.fov_diagonal_mm - expected_diag) < 0.1

    def test_fov_increases_with_distance(self):
        """Verify FOV increases proportionally with distance."""
        result_2m = calculate_fov_and_ppi(85, 2000)
        result_4m = calculate_fov_and_ppi(85, 4000)

        # FOV should double when distance doubles
        assert (
            abs(result_4m.fov_horizontal_mm / result_2m.fov_horizontal_mm - 2.0) < 0.01
        )

    def test_fov_decreases_with_focal_length(self):
        """Verify FOV decreases with increasing focal length."""
        result_85mm = calculate_fov_and_ppi(85, 2000)
        result_300mm = calculate_fov_and_ppi(300, 2000)

        # 300mm should have ~3.53x smaller FOV than 85mm
        ratio = result_85mm.fov_horizontal_mm / result_300mm.fov_horizontal_mm
        assert abs(ratio - 300 / 85) < 0.01


class TestPPICalculations:
    """Tests for Pixels Per Inch calculations."""

    def test_300mm_at_2m_ppi(self):
        """Test 300mm lens at 2m achieves ~600 PPI."""
        result = calculate_fov_and_ppi(300, 2000)

        # PPI = 25.4 * 300 / (0.00625 * 2000) = 609.6
        expected_ppi = (25.4 * 300) / (0.00625 * 2000)
        assert abs(result.ppi - expected_ppi) < 0.1
        assert abs(result.ppi - 609.6) < 0.5

    def test_85mm_at_2m_ppi(self):
        """Test 85mm lens at 2m PPI calculation."""
        result = calculate_fov_and_ppi(85, 2000)

        # PPI = 25.4 * 85 / (0.00625 * 2000) = 172.72
        expected_ppi = (25.4 * 85) / (0.00625 * 2000)
        assert abs(result.ppi - expected_ppi) < 0.1
        assert abs(result.ppi - 172.7) < 0.5

    def test_ppi_decreases_with_distance(self):
        """Verify PPI decreases inversely with distance."""
        result_2m = calculate_fov_and_ppi(300, 2000)
        result_4m = calculate_fov_and_ppi(300, 4000)

        # PPI should halve when distance doubles
        assert abs(result_2m.ppi / result_4m.ppi - 2.0) < 0.01

    def test_ppi_increases_with_focal_length(self):
        """Verify PPI increases proportionally with focal length."""
        result_85mm = calculate_fov_and_ppi(85, 2000)
        result_300mm = calculate_fov_and_ppi(300, 2000)

        # PPI ratio should match focal length ratio
        ratio = result_300mm.ppi / result_85mm.ppi
        assert abs(ratio - 300 / 85) < 0.01

    def test_2m_to_5m_ppi_degradation(self):
        """Test Dr. Berns' question: 2m to 5m impact on PPI."""
        for focal_length in [85, 300]:
            result_2m = calculate_fov_and_ppi(focal_length, 2000)
            result_5m = calculate_fov_and_ppi(focal_length, 5000)

            # PPI should decrease by 60% (5000/2000 = 2.5, so PPI drops to 40%)
            ratio = result_5m.ppi / result_2m.ppi
            assert abs(ratio - 0.4) < 0.01  # 2000/5000 = 0.4


class TestMeetsTargetFlags:
    """Tests for target achievement flags."""

    def test_300mm_2m_meets_ppi_target(self):
        """Verify 300mm at 2m meets 600 PPI target."""
        result = calculate_fov_and_ppi(300, 2000)
        assert result.meets_ppi_target is True

    def test_85mm_never_meets_ppi_target(self):
        """Verify 85mm never meets 600 PPI target in working range."""
        for distance in [2000, 3000, 4000, 5000]:
            result = calculate_fov_and_ppi(85, distance)
            assert result.meets_ppi_target is False

    def test_300mm_5m_does_not_meet_ppi_target(self):
        """Verify 300mm at 5m does not meet 600 PPI target."""
        result = calculate_fov_and_ppi(300, 5000)
        assert result.meets_ppi_target is False
        assert result.ppi < 600

    def test_lctf_compatibility_300mm(self):
        """Verify 300mm lens is LCTF compatible."""
        result = calculate_fov_and_ppi(300, 2000)
        assert result.lctf_compatible is True
        # Half angle should be < 7.5 degrees
        assert result.angular_fov_diagonal / 2 <= LCTF_HALF_ANGLE_LIMIT

    def test_lctf_compatibility_85mm(self):
        """Verify 85mm lens exceeds LCTF angular limits."""
        result = calculate_fov_and_ppi(85, 2000)
        assert result.lctf_compatible is False
        # Half angle should be > 7.5 degrees
        assert result.angular_fov_diagonal / 2 > LCTF_HALF_ANGLE_LIMIT


class TestDistanceCalculations:
    """Tests for distance calculation functions."""

    def test_distance_for_600_ppi_300mm(self):
        """Calculate distance for 600 PPI with 300mm lens."""
        distance = calculate_distance_for_ppi(300, 600)

        # Should be approximately 2032mm
        # D = 25.4 * 300 / (0.00625 * 600) = 2032
        expected = (25.4 * 300) / (0.00625 * 600)
        assert abs(distance - expected) < 1
        assert abs(distance - 2032) < 5

    def test_distance_for_600_ppi_85mm(self):
        """Calculate distance for 600 PPI with 85mm lens."""
        distance = calculate_distance_for_ppi(85, 600)

        # Should be approximately 575mm
        expected = (25.4 * 85) / (0.00625 * 600)
        assert abs(distance - expected) < 1

    def test_distance_for_full_painting_85mm(self):
        """Calculate distance to capture full painting diagonal with 85mm."""
        # Target diagonal = 43.27 inches = 1099 mm
        target_diag_mm = 43.27 * 25.4
        distance = calculate_distance_for_fov(85, target_diag_mm)

        # Verify by calculating FOV at that distance
        result = calculate_fov_and_ppi(85, distance)
        assert abs(result.fov_diagonal_mm - target_diag_mm) < 1


class TestStitchingCalculations:
    """Tests for image stitching requirements."""

    def test_stitching_300mm_at_2m(self):
        """Test stitching calculation for 300mm at 2m."""
        cols, rows, total = calculate_images_for_stitching(300, 2000)

        # With 10% overlap, effective coverage is 90% of FOV
        # FOV: 240mm x 160mm, effective: 216mm x 144mm
        # Target: 609.6mm x 914.4mm
        # Cols: ceil(609.6/216) = 3
        # Rows: ceil(914.4/144) = 7
        assert cols >= 3
        assert rows >= 6
        assert total >= 15  # Should need at least 15 images

    def test_stitching_85mm_at_3m(self):
        """Test stitching for 85mm at 3m - verify reasonable stitching count."""
        cols, rows, total = calculate_images_for_stitching(85, 3000)

        # 85mm at 3m: FOV ~50" x 33" which covers 24x36" painting
        # With 10% overlap, effective coverage is ~45" x 30"
        # For 24" wide: 1 column needed
        # For 36" tall: may need 2 rows depending on overlap
        # Total should be small (1-2 images)
        assert total <= 2


class TestEdgeCases:
    """Tests for error handling and edge cases."""

    def test_zero_focal_length_raises_error(self):
        """Verify zero focal length raises ValueError."""
        with pytest.raises(ValueError):
            calculate_fov_and_ppi(0, 2000)

    def test_negative_focal_length_raises_error(self):
        """Verify negative focal length raises ValueError."""
        with pytest.raises(ValueError):
            calculate_fov_and_ppi(-85, 2000)

    def test_zero_distance_raises_error(self):
        """Verify zero distance raises ValueError."""
        with pytest.raises(ValueError):
            calculate_fov_and_ppi(85, 0)

    def test_negative_distance_raises_error(self):
        """Verify negative distance raises ValueError."""
        with pytest.raises(ValueError):
            calculate_fov_and_ppi(85, -2000)

    def test_very_short_distance(self):
        """Test calculation at very short distance."""
        result = calculate_fov_and_ppi(300, 500)
        assert result.ppi > 2000  # Very high PPI at close range
        assert result.fov_horizontal_mm < 100  # Very narrow FOV


class TestLookupTable:
    """Tests for lookup table generation."""

    def test_lookup_table_generation(self):
        """Test lookup table generates correct number of results."""
        focal_lengths = [85, 300]
        distances = [2000, 3000, 4000, 5000]

        results = generate_lookup_table(focal_lengths, distances)

        assert len(results) == len(focal_lengths) * len(distances)

    def test_lookup_table_result_types(self):
        """Verify all results are FOVResult objects."""
        results = generate_lookup_table([85], [2000])

        assert len(results) == 1
        assert isinstance(results[0], FOVResult)


class TestPixelSizeInObjectSpace:
    """Tests for pixel size calculations."""

    def test_pixel_size_object_300mm_2m(self):
        """Test pixel size in object space for 300mm at 2m."""
        result = calculate_fov_and_ppi(300, 2000)

        # Pixel size = 6.25um * 2000 / 300 = 41.67 um = 0.04167 mm
        expected = (PIXEL_PITCH_MM * 2000) / 300
        assert abs(result.pixel_size_object_mm - expected) < 0.0001
        assert abs(result.pixel_size_object_mm - 0.04167) < 0.001

    def test_pixel_size_ppi_relationship(self):
        """Verify PPI = 25.4 / pixel_size_object_mm."""
        result = calculate_fov_and_ppi(300, 2000)

        calculated_ppi = 25.4 / result.pixel_size_object_mm
        assert abs(calculated_ppi - result.ppi) < 0.01


class TestAngularFOV:
    """Tests for angular field of view calculations."""

    def test_85mm_angular_fov(self):
        """Test angular FOV for 85mm lens."""
        result = calculate_fov_and_ppi(85, 2000)

        # Angular FOV only depends on focal length and sensor size
        expected_diagonal = 2 * math.degrees(math.atan(SENSOR_DIAGONAL_MM / (2 * 85)))
        assert abs(result.angular_fov_diagonal - expected_diagonal) < 0.1
        assert abs(result.angular_fov_diagonal - 28.6) < 0.5

    def test_300mm_angular_fov(self):
        """Test angular FOV for 300mm lens."""
        result = calculate_fov_and_ppi(300, 2000)

        expected_diagonal = 2 * math.degrees(math.atan(SENSOR_DIAGONAL_MM / (2 * 300)))
        assert abs(result.angular_fov_diagonal - expected_diagonal) < 0.1
        assert abs(result.angular_fov_diagonal - 8.2) < 0.5

    def test_angular_fov_independent_of_distance(self):
        """Verify angular FOV doesn't change with distance."""
        result_2m = calculate_fov_and_ppi(85, 2000)
        result_5m = calculate_fov_and_ppi(85, 5000)

        assert (
            abs(result_2m.angular_fov_diagonal - result_5m.angular_fov_diagonal) < 0.01
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
