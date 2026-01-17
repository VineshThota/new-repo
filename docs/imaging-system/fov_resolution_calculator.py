#!/usr/bin/env python3
"""
FOV and Resolution Calculator for Multispectral Imaging System

This module provides calculations for Field of View (FOV) and Pixels Per Inch (PPI)
based on the Canon EOS 5D Mark III sensor specifications and lens configurations.

System Specifications:
- Sensor: Canon EOS 5D Mark III (36 x 24 mm, 6.25 µm pixel pitch)
- Target: 24 x 36 inch painting (609.6 x 914.4 mm)
- Lenses: 85mm and 300mm EFL
- Target Resolution: 600 PPI
"""

import math
from dataclasses import dataclass
from typing import List, Tuple


# =============================================================================
# System Constants
# =============================================================================

# Sensor Specifications (Canon EOS 5D Mark III)
SENSOR_WIDTH_MM: float = 36.0
SENSOR_HEIGHT_MM: float = 24.0
PIXEL_PITCH_UM: float = 6.25
PIXEL_PITCH_MM: float = PIXEL_PITCH_UM / 1000.0

# Derived Sensor Constants
SENSOR_DIAGONAL_MM: float = math.sqrt(SENSOR_WIDTH_MM**2 + SENSOR_HEIGHT_MM**2)
PIXELS_HORIZONTAL: int = int(SENSOR_WIDTH_MM / PIXEL_PITCH_MM)
PIXELS_VERTICAL: int = int(SENSOR_HEIGHT_MM / PIXEL_PITCH_MM)
TOTAL_MEGAPIXELS: float = (PIXELS_HORIZONTAL * PIXELS_VERTICAL) / 1_000_000

# Target Painting (24 x 36 inches)
TARGET_WIDTH_IN: float = 24.0
TARGET_HEIGHT_IN: float = 36.0
TARGET_WIDTH_MM: float = TARGET_WIDTH_IN * 25.4
TARGET_HEIGHT_MM: float = TARGET_HEIGHT_IN * 25.4
TARGET_DIAGONAL_IN: float = math.sqrt(TARGET_WIDTH_IN**2 + TARGET_HEIGHT_IN**2)
TARGET_DIAGONAL_MM: float = TARGET_DIAGONAL_IN * 25.4

# Available Lenses (mm)
LENS_85MM: float = 85.0
LENS_300MM: float = 300.0

# Target Resolution (PPI)
TARGET_PPI: float = 600.0

# LCTF Angular Constraint (degrees)
LCTF_HALF_ANGLE_LIMIT: float = 7.5

# Working Distance Range (mm)
MIN_DISTANCE_MM: float = 2000.0
MAX_DISTANCE_MM: float = 5000.0


@dataclass
class FOVResult:
    """Data class containing FOV and resolution calculation results."""

    focal_length_mm: float
    distance_mm: float
    magnification: float

    # FOV in millimeters
    fov_horizontal_mm: float
    fov_vertical_mm: float
    fov_diagonal_mm: float

    # FOV in inches
    fov_horizontal_in: float
    fov_vertical_in: float
    fov_diagonal_in: float

    # Angular FOV (degrees)
    angular_fov_horizontal: float
    angular_fov_vertical: float
    angular_fov_diagonal: float

    # Resolution
    ppi: float
    pixel_size_object_mm: float

    # Derived
    meets_ppi_target: bool
    covers_target_painting: bool
    lctf_compatible: bool


def calculate_fov_and_ppi(focal_length_mm: float, distance_mm: float) -> FOVResult:
    """
    Calculate Field of View and Pixels Per Inch for given lens and distance.

    Args:
        focal_length_mm: Effective focal length of lens in millimeters
        distance_mm: Object distance (working distance) in millimeters

    Returns:
        FOVResult containing all calculated values

    Raises:
        ValueError: If focal_length_mm or distance_mm is non-positive
    """
    if focal_length_mm <= 0:
        raise ValueError(f"Focal length must be positive, got {focal_length_mm}")
    if distance_mm <= 0:
        raise ValueError(f"Distance must be positive, got {distance_mm}")

    # Magnification (thin lens approximation)
    magnification = focal_length_mm / distance_mm

    # FOV in object space (mm)
    fov_h_mm = SENSOR_WIDTH_MM / magnification
    fov_v_mm = SENSOR_HEIGHT_MM / magnification
    fov_diag_mm = SENSOR_DIAGONAL_MM / magnification

    # Convert to inches
    fov_h_in = fov_h_mm / 25.4
    fov_v_in = fov_v_mm / 25.4
    fov_diag_in = fov_diag_mm / 25.4

    # Angular FOV (degrees)
    angular_h = 2 * math.degrees(math.atan(SENSOR_WIDTH_MM / (2 * focal_length_mm)))
    angular_v = 2 * math.degrees(math.atan(SENSOR_HEIGHT_MM / (2 * focal_length_mm)))
    angular_diag = 2 * math.degrees(
        math.atan(SENSOR_DIAGONAL_MM / (2 * focal_length_mm))
    )

    # Pixel size in object space
    pixel_size_object_mm = PIXEL_PITCH_MM / magnification

    # PPI calculation
    ppi = 25.4 / pixel_size_object_mm

    # Check constraints
    meets_ppi = ppi >= TARGET_PPI
    covers_painting = (
        fov_h_mm >= TARGET_WIDTH_MM and fov_v_mm >= TARGET_HEIGHT_MM
    ) or (fov_diag_mm >= TARGET_DIAGONAL_MM)

    # LCTF compatibility (half-angle must be <= 7.5 degrees)
    max_half_angle = angular_diag / 2
    lctf_ok = max_half_angle <= LCTF_HALF_ANGLE_LIMIT

    return FOVResult(
        focal_length_mm=focal_length_mm,
        distance_mm=distance_mm,
        magnification=magnification,
        fov_horizontal_mm=fov_h_mm,
        fov_vertical_mm=fov_v_mm,
        fov_diagonal_mm=fov_diag_mm,
        fov_horizontal_in=fov_h_in,
        fov_vertical_in=fov_v_in,
        fov_diagonal_in=fov_diag_in,
        angular_fov_horizontal=angular_h,
        angular_fov_vertical=angular_v,
        angular_fov_diagonal=angular_diag,
        ppi=ppi,
        pixel_size_object_mm=pixel_size_object_mm,
        meets_ppi_target=meets_ppi,
        covers_target_painting=covers_painting,
        lctf_compatible=lctf_ok,
    )


def calculate_distance_for_ppi(focal_length_mm: float, target_ppi: float) -> float:
    """
    Calculate the required distance to achieve a specific PPI.

    Args:
        focal_length_mm: Effective focal length of lens in millimeters
        target_ppi: Desired pixels per inch

    Returns:
        Required distance in millimeters
    """
    # PPI = 25.4 * f / (pixel_pitch * D)
    # D = 25.4 * f / (pixel_pitch * PPI)
    return (25.4 * focal_length_mm) / (PIXEL_PITCH_MM * target_ppi)


def calculate_distance_for_fov(
    focal_length_mm: float, required_fov_diagonal_mm: float
) -> float:
    """
    Calculate the required distance to achieve a specific diagonal FOV.

    Args:
        focal_length_mm: Effective focal length of lens in millimeters
        required_fov_diagonal_mm: Required diagonal FOV in millimeters

    Returns:
        Required distance in millimeters
    """
    # FOV_diag = sensor_diag / m = sensor_diag * D / f
    # D = FOV_diag * f / sensor_diag
    return (required_fov_diagonal_mm * focal_length_mm) / SENSOR_DIAGONAL_MM


def calculate_images_for_stitching(
    focal_length_mm: float,
    distance_mm: float,
    target_width_mm: float = TARGET_WIDTH_MM,
    target_height_mm: float = TARGET_HEIGHT_MM,
    overlap_percent: float = 10.0,
) -> Tuple[int, int, int]:
    """
    Calculate number of images required to stitch a full painting.

    Args:
        focal_length_mm: Effective focal length of lens
        distance_mm: Object distance in millimeters
        target_width_mm: Target painting width
        target_height_mm: Target painting height
        overlap_percent: Percentage overlap between adjacent images

    Returns:
        Tuple of (columns, rows, total_images)
    """
    result = calculate_fov_and_ppi(focal_length_mm, distance_mm)

    # Effective coverage per image (accounting for overlap)
    overlap_factor = 1 - (overlap_percent / 100)
    effective_width = result.fov_horizontal_mm * overlap_factor
    effective_height = result.fov_vertical_mm * overlap_factor

    # Number of images needed
    columns = max(1, math.ceil(target_width_mm / effective_width))
    rows = max(1, math.ceil(target_height_mm / effective_height))

    return columns, rows, columns * rows


def generate_lookup_table(
    focal_lengths: List[float], distances_mm: List[float]
) -> List[FOVResult]:
    """
    Generate a lookup table for multiple lens/distance combinations.

    Args:
        focal_lengths: List of focal lengths to calculate
        distances_mm: List of distances to calculate

    Returns:
        List of FOVResult objects
    """
    results = []
    for fl in focal_lengths:
        for d in distances_mm:
            results.append(calculate_fov_and_ppi(fl, d))
    return results


def print_lookup_table(results: List[FOVResult]) -> None:
    """Print formatted lookup table to console."""
    print("\n" + "=" * 120)
    print("FOV vs Resolution Lookup Table")
    print("=" * 120)
    print(
        f"{'Lens':>6} | {'Dist (m)':>8} | {'H-FOV (in)':>10} | {'V-FOV (in)':>10} | "
        f"{'Diag (in)':>10} | {'PPI':>8} | {'600 PPI':>7} | {'Full Painting':>13} | {'LCTF OK':>7}"
    )
    print("-" * 120)

    for r in results:
        ppi_status = "YES" if r.meets_ppi_target else "NO"
        painting_status = "YES" if r.covers_target_painting else "NO"
        lctf_status = "YES" if r.lctf_compatible else "NO"

        print(
            f"{r.focal_length_mm:>5.0f}mm | {r.distance_mm / 1000:>8.2f} | "
            f"{r.fov_horizontal_in:>10.1f} | {r.fov_vertical_in:>10.1f} | "
            f"{r.fov_diagonal_in:>10.1f} | {r.ppi:>8.1f} | {ppi_status:>7} | "
            f"{painting_status:>13} | {lctf_status:>7}"
        )


def verify_sensor_specifications() -> bool:
    """
    Verify that sensor specifications are self-consistent.

    Returns:
        True if all specifications are consistent
    """
    # Check pixel pitch calculation
    calculated_pitch = SENSOR_WIDTH_MM / PIXELS_HORIZONTAL
    pitch_ok = abs(calculated_pitch - PIXEL_PITCH_MM) < 0.0001

    # Check megapixels
    expected_mp = 22.3
    mp_ok = abs(TOTAL_MEGAPIXELS - expected_mp) < 0.5

    # Check sensor diagonal
    expected_diagonal = 43.27
    diagonal_ok = abs(SENSOR_DIAGONAL_MM - expected_diagonal) < 0.1

    print("\nSensor Specification Verification:")
    print(
        f"  Pixel pitch: {calculated_pitch:.4f} mm (expected: {PIXEL_PITCH_MM:.4f} mm) - {'PASS' if pitch_ok else 'FAIL'}"
    )
    print(
        f"  Megapixels: {TOTAL_MEGAPIXELS:.2f} MP (expected: ~{expected_mp} MP) - {'PASS' if mp_ok else 'FAIL'}"
    )
    print(
        f"  Diagonal: {SENSOR_DIAGONAL_MM:.2f} mm (expected: ~{expected_diagonal} mm) - {'PASS' if diagonal_ok else 'FAIL'}"
    )

    return pitch_ok and mp_ok and diagonal_ok


def main() -> None:
    """Main function demonstrating calculator usage."""
    print("=" * 60)
    print("FOV and Resolution Calculator for Multispectral Imaging")
    print("=" * 60)

    # Verify sensor specifications
    verify_sensor_specifications()

    # Define distances (2m to 5m in 250mm steps)
    distances = [
        2000,
        2250,
        2500,
        2750,
        3000,
        3250,
        3500,
        3750,
        4000,
        4250,
        4500,
        4750,
        5000,
    ]

    # Calculate for both lenses
    results = generate_lookup_table([LENS_85MM, LENS_300MM], distances)
    print_lookup_table(results)

    # Key findings for Dr. Berns
    print("\n" + "=" * 60)
    print("Key Findings for Dr. Berns")
    print("=" * 60)

    # 300mm at 2m (best resolution)
    best_res = calculate_fov_and_ppi(300, 2000)
    print("\n300mm lens at 2m:")
    print(
        f"  - PPI: {best_res.ppi:.1f} {'(MEETS TARGET)' if best_res.meets_ppi_target else ''}"
    )
    print(
        f'  - FOV: {best_res.fov_horizontal_in:.1f}" x {best_res.fov_vertical_in:.1f}" ({best_res.fov_diagonal_in:.1f}" diagonal)'
    )
    print(f"  - LCTF Compatible: {'Yes' if best_res.lctf_compatible else 'No'}")

    # 85mm at 2m
    wide_2m = calculate_fov_and_ppi(85, 2000)
    print("\n85mm lens at 2m:")
    print(f"  - PPI: {wide_2m.ppi:.1f}")
    print(
        f'  - FOV: {wide_2m.fov_horizontal_in:.1f}" x {wide_2m.fov_vertical_in:.1f}" ({wide_2m.fov_diagonal_in:.1f}" diagonal)'
    )
    print(
        f'  - Covers 24x36" painting: {"Yes" if wide_2m.covers_target_painting else "No"}'
    )

    # Impact of 2m to 5m
    print("\n" + "-" * 60)
    print("Impact of Distance Change (2m -> 5m):")
    for fl in [85, 300]:
        r_2m = calculate_fov_and_ppi(fl, 2000)
        r_5m = calculate_fov_and_ppi(fl, 5000)
        ppi_change = ((r_5m.ppi - r_2m.ppi) / r_2m.ppi) * 100
        fov_change = (
            (r_5m.fov_diagonal_in - r_2m.fov_diagonal_in) / r_2m.fov_diagonal_in
        ) * 100

        print(f"\n{fl}mm lens:")
        print(f"  - PPI: {r_2m.ppi:.1f} -> {r_5m.ppi:.1f} ({ppi_change:+.1f}%)")
        print(
            f'  - FOV diagonal: {r_2m.fov_diagonal_in:.1f}" -> {r_5m.fov_diagonal_in:.1f}" ({fov_change:+.1f}%)'
        )

    # Stitching requirements
    print("\n" + "-" * 60)
    print('Stitching Requirements for 24x36" at 600 PPI:')
    cols, rows, total = calculate_images_for_stitching(300, 2000)
    print(f"  - Using 300mm at 2m: {cols} columns x {rows} rows = {total} images")

    # Required distance for target
    dist_for_600ppi = calculate_distance_for_ppi(300, 600)
    print(
        f"\n  - Distance for 600 PPI with 300mm lens: {dist_for_600ppi:.0f}mm ({dist_for_600ppi / 1000:.2f}m)"
    )

    dist_for_full_painting = calculate_distance_for_fov(85, TARGET_DIAGONAL_MM)
    print(
        f"  - Distance for full painting with 85mm lens: {dist_for_full_painting:.0f}mm ({dist_for_full_painting / 1000:.2f}m)"
    )


if __name__ == "__main__":
    main()
