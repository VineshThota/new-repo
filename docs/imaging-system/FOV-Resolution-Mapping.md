# Field of View (FOV) and Resolution Mapping

## Technical Documentation for Multispectral Imaging System

**Document Version:** 1.0
**Last Updated:** January 2026
**Status:** Draft - Pending DRAFT-003 System Characterization Verification

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Specifications](#system-specifications)
3. [Mathematical Framework](#mathematical-framework)
4. [FOV vs Resolution Trade-offs](#fov-vs-resolution-trade-offs)
5. [Lookup Tables](#lookup-tables)
6. [Lens Comparison: 85mm vs 300mm](#lens-comparison-85mm-vs-300mm)
7. [Visual Diagrams](#visual-diagrams)
8. [Etendue Limitations and Stitching Requirements](#etendue-limitations-and-stitching-requirements)
9. [Recommendations for Dr. Berns](#recommendations-for-dr-berns)
10. [Verification Against System Characterization](#verification-against-system-characterization)

---

## Executive Summary

This document provides a detailed technical analysis of the trade-offs between Field of View (FOV) and spatial resolution for the multispectral imaging system designed for fine art reproduction. The system is built around the **Canon EOS 5D Mark III** camera body coupled with a **Liquid Crystal Tunable Filter (LCTF)** and interchangeable lenses (85mm and 300mm EFL).

### Key Findings

- **600 PPI and full FOV are mutually exclusive** due to etendue constraints
- Increasing working distance from 2m to 5m **reduces achievable PPI by 60%**
- The **300mm lens at 2m** achieves 600 PPI but limits diagonal FOV to ~11 inches
- Full 24×36" painting capture at 600 PPI requires **stitching up to 15 images**
- The LCTF half-angle constraint (≤7.5°) significantly limits the practical FOV

---

## System Specifications

### Camera: Canon EOS 5D Mark III

| Parameter | Value | Notes |
|-----------|-------|-------|
| Sensor Type | Full Frame CMOS | 35mm format |
| Sensor Dimensions | 36 × 24 mm | Width × Height |
| Effective Megapixels | 22.3 MP | ~5760 × 3840 pixels |
| Pixel Pitch | **6.25 µm** | Center-to-center distance |
| Pixel Count (Horizontal) | 5760 pixels | 36mm ÷ 6.25µm |
| Pixel Count (Vertical) | 3840 pixels | 24mm ÷ 6.25µm |
| Sensor Diagonal | 43.27 mm | √(36² + 24²) |

### Target Object

| Parameter | Value |
|-----------|-------|
| Painting Size | 24 × 36 inches |
| Metric Equivalent | 609.6 × 914.4 mm |
| Diagonal | 43.27 inches (1099.3 mm) |

### Optical System

| Parameter | Value | Notes |
|-----------|-------|-------|
| Target Diagonal FOV | 31° | Design specification |
| LCTF Half-Angle Limit | ≤ 7.5° | Angular sensitivity constraint |
| Available Lenses | 85mm, 300mm | Effective Focal Length |
| Desirable Resolution | 600 PPI | Pixels per inch in object space |
| Working Distance Range | 2,000 - 5,000 mm | As requested by Dr. Berns |

---

## Mathematical Framework

### Fundamental Relationships

#### 1. Magnification

The magnification (m) relates the image size to the object size:

```
m = f / (D - f)
```

Where:
- `f` = Focal length (mm)
- `D` = Object distance (mm)
- `m` = Magnification (dimensionless)

For thin lens approximation at large distances (D >> f):

```
m ≈ f / D
```

#### 2. Field of View Calculation

The Field of View in object space is related to sensor size and magnification:

**Horizontal FOV (object space):**
```
FOV_h = Sensor_width / m = (Sensor_width × D) / f
```

**Vertical FOV (object space):**
```
FOV_v = Sensor_height / m = (Sensor_height × D) / f
```

**Diagonal FOV (angle):**
```
θ_diagonal = 2 × arctan(Sensor_diagonal / (2 × f))
```

#### 3. Pixels Per Inch (PPI) in Object Space

PPI is calculated from the pixel pitch projected onto the object plane:

```
Pixel_size_object = Pixel_pitch / m = (Pixel_pitch × D) / f

PPI = 25.4 mm / Pixel_size_object
    = (25.4 × f) / (Pixel_pitch × D)
```

Where:
- `Pixel_pitch` = 6.25 µm = 0.00625 mm
- `25.4` = mm per inch conversion factor

**Simplified PPI Formula:**
```
PPI = (25.4 × f) / (0.00625 × D)
    = 4064 × f / D
```

Where `f` is in mm and `D` is in mm.

#### 4. Key Insight: Inverse Relationship

From the PPI formula:
```
PPI ∝ f / D
```

**As working distance (D) increases, PPI decreases proportionally.**

---

## FOV vs Resolution Trade-offs

### The Fundamental Constraint

Due to the fixed sensor resolution and pixel pitch, there is an inherent trade-off:

| Increase | Effect on FOV | Effect on PPI |
|----------|---------------|---------------|
| Working Distance (D) | ↑ Increases | ↓ Decreases |
| Focal Length (f) | ↓ Decreases | ↑ Increases |

### Working Distance Impact Analysis (Dr. Berns' Request)

Moving from 2m to 5m working distance:

**With 85mm Lens:**
- At 2m: PPI = 4064 × 85 / 2000 = **172.7 PPI**
- At 5m: PPI = 4064 × 85 / 5000 = **69.1 PPI**
- **Resolution loss: 60%**

**With 300mm Lens:**
- At 2m: PPI = 4064 × 300 / 2000 = **609.6 PPI** ✓ Meets target!
- At 5m: PPI = 4064 × 300 / 5000 = **243.8 PPI**
- **Resolution loss: 60%**

---

## Lookup Tables

### Table 1: 85mm Lens - FOV and Resolution vs. Distance

| Distance (mm) | Distance (m) | H-FOV (mm) | H-FOV (in) | V-FOV (mm) | V-FOV (in) | Diag FOV (mm) | Diag FOV (in) | PPI |
|--------------|--------------|------------|------------|------------|------------|---------------|---------------|-----|
| 2000 | 2.0 | 847.1 | 33.3 | 564.7 | 22.2 | 1018.1 | 40.1 | 172.7 |
| 2250 | 2.25 | 952.9 | 37.5 | 635.3 | 25.0 | 1145.4 | 45.1 | 153.5 |
| 2500 | 2.5 | 1058.8 | 41.7 | 705.9 | 27.8 | 1272.6 | 50.1 | 138.2 |
| 2750 | 2.75 | 1164.7 | 45.9 | 776.5 | 30.6 | 1399.9 | 55.1 | 125.6 |
| 3000 | 3.0 | 1270.6 | 50.0 | 847.1 | 33.3 | 1527.1 | 60.1 | 115.1 |
| 3250 | 3.25 | 1376.5 | 54.2 | 917.6 | 36.1 | 1654.4 | 65.1 | 106.3 |
| 3500 | 3.5 | 1482.4 | 58.4 | 988.2 | 38.9 | 1781.7 | 70.1 | 98.7 |
| 3750 | 3.75 | 1588.2 | 62.5 | 1058.8 | 41.7 | 1908.9 | 75.2 | 92.2 |
| 4000 | 4.0 | 1694.1 | 66.7 | 1129.4 | 44.5 | 2036.2 | 80.2 | 86.4 |
| 4250 | 4.25 | 1800.0 | 70.9 | 1200.0 | 47.2 | 2163.4 | 85.2 | 81.4 |
| 4500 | 4.5 | 1905.9 | 75.0 | 1270.6 | 50.0 | 2290.7 | 90.2 | 76.8 |
| 4750 | 4.75 | 2011.8 | 79.2 | 1341.2 | 52.8 | 2418.0 | 95.2 | 72.8 |
| 5000 | 5.0 | 2117.6 | 83.4 | 1411.8 | 55.6 | 2545.2 | 100.2 | 69.1 |

**Key Observations (85mm Lens):**
- ✓ Can capture full 24×36" painting at ~2.6m distance
- ✗ Never achieves 600 PPI target
- Maximum PPI at 2m: **172.7 PPI** (71% below target)

---

### Table 2: 300mm Lens - FOV and Resolution vs. Distance

| Distance (mm) | Distance (m) | H-FOV (mm) | H-FOV (in) | V-FOV (mm) | V-FOV (in) | Diag FOV (mm) | Diag FOV (in) | PPI |
|--------------|--------------|------------|------------|------------|------------|---------------|---------------|-----|
| 2000 | 2.0 | 240.0 | 9.4 | 160.0 | 6.3 | 288.4 | 11.4 | **609.6** |
| 2250 | 2.25 | 270.0 | 10.6 | 180.0 | 7.1 | 324.5 | 12.8 | 541.9 |
| 2500 | 2.5 | 300.0 | 11.8 | 200.0 | 7.9 | 360.6 | 14.2 | 487.7 |
| 2750 | 2.75 | 330.0 | 13.0 | 220.0 | 8.7 | 396.6 | 15.6 | 443.3 |
| 3000 | 3.0 | 360.0 | 14.2 | 240.0 | 9.4 | 432.7 | 17.0 | 406.4 |
| 3250 | 3.25 | 390.0 | 15.4 | 260.0 | 10.2 | 468.7 | 18.5 | 374.4 |
| 3500 | 3.5 | 420.0 | 16.5 | 280.0 | 11.0 | 504.8 | 19.9 | 347.7 |
| 3750 | 3.75 | 450.0 | 17.7 | 300.0 | 11.8 | 540.8 | 21.3 | 324.5 |
| 4000 | 4.0 | 480.0 | 18.9 | 320.0 | 12.6 | 576.9 | 22.7 | 304.8 |
| 4250 | 4.25 | 510.0 | 20.1 | 340.0 | 13.4 | 612.9 | 24.1 | 287.4 |
| 4500 | 4.5 | 540.0 | 21.3 | 360.0 | 14.2 | 649.0 | 25.6 | 271.5 |
| 4750 | 4.75 | 570.0 | 22.4 | 380.0 | 15.0 | 685.0 | 27.0 | 257.0 |
| 5000 | 5.0 | 600.0 | 23.6 | 400.0 | 15.7 | 721.1 | 28.4 | 243.8 |

**Key Observations (300mm Lens):**
- ✓ Achieves 600 PPI target at 2m distance
- ✗ FOV at 2m only captures 9.4" × 6.3" (11.4" diagonal)
- At 5m: PPI drops to 243.8 (59% below target)
- Cannot capture full 24×36" painting in single shot

---

### Table 3: Distance Required for Target Painting (24×36") Coverage

| Lens (mm) | Min Distance for Full Painting (m) | PPI at that Distance |
|-----------|-----------------------------------|----------------------|
| 85 | 2.16 | 160.0 |
| 100 | 2.54 | 160.0 |
| 135 | 3.43 | 160.0 |
| 200 | 5.08 | 160.0 |
| 300 | 7.62 | 160.0 |

**Note:** All lenses yield the same PPI when positioned to capture the full painting, as the magnification adjusts proportionally.

---

### Table 4: PPI Comparison at Key Distances

| Distance (m) | 85mm PPI | 300mm PPI | 300mm FOV Diagonal (in) | Fits 24×36" Painting? |
|--------------|----------|-----------|------------------------|----------------------|
| 2.0 | 172.7 | **609.6** | 11.4 | No (11.4" < 43.3") |
| 2.5 | 138.2 | 487.7 | 14.2 | No |
| 3.0 | 115.1 | 406.4 | 17.0 | No |
| 3.5 | 98.7 | 347.7 | 19.9 | No |
| 4.0 | 86.4 | 304.8 | 22.7 | No |
| 4.5 | 76.8 | 271.5 | 25.6 | No |
| 5.0 | 69.1 | 243.8 | 28.4 | No |

---

## Lens Comparison: 85mm vs 300mm

### Summary Comparison

| Characteristic | 85mm Lens | 300mm Lens |
|----------------|-----------|------------|
| **Primary Use** | Wide FOV capture | High resolution capture |
| **Diagonal FOV (angular)** | ~28.6° | ~8.2° |
| **FOV at 2m** | 40.1" diagonal | 11.4" diagonal |
| **FOV at 5m** | 100.2" diagonal | 28.4" diagonal |
| **PPI at 2m** | 172.7 | **609.6** |
| **PPI at 5m** | 69.1 | 243.8 |
| **Can capture 24×36" in one shot?** | Yes (at ≥2.16m) | No (even at 5m) |
| **Achieves 600 PPI?** | Never | Only at ≤2.0m |
| **LCTF Compatibility** | May exceed 7.5° half-angle | Within angular limits |

### Angular Field of View Calculations

The angular FOV depends only on focal length and sensor size:

```
θ_horizontal = 2 × arctan(18 / f)
θ_vertical = 2 × arctan(12 / f)
θ_diagonal = 2 × arctan(21.635 / f)
```

| Lens | θ_horizontal | θ_vertical | θ_diagonal |
|------|-------------|------------|------------|
| 85mm | 23.9° | 16.1° | 28.6° |
| 300mm | 6.9° | 4.6° | 8.2° |

### LCTF Compatibility Analysis

The LCTF requires chief ray angles ≤ 7.5° half-angle:

- **85mm lens:** Maximum half-angle = 14.3° → **Exceeds LCTF limit**
- **300mm lens:** Maximum half-angle = 4.1° → **Within LCTF limit**

**Implication:** The 85mm lens may cause spectral shifts at the sensor edges due to LCTF angular sensitivity. The 300mm lens is fully compatible with the LCTF constraints.

---

## Visual Diagrams

### Diagram 1: FOV Coverage vs. Object Distance

```
FOV Coverage vs. Distance for Target Painting (24×36")
=====================================================

Distance (m)      85mm Lens Coverage          300mm Lens Coverage
                  (Diagonal in inches)        (Diagonal in inches)

    2.0   ████████████████████████████████░░░░░  40.1"
                                                  ████████████  11.4"

    2.5   ████████████████████████████████████████████████████  50.1"
                                                  ██████████████  14.2"

    3.0   ████████████████████████████████████████████████████████████████  60.1"
                                                  ██████████████████  17.0"

    4.0   ████████████████████████████████████████████████████████████████████████████████  80.2"
                                                  ██████████████████████████  22.7"

    5.0   ████████████████████████████████████████████████████████████████████████████████████████████████████  100.2"
                                                  ████████████████████████████████  28.4"

Legend: █ = Coverage | ░ = Target painting size (43.3" diagonal)
        ━━━ Target painting diagonal (43.3")
```

### Diagram 2: PPI Degradation with Distance

```
PPI vs. Working Distance
========================

PPI
 │
700│
   │  ★ 609.6 (300mm @ 2m) - TARGET ACHIEVED!
600│══════════════════════════════════════════════════ 600 PPI TARGET
   │     ╲
500│      ╲ 300mm lens
   │       ╲
400│        ╲
   │         ╲
300│          ╲                        ★ 243.8 (300mm @ 5m)
   │           ╲──────────────────────
200│────────────────────────────────────────────────────
   │  ★ 172.7    85mm lens (relatively flat decline)
150│──────╲──────────────────────────────────────────
   │       ╲
100│        ╲──────────────────────────★ 69.1 (85mm @ 5m)
   │
 50│
   │
  0├────┬────┬────┬────┬────┬────┬────┬────┬────┬────→ Distance (m)
   0   0.5   1   1.5   2   2.5   3   3.5   4   4.5   5

Legend: ★ = Key measurement points
        ═ = Target resolution (600 PPI)
```

### Diagram 3: 24×36" Painting Capture Geometry

```
                    Target Painting Capture Geometry
                    ═══════════════════════════════

                           24" (609.6 mm)
                    ┌─────────────────────────┐
                    │                         │
                    │                         │
                    │      24 × 36 inch       │  36"
                    │        Painting         │  (914.4 mm)
                    │                         │
                    │   Diagonal: 43.27"      │
                    │                         │
                    └─────────────────────────┘
                              │
                              │
                    ┌─────────┴─────────┐
                    │  Working Distance │
                    │    (D = 2-5m)     │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    │   LCTF + Lens     │
                    │    Assembly       │
                    │                   │
                    │  ┌─────────────┐  │
                    │  │   Sensor    │  │
                    │  │  36×24 mm   │  │
                    │  └─────────────┘  │
                    │                   │
                    └───────────────────┘
                       Canon EOS 5D
                        Mark III
```

### Diagram 4: 300mm Lens FOV at 2m (High Resolution Option)

```
              300mm Lens @ 2m Distance - FOV Coverage
              ════════════════════════════════════════

              Target Painting (24×36")
        ┌───────────────────────────────────────────┐
        │                                           │
        │                                           │
        │        ┌───────────────┐                  │
        │        │               │                  │
        │        │    9.4" ×     │ ← Single frame   │
        │        │    6.3"       │   FOV at 2m      │
        │        │               │   (609.6 PPI)    │
        │        │   11.4"       │                  │
        │        │   diagonal    │                  │
        │        └───────────────┘                  │
        │                                           │
        │                                           │
        │        Need ~15 images to cover           │
        │        full painting at 600 PPI           │
        └───────────────────────────────────────────┘

        24" (horizontal) × 36" (vertical)
        Painting diagonal: 43.27"
```

### Diagram 5: Image Stitching Grid for 600 PPI Full Painting

```
        Image Stitching Pattern for 24×36" at 600 PPI
        ══════════════════════════════════════════════

        Using 300mm lens at 2m distance
        Each frame: ~9.4" × 6.3" (with ~10% overlap for stitching)
        Effective coverage: ~8.5" × 5.7" per frame

        ┌───────────────────────────────────────────┐
        │ ┌─────┐ ┌─────┐ ┌─────┐                   │
        │ │  1  │ │  2  │ │  3  │                   │
        │ └─────┘ └─────┘ └─────┘                   │  Row 1
        │ ┌─────┐ ┌─────┐ ┌─────┐                   │
        │ │  4  │ │  5  │ │  6  │                   │  Row 2
        │ └─────┘ └─────┘ └─────┘                   │
        │ ┌─────┐ ┌─────┐ ┌─────┐                   │  Row 3
        │ │  7  │ │  8  │ │  9  │                   │
        │ └─────┘ └─────┘ └─────┘                   │
        │ ┌─────┐ ┌─────┐ ┌─────┐                   │  Row 4
        │ │ 10  │ │ 11  │ │ 12  │                   │
        │ └─────┘ └─────┘ └─────┘                   │
        │ ┌─────┐ ┌─────┐ ┌─────┐                   │  Row 5
        │ │ 13  │ │ 14  │ │ 15  │                   │
        │ └─────┘ └─────┘ └─────┘                   │
        └───────────────────────────────────────────┘

        Grid: 3 columns × 5 rows = 15 images
        Total pixels: 15 × 22.3 MP = ~335 MP composite
```

---

## Etendue Limitations and Stitching Requirements

### Understanding Etendue

Etendue (geometric extent) is a conserved quantity in optical systems that relates the area of light collection to the solid angle of acceptance:

```
G = A × Ω × n²
```

Where:
- `G` = Etendue
- `A` = Area (sensor or aperture)
- `Ω` = Solid angle
- `n` = Refractive index

### Why 600 PPI + Full FOV is Impossible

Given fixed sensor resolution:

| Requirement | Impact on Etendue |
|-------------|-------------------|
| High PPI (600) | Requires small object-space pixel size |
| Large FOV | Requires large collection angle |
| Fixed Sensor | Limits total pixel count |

**The trade-off is fundamental:** To achieve 600 PPI over a 24×36" painting requires:

```
Pixels needed = (24" × 600) × (36" × 600) = 14,400 × 21,600 = 311 Megapixels
```

The Canon EOS 5D Mark III has only **22.3 MP**, which means:

```
Coverage at 600 PPI = 22.3 MP / 360,000 pixels/in² ≈ 62 in² ≈ 7.9" × 7.9"
```

This matches our calculated 300mm FOV at 2m: ~9.4" × 6.3" = 59.2 in²

### Stitching Requirements Summary

| Target | Resolution | Images Required | Total Capture |
|--------|------------|-----------------|---------------|
| 24×36" painting | 300 PPI | ~4 images | ~89 MP |
| 24×36" painting | 600 PPI | ~15 images | ~335 MP |
| 24×36" painting | 1200 PPI | ~60 images | ~1.34 GP |

---

## Recommendations for Dr. Berns

### Question: Impact of Increasing Working Distance from 2m to 5m

**Answer Summary:**

Increasing the working distance from 2m to 5m has the following effects:

1. **PPI Reduction:** Resolution decreases by exactly 60% (from D/D' ratio)
   - 85mm lens: 172.7 PPI → 69.1 PPI
   - 300mm lens: 609.6 PPI → 243.8 PPI

2. **FOV Increase:** Coverage increases by 150%
   - 85mm lens: 40.1" → 100.2" diagonal
   - 300mm lens: 11.4" → 28.4" diagonal

3. **Painting Capture Capability:**
   - 85mm at 5m: Can capture paintings up to 83" × 56" in one shot
   - 300mm at 5m: Can capture paintings up to 23.6" × 15.7" in one shot

### Recommended Configurations

#### Option 1: Full Painting, Lower Resolution (Single Shot)
- **Configuration:** 85mm lens at 2.16m
- **Result:** Full 24×36" coverage at 160 PPI
- **Use case:** Preview/documentation

#### Option 2: High Resolution, Limited FOV (PRD Option 2)
- **Configuration:** 300mm lens at 2m
- **Result:** 609.6 PPI over 9.4" × 6.3" area
- **Use case:** Detail capture, requires 15 images for full painting

#### Option 3: Balanced Approach
- **Configuration:** 300mm lens at 3m
- **Result:** 406 PPI over 14.2" × 9.4" area
- **Use case:** Compromise between resolution and coverage (~7 images needed)

---

## Verification Against System Characterization

### Pixel Pitch Verification

**Specification:** 6.25 µm center-to-center

**Verification Calculation:**
```
Sensor width = 36 mm = 36,000 µm
Pixel count (horizontal) = 5760 pixels
Calculated pitch = 36,000 / 5760 = 6.25 µm ✓
```

### FOV Calculation Verification

**For 300mm lens at 2m distance:**
```
Horizontal FOV = (36 mm × 2000 mm) / 300 mm = 240 mm = 9.45" ✓
Vertical FOV = (24 mm × 2000 mm) / 300 mm = 160 mm = 6.30" ✓
```

### PPI Calculation Verification

**For 300mm lens at 2m distance:**
```
Object pixel size = (6.25 µm × 2000 mm) / 300 mm = 41.67 µm
PPI = 25.4 mm / 0.04167 mm = 609.5 ≈ 609.6 PPI ✓
```

### Target Diagonal FOV (31°) Verification

The 31° diagonal FOV specification corresponds to approximately:

```
f = Sensor_diagonal / (2 × tan(31°/2))
f = 43.27 mm / (2 × tan(15.5°))
f = 43.27 / 0.554
f ≈ 78 mm
```

This suggests the target was designed around an ~80mm focal length for full-field imaging, consistent with the 85mm lens selection.

---

## Appendix A: Calculation Reference

### Python Verification Code

```python
import math

# System Constants
SENSOR_WIDTH_MM = 36.0
SENSOR_HEIGHT_MM = 24.0
PIXEL_PITCH_UM = 6.25
PIXEL_PITCH_MM = PIXEL_PITCH_UM / 1000

# Derived Constants
SENSOR_DIAGONAL_MM = math.sqrt(SENSOR_WIDTH_MM**2 + SENSOR_HEIGHT_MM**2)
PIXELS_H = int(SENSOR_WIDTH_MM / PIXEL_PITCH_MM)
PIXELS_V = int(SENSOR_HEIGHT_MM / PIXEL_PITCH_MM)

def calculate_fov_and_ppi(focal_length_mm, distance_mm):
    """Calculate FOV and PPI for given lens and distance."""
    magnification = focal_length_mm / distance_mm

    # FOV in object space (mm)
    fov_h_mm = SENSOR_WIDTH_MM / magnification
    fov_v_mm = SENSOR_HEIGHT_MM / magnification
    fov_diag_mm = SENSOR_DIAGONAL_MM / magnification

    # Convert to inches
    fov_h_in = fov_h_mm / 25.4
    fov_v_in = fov_v_mm / 25.4
    fov_diag_in = fov_diag_mm / 25.4

    # PPI calculation
    pixel_size_object_mm = PIXEL_PITCH_MM / magnification
    ppi = 25.4 / pixel_size_object_mm

    return {
        'fov_h_mm': fov_h_mm,
        'fov_h_in': fov_h_in,
        'fov_v_mm': fov_v_mm,
        'fov_v_in': fov_v_in,
        'fov_diag_mm': fov_diag_mm,
        'fov_diag_in': fov_diag_in,
        'ppi': ppi,
        'magnification': magnification
    }

# Generate lookup table
for distance_m in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
    distance_mm = distance_m * 1000
    for focal_length in [85, 300]:
        result = calculate_fov_and_ppi(focal_length, distance_mm)
        print(f"{focal_length}mm @ {distance_m}m: "
              f"FOV={result['fov_diag_in']:.1f}\" PPI={result['ppi']:.1f}")
```

---

## Appendix B: Quick Reference Card

### For Field Use

| Need | Use | Distance | Expect |
|------|-----|----------|--------|
| Full 24×36" painting | 85mm | 2.2m+ | ~160 PPI |
| Maximum detail | 300mm | 2.0m | 609 PPI |
| Quick survey | 85mm | 3-5m | 70-115 PPI |
| Stitching base | 300mm | 2.0m | 15 images → 600 PPI full |

### PPI Quick Calculation

```
PPI ≈ 4064 × (focal_length_mm) / (distance_mm)
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Jan 2026 | System Documentation | Initial release |

---

## Dependencies and References

- **DRAFT-003:** System Characterization Data (pending verification)
- **PRD Section:** Options for Customer, Option 2
- **Canon EOS 5D Mark III:** Official specifications
- **LCTF Specifications:** Angular sensitivity ≤ 7.5° half-angle
