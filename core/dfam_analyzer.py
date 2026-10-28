"""
DfAM (Design for Additive Manufacturing) Analyzer for 3D Printability.
Pure Python standard library implementation - measures bounding box, volume,
overhang angles, and Anycubic Kobra build volume compatibility.
"""

from __future__ import annotations
import struct
from pathlib import Path
from typing import Tuple, List, Optional
import math
import logging

from core.models import DFAMReport

logger = logging.getLogger(__name__)

# Default build volume for Anycubic Kobra (standard: 220x220x250 mm)
DEFAULT_BED_X = 220.0
DEFAULT_BED_Y = 220.0
DEFAULT_BED_Z = 250.0
PLA_DENSITY_G_CM3 = 1.24
DEFAULT_INFILL_RATIO = 0.20


def is_binary_stl(data: bytes) -> bool:
    """Detects whether STL data is binary or ASCII."""
    if len(data) < 84:
        return False
    # Check if header starts with 'solid'
    if data[:5].lower() == b"solid":
        # Could still be binary if file size matches 84 + num_triangles * 50
        num_triangles = struct.unpack("<I", data[80:84])[0]
        expected_size = 84 + num_triangles * 50
        if len(data) == expected_size:
            return True
        # If it contains null bytes, it is binary
        if b"\x00" in data[:80]:
            return True
        return False
    return True


def analyze_stl_file(
    file_path: Path,
    bed_x: float = DEFAULT_BED_X,
    bed_y: float = DEFAULT_BED_Y,
    bed_z: float = DEFAULT_BED_Z,
) -> DFAMReport:
    """
    Analyzes an STL mesh file against DfAM 3D printing rules for Anycubic Kobra.
    """
    file_name = file_path.name
    if not file_path.exists():
        return DFAMReport(
            file_name=file_name,
            triangles=0,
            size_x=0.0,
            size_y=0.0,
            size_z=0.0,
            fits_bed=False,
            volume_cm3=0.0,
            weight_pla_g=0.0,
            overhang_ratio=0.0,
            needs_support=False,
            text=f"❌ Không tìm thấy tệp `{file_name}`.",
            error="File not found",
        )

    try:
        with open(file_path, "rb") as f:
            data = f.read()

        if len(data) < 15:
            return DFAMReport(
                file_name=file_name,
                triangles=0,
                size_x=0.0,
                size_y=0.0,
                size_z=0.0,
                fits_bed=False,
                volume_cm3=0.0,
                weight_pla_g=0.0,
                overhang_ratio=0.0,
                needs_support=False,
                text="❌ Tệp STL rỗng hoặc bị lỗi định dạng.",
                error="Invalid STL",
            )

        binary = is_binary_stl(data)
        min_x = min_y = min_z = float("inf")
        max_x = max_y = max_z = float("-inf")
        total_signed_volume = 0.0
        overhang_facets = 0
        total_facets = 0

        if binary:
            if len(data) < 84:
                raise ValueError("Binary STL file header too short.")
            total_facets = struct.unpack("<I", data[80:84])[0]
            offset = 84
            # Each triangle is 50 bytes: 12 normal + 36 vertices + 2 attr
            for _ in range(total_facets):
                if offset + 50 > len(data):
                    break
                nx, ny, nz = struct.unpack("<3f", data[offset : offset + 12])
                v1x, v1y, v1z = struct.unpack("<3f", data[offset + 12 : offset + 24])
                v2x, v2y, v2z = struct.unpack("<3f", data[offset + 24 : offset + 36])
                v3x, v3y, v3z = struct.unpack("<3f", data[offset + 36 : offset + 48])
                offset += 50

                # Bounding box
                min_x = min(min_x, v1x, v2x, v3x)
                max_x = max(max_x, v1x, v2x, v3x)
                min_y = min(min_y, v1y, v2y, v3y)
                max_y = max(max_y, v1y, v2y, v3y)
                min_z = min(min_z, v1z, v2z, v3z)
                max_z = max(max_z, v1z, v2z, v3z)

                # Signed volume of tetrahedron from origin: (v1 x v2) . v3 / 6
                cross_x = v1y * v2z - v1z * v2y
                cross_y = v1z * v2x - v1x * v2z
                cross_z = v1x * v2y - v1y * v2x
                total_signed_volume += (cross_x * v3x + cross_y * v3y + cross_z * v3z) / 6.0

                # Overhang check: normal pointing downward steeper than 45 deg
                norm_len = math.sqrt(nx * nx + ny * ny + nz * nz)
                if norm_len > 1e-6:
                    unit_nz = nz / norm_len
                    # -0.7071 is cos(135 deg) = 45 deg overhang below horizon
                    if unit_nz < -0.7071:
                        overhang_facets += 1

        else:
            # ASCII STL parser
            text_str = data.decode("utf-8", errors="ignore")
            lines = text_str.splitlines()
            current_normal = (0.0, 0.0, 0.0)
            verts: List[Tuple[float, float, float]] = []

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == "facet" and len(parts) >= 5 and parts[1] == "normal":
                    try:
                        current_normal = (float(parts[2]), float(parts[3]), float(parts[4]))
                    except ValueError:
                        current_normal = (0.0, 0.0, 0.0)
                    verts = []
                elif parts[0] == "vertex" and len(parts) >= 4:
                    try:
                        vx, vy, vz = float(parts[1]), float(parts[2]), float(parts[3])
                        verts.append((vx, vy, vz))
                    except ValueError:
                        pass
                elif parts[0] == "endfacet":
                    if len(verts) == 3:
                        total_facets += 1
                        (v1x, v1y, v1z), (v2x, v2y, v2z), (v3x, v3y, v3z) = verts
                        min_x = min(min_x, v1x, v2x, v3x)
                        max_x = max(max_x, v1x, v2x, v3x)
                        min_y = min(min_y, v1y, v2y, v3y)
                        max_y = max(max_y, v1y, v2y, v3y)
                        min_z = min(min_z, v1z, v2z, v3z)
                        max_z = max(max_z, v1z, v2z, v3z)

                        cross_x = v1y * v2z - v1z * v2y
                        cross_y = v1z * v2x - v1x * v2z
                        cross_z = v1x * v2y - v1y * v2x
                        total_signed_volume += (cross_x * v3x + cross_y * v3y + cross_z * v3z) / 6.0

                        nx, ny, nz = current_normal
                        norm_len = math.sqrt(nx * nx + ny * ny + nz * nz)
                        if norm_len > 1e-6 and (nz / norm_len) < -0.7071:
                            overhang_facets += 1

        if total_facets == 0 or min_x == float("inf"):
            return DFAMReport(
                file_name=file_name,
                triangles=0,
                size_x=0.0,
                size_y=0.0,
                size_z=0.0,
                fits_bed=False,
                volume_cm3=0.0,
                weight_pla_g=0.0,
                overhang_ratio=0.0,
                needs_support=False,
                text="❌ Không tìm thấy tam giác lưới (facets) hợp lệ trong tệp STL.",
                error="Zero triangles",
            )

        size_x = round(max(0.0, max_x - min_x), 2)
        size_y = round(max(0.0, max_y - min_y), 2)
        size_z = round(max(0.0, max_z - min_z), 2)

        fits_bed = (size_x <= bed_x and size_y <= bed_y and size_z <= bed_z)
        volume_mm3 = abs(total_signed_volume)
        volume_cm3 = round(volume_mm3 / 1000.0, 2)

        # Estimated weight: solid volume * infill ratio * density
        weight_g = round(volume_cm3 * DEFAULT_INFILL_RATIO * PLA_DENSITY_G_CM3, 1)
        overhang_ratio = round((overhang_facets / total_facets) * 100.0, 1) if total_facets > 0 else 0.0
        needs_support = overhang_ratio > 3.0

        # Construct Report Text
        bed_status = "✅ Vừa vặn bàn in" if fits_bed else f"⚠️ Vượt quá bàn in Anycubic Kobra ({bed_x}x{bed_y}x{bed_z} mm)"
        support_status = f"⚠️ Cần bật Support (~{overhang_ratio}% diện tích có góc dốc > 45°)" if needs_support else "✅ Không cần Support (Góc dốc an toàn < 45°)"

        report_lines = [
            f"🖨️ **Báo Cáo Khả Dụng In 3D (DfAM) - Anycubic Kobra**",
            f"📦 **Tệp:** `{file_name}`",
            "",
            f"📐 **Kích thước mô hình:** `{size_x} × {size_y} × {size_z} mm`",
            f"📊 **Số lượng tam giác (Mesh):** `{total_facets:,}` facets",
            f"🧊 **Thể tích ước tính:** `{volume_cm3} cm³`",
            f"⚖️ **Trọng lượng PLA (20% infill):** `~{weight_g} g`",
            "",
            f"🎯 **Đánh giá DfAM & Bàn in:**",
            f"• **Bàn in Anycubic Kobra:** {bed_status}",
            f"• **Góc dốc Overhang (> 45°):** {support_status}",
        ]

        if needs_support:
            report_lines.append("\n💡 **Khuyến nghị:** Nên dùng **Tree Support (Organic)** trong OrcaSlicer/Cura để dễ bóc và tiết kiệm nhựa.")
        else:
            report_lines.append("\n💡 **Khuyến nghị:** Mô hình in thẳng không cần support, bề mặt đáy phẳng đẹp.")

        return DFAMReport(
            file_name=file_name,
            triangles=total_facets,
            size_x=size_x,
            size_y=size_y,
            size_z=size_z,
            fits_bed=fits_bed,
            volume_cm3=volume_cm3,
            weight_pla_g=weight_g,
            overhang_ratio=overhang_ratio,
            needs_support=needs_support,
            text="\n".join(report_lines),
        )

    except Exception as e:
        logger.error(f"Failed to analyze STL file {file_path}: {e}", exc_info=True)
        return DFAMReport(
            file_name=file_name,
            triangles=0,
            size_x=0.0,
            size_y=0.0,
            size_z=0.0,
            fits_bed=False,
            volume_cm3=0.0,
            weight_pla_g=0.0,
            overhang_ratio=0.0,
            needs_support=False,
            text=f"❌ Lỗi khi phân tích tệp 3D: {str(e)}",
            error=str(e),
        )
