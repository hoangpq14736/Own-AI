"""
G-code File Parser & Print Statistics Extractor.
Scans header and footer comments from OrcaSlicer, Anycubic Slicer, PrusaSlicer, and Cura.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Optional
import logging

from core.models import GCodeReport

logger = logging.getLogger(__name__)


def parse_gcode_file(file_path: Path) -> GCodeReport:
    """
    Parses metadata and print statistics from a sliced G-code file.
    Reads beginning and trailing bytes for sub-second execution even on 100MB+ files.
    """
    file_name = file_path.name
    if not file_path.exists():
        return GCodeReport(
            file_name=file_name,
            slicer="Unknown",
            print_time="0m",
            filament_length_m=0.0,
            filament_weight_g=0.0,
            filament_type="PLA",
            nozzle_temp=0.0,
            bed_temp=0.0,
            layer_height=0.2,
            total_layers=0,
            text=f"❌ Không tìm thấy tệp `{file_name}`.",
            error="File not found",
        )

    try:
        # Read first 120KB and last 120KB where all slicers store metadata
        file_size = file_path.stat().st_size
        head_bytes = b""
        tail_bytes = b""
        with open(file_path, "rb") as f:
            head_bytes = f.read(128 * 1024)
            if file_size > 128 * 1024:
                f.seek(max(0, file_size - 128 * 1024))
                tail_bytes = f.read()

        text_chunk = (head_bytes + b"\n" + tail_bytes).decode("utf-8", errors="ignore")

        # 1. Detect Slicer
        slicer = "OrcaSlicer / PrusaSlicer"
        if "Anycubic" in text_chunk or "anycubic" in text_chunk.lower():
            slicer = "Anycubic Slicer"
        elif "Cura" in text_chunk or "Generated with Cura" in text_chunk:
            slicer = "UltiMaker Cura"
        elif "PrusaSlicer" in text_chunk:
            slicer = "PrusaSlicer"
        elif "OrcaSlicer" in text_chunk:
            slicer = "OrcaSlicer"

        # 2. Extract Estimated Print Time
        # OrcaSlicer / Prusa: ; estimated printing time (normal mode) = 1h 45m 20s
        # Cura: ;TIME:6320 (seconds)
        print_time = "Không rõ"
        time_match = re.search(r";\s*estimated printing time(?:\s*\(normal mode\))?\s*=\s*(.+)", text_chunk, re.IGNORECASE)
        if time_match:
            print_time = time_match.group(1).strip()
        else:
            cura_time = re.search(r";TIME:(\d+)", text_chunk)
            if cura_time:
                secs = int(cura_time.group(1))
                hrs = secs // 3600
                mins = (secs % 3600) // 60
                print_time = f"{hrs}h {mins}m" if hrs > 0 else f"{mins}m"

        # 3. Extract Filament Usage
        # filament used [mm] = 4520.2
        # filament used [g] = 13.5
        # Cura: ;Filament used: 1.5m
        fil_length_m = 0.0
        fil_weight_g = 0.0
        fil_type = "PLA"

        len_match = re.search(r";\s*filament used \[mm\]\s*=\s*([\d\.]+)", text_chunk, re.IGNORECASE)
        if len_match:
            fil_length_m = round(float(len_match.group(1)) / 1000.0, 2)
        else:
            cura_len = re.search(r";Filament used:\s*([\d\.]+)m", text_chunk, re.IGNORECASE)
            if cura_len:
                fil_length_m = round(float(cura_len.group(1)), 2)

        weight_match = re.search(r";\s*filament used \[g\]\s*=\s*([\d\.]+)", text_chunk, re.IGNORECASE)
        if weight_match:
            fil_weight_g = round(float(weight_match.group(1)), 1)
        elif fil_length_m > 0:
            # Approximate PLA weight for 1.75mm filament: ~3g per meter
            fil_weight_g = round(fil_length_m * 3.0, 1)

        type_match = re.search(r";\s*filament_type\s*=\s*([A-Za-z0-9_\-]+)", text_chunk, re.IGNORECASE)
        if type_match:
            fil_type = type_match.group(1).strip().upper()

        # 4. Temperatures
        nozzle_temp = 205.0
        bed_temp = 60.0

        noz_match = re.search(r";\s*nozzle_temperature\s*=\s*([\d\.]+)", text_chunk, re.IGNORECASE)
        if noz_match:
            nozzle_temp = float(noz_match.group(1))
        else:
            m104_match = re.search(r"M10[49]\s+S(\d+)", text_chunk)
            if m104_match:
                nozzle_temp = float(m104_match.group(1))

        bed_match = re.search(r";\s*bed_temperature\s*=\s*([\d\.]+)", text_chunk, re.IGNORECASE)
        if bed_match:
            bed_temp = float(bed_match.group(1))
        else:
            m140_match = re.search(r"M1[49]0\s+S(\d+)", text_chunk)
            if m140_match:
                bed_temp = float(m140_match.group(1))

        # 5. Layer Height & Total Layers
        layer_height = 0.2
        lh_match = re.search(r";\s*layer_height\s*=\s*([\d\.]+)", text_chunk, re.IGNORECASE)
        if lh_match:
            layer_height = float(lh_match.group(1))
        else:
            cura_lh = re.search(r";Layer height:\s*([\d\.]+)", text_chunk, re.IGNORECASE)
            if cura_lh:
                layer_height = float(cura_lh.group(1))

        total_layers = 0
        layers_match = re.search(r";\s*total layers count\s*=\s*(\d+)", text_chunk, re.IGNORECASE)
        if layers_match:
            total_layers = int(layers_match.group(1))
        else:
            cura_layers = re.search(r";LAYER_COUNT:(\d+)", text_chunk, re.IGNORECASE)
            if cura_layers:
                total_layers = int(cura_layers.group(1))

        # Format clean Markdown summary
        report_lines = [
            f"📄 **Thông Số File Cắt Lát (G-Code)**",
            f"📦 **Tệp:** `{file_name}`",
            f"🔪 **Phần mềm Slicer:** `{slicer}`",
            "",
            f"⏱️ **Thời gian in ước tính:** `{print_time}`",
            f"🧵 **Vật liệu:** `{fil_type}` (`{fil_length_m} m` | `~{fil_weight_g} g`)",
            f"🌡️ **Nhiệt độ:** Đầu phun `{int(nozzle_temp)}°C` | Bàn nhiệt `{int(bed_temp)}°C`",
            f"📏 **Chiều cao lớp (Layer):** `{layer_height} mm` ({total_layers} layers)",
            "",
            f"✅ Sẵn sàng copy vào thẻ nhớ/USB hoặc gửi sang Anycubic Kobra!",
        ]

        return GCodeReport(
            file_name=file_name,
            slicer=slicer,
            print_time=print_time,
            filament_length_m=fil_length_m,
            filament_weight_g=fil_weight_g,
            filament_type=fil_type,
            nozzle_temp=nozzle_temp,
            bed_temp=bed_temp,
            layer_height=layer_height,
            total_layers=total_layers,
            text="\n".join(report_lines),
        )

    except Exception as e:
        logger.error(f"Failed to parse G-code {file_path}: {e}", exc_info=True)
        return GCodeReport(
            file_name=file_name,
            slicer="Unknown",
            print_time="0m",
            filament_length_m=0.0,
            filament_weight_g=0.0,
            filament_type="PLA",
            nozzle_temp=0.0,
            bed_temp=0.0,
            layer_height=0.2,
            total_layers=0,
            text=f"❌ Lỗi khi đọc file G-code: {str(e)}",
            error=str(e),
        )
