"""
3D Printer Telemetry Client for Anycubic Kobra.
Supports Moonraker (Klipper) and OctoPrint REST APIs.
"""

from __future__ import annotations
import logging
from typing import Optional
import aiohttp

from core.models import PrinterStatus

logger = logging.getLogger(__name__)


class PrinterClient:
    """
    Connects to Anycubic Kobra via Moonraker (Klipper) or OctoPrint.
    """
    def __init__(
        self,
        url: str = "",
        api_type: str = "moonraker",
        api_key: str = "",
        timeout_seconds: float = 5.0,
    ):
        self.url = url.strip().rstrip("/")
        self.api_type = api_type.strip().lower()
        self.api_key = api_key.strip()
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    @property
    def is_configured(self) -> bool:
        return bool(self.url)

    async def get_status(self) -> PrinterStatus:
        """Fetches live temperature, job progress, and status from the printer."""
        if not self.is_configured:
            help_text = (
                "🖨️ **Máy in Anycubic Kobra - Trạng Thái Offline**\n\n"
                "ℹ️ Chưa cấu hình địa chỉ kết nối máy in trong `.env`.\n\n"
                "💡 **Cách kết nối với Anycubic Kobra:**\n"
                "1. Nếu bạn cắm Kobra qua Raspberry Pi / TV Box (chạy Klipper / OctoPrint):\n"
                "   • Thêm vào `.env`: `PRINTER_URL=http://192.168.1.x:7125`\n"
                "   • `PRINTER_API_TYPE=moonraker` (hoặc `octoprint`)\n"
                "2. Bạn vẫn có thể dùng các tính năng 3D ngoại tuyến:\n"
                "   • Gửi file `.stl`, `.3mf`, `.step` $\\rightarrow$ Bot tự phân tích DfAM (`/dfam`).\n"
                "   • Gửi file `.gcode` $\\rightarrow$ Bot bóc tách thông số nhiệt độ & thời gian in (`/gcode`)."
            )
            return PrinterStatus(
                is_connected=False,
                printer_name="Anycubic Kobra",
                state="Offline / Unconfigured",
                nozzle_temp=0.0,
                nozzle_target=0.0,
                bed_temp=0.0,
                bed_target=0.0,
                progress_percent=0.0,
                time_elapsed_min=0.0,
                time_remaining_min=0.0,
                file_name="",
                text=help_text,
            )

        headers = {}
        if self.api_key:
            headers["X-Api-Key"] = self.api_key

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                if self.api_type == "octoprint":
                    return await self._fetch_octoprint(session, headers)
                else:
                    return await self._fetch_moonraker(session, headers)
        except Exception as e:
            logger.warning(f"Failed to connect to printer at {self.url}: {e}")
            err_text = (
                f"🖨️ **Anycubic Kobra:** ⚠️ Không thể kết nối tới `{self.url}`\n"
                f"Lỗi: `{str(e)}`\n\n"
                f"Kiểm tra xem máy in hoặc Raspberry Pi/Moonraker có đang bật không!"
            )
            return PrinterStatus(
                is_connected=False,
                printer_name="Anycubic Kobra",
                state="Connection Error",
                nozzle_temp=0.0,
                nozzle_target=0.0,
                bed_temp=0.0,
                bed_target=0.0,
                progress_percent=0.0,
                time_elapsed_min=0.0,
                time_remaining_min=0.0,
                file_name="",
                text=err_text,
            )

    async def _fetch_moonraker(self, session: aiohttp.ClientSession, headers: dict) -> PrinterStatus:
        query_url = f"{self.url}/printer/objects/query?print_stats&extruder&heater_bed"
        async with session.get(query_url, headers=headers) as resp:
            if resp.status != 200:
                raise ValueError(f"Moonraker returned HTTP {resp.status}")
            data = await resp.json()
            status = data.get("result", {}).get("status", {})

            stats = status.get("print_stats", {})
            state = stats.get("state", "Standby").capitalize()
            filename = stats.get("filename", "")
            print_duration = stats.get("print_duration", 0.0) / 60.0

            extruder = status.get("extruder", {})
            nozzle_temp = round(extruder.get("temperature", 0.0), 1)
            nozzle_target = round(extruder.get("target", 0.0), 1)

            heater_bed = status.get("heater_bed", {})
            bed_temp = round(heater_bed.get("temperature", 0.0), 1)
            bed_target = round(heater_bed.get("target", 0.0), 1)

            progress = 0.0
            if state.lower() == "printing" and stats.get("total_duration", 0) > 0:
                progress = round(min(100.0, (stats.get("print_duration", 0) / stats.get("total_duration", 1)) * 100.0), 1)

            text = (
                f"🖨️ **Anycubic Kobra - Klipper Telemetry**\n\n"
                f"• **Trạng thái:** `{state}`\n"
                f"• **Đầu phun (Nozzle):** `{nozzle_temp}°C` / `{nozzle_target}°C`\n"
                f"• **Bàn nhiệt (Bed):** `{bed_temp}°C` / `{bed_target}°C`\n"
                f"• **Tiến độ in:** `{progress}%` (Đã in: `{round(print_duration, 1)}m`)\n"
                f"• **Tệp đang in:** `{filename or 'Không có'}`"
            )
            return PrinterStatus(
                is_connected=True,
                printer_name="Anycubic Kobra",
                state=state,
                nozzle_temp=nozzle_temp,
                nozzle_target=nozzle_target,
                bed_temp=bed_temp,
                bed_target=bed_target,
                progress_percent=progress,
                time_elapsed_min=print_duration,
                time_remaining_min=0.0,
                file_name=filename,
                text=text,
            )

    async def _fetch_octoprint(self, session: aiohttp.ClientSession, headers: dict) -> PrinterStatus:
        async with session.get(f"{self.url}/api/printer", headers=headers) as p_resp:
            p_data = await p_resp.json() if p_resp.status == 200 else {}

        async with session.get(f"{self.url}/api/job", headers=headers) as j_resp:
            j_data = await j_resp.json() if j_resp.status == 200 else {}

        state = p_data.get("state", {}).get("text", "Operational")
        temps = p_data.get("temperature", {})
        tool0 = temps.get("tool0", {})
        bed = temps.get("bed", {})

        nozzle_temp = tool0.get("actual", 0.0)
        nozzle_target = tool0.get("target", 0.0)
        bed_temp = bed.get("actual", 0.0)
        bed_target = bed.get("target", 0.0)

        job = j_data.get("job", {})
        progress_data = j_data.get("progress", {})
        filename = job.get("file", {}).get("name", "")
        progress = round(progress_data.get("completion", 0.0) or 0.0, 1)
        elapsed_min = round((progress_data.get("printTime", 0) or 0) / 60.0, 1)

        text = (
            f"🖨️ **Anycubic Kobra - OctoPrint Telemetry**\n\n"
            f"• **Trạng thái:** `{state}`\n"
            f"• **Đầu phun (Nozzle):** `{nozzle_temp}°C` / `{nozzle_target}°C`\n"
            f"• **Bàn nhiệt (Bed):** `{bed_temp}°C` / `{bed_target}°C`\n"
            f"• **Tiến độ in:** `{progress}%` (Đã in: `{elapsed_min}m`)\n"
            f"• **Tệp đang in:** `{filename or 'Không có'}`"
        )
        return PrinterStatus(
            is_connected=True,
            printer_name="Anycubic Kobra",
            state=state,
            nozzle_temp=nozzle_temp,
            nozzle_target=nozzle_target,
            bed_temp=bed_temp,
            bed_target=bed_target,
            progress_percent=progress,
            time_elapsed_min=elapsed_min,
            time_remaining_min=0.0,
            file_name=filename,
            text=text,
        )
