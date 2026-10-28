"""
VPS Network Speedtest Diagnostics Module.
Measures live bandwidth (Download/Upload/Ping) on host VPS per ECC standards.
"""

from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeedtestResult:
    download_mbps: float
    upload_mbps: float
    ping_ms: float
    server_name: str
    server_country: str
    sponsor: str
    text: str
    error: Optional[str] = None


def _run_speedtest_sync() -> SpeedtestResult:
    """Synchronous execution of speedtest-cli."""
    try:
        import speedtest
    except ImportError:
        return SpeedtestResult(
            download_mbps=0.0,
            upload_mbps=0.0,
            ping_ms=0.0,
            server_name="",
            server_country="",
            sponsor="",
            text="⚠️ Thư viện `speedtest-cli` chưa được cài đặt.",
            error="speedtest-cli not installed"
        )

    try:
        st = speedtest.Speedtest()
        st.get_best_server()
        download_bps = st.download()
        upload_bps = st.upload()
        results = st.results.dict()

        download_mbps = round(download_bps / 1_000_000, 2)
        upload_mbps = round(upload_bps / 1_000_000, 2)
        ping_ms = round(results.get("ping", 0.0), 2)
        server = results.get("server", {})
        server_name = server.get("name", "Unknown")
        server_country = server.get("country", "")
        sponsor = server.get("sponsor", "")

        formatted_text = (
            f"🚀 **KẾT QUẢ ĐO TỐC ĐỘ MẠNG VPS (SPEEDTEST)**\n\n"
            f"• 📥 **Tốc độ Tải xuống (Download):** `{download_mbps} Mbps`\n"
            f"• 📤 **Tốc độ Tải lên (Upload):** `{upload_mbps} Mbps`\n"
            f"• 📶 **Độ trễ (Ping):** `{ping_ms} ms`\n"
            f"• 🏢 **Máy chủ đo:** `{sponsor}` ({server_name}, {server_country})\n"
        )

        return SpeedtestResult(
            download_mbps=download_mbps,
            upload_mbps=upload_mbps,
            ping_ms=ping_ms,
            server_name=server_name,
            server_country=server_country,
            sponsor=sponsor,
            text=formatted_text,
            error=None
        )

    except Exception as e:
        logger.error(f"Speedtest execution failed: {e}")
        return SpeedtestResult(
            download_mbps=0.0,
            upload_mbps=0.0,
            ping_ms=0.0,
            server_name="",
            server_country="",
            sponsor="",
            text=f"❌ Không thể thực hiện kiểm tra tốc độ mạng: {str(e)}",
            error=str(e)
        )


async def run_vps_speedtest() -> SpeedtestResult:
    """Asynchronously runs the speedtest in an offloaded thread."""
    return await asyncio.to_thread(_run_speedtest_sync)
