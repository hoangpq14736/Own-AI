import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from core.printer_client import PrinterClient


@pytest.mark.asyncio
async def test_printer_unconfigured():
    client = PrinterClient(url="")
    assert client.is_configured is False
    status = await client.get_status()
    assert status.is_connected is False
    assert "Anycubic Kobra - Trạng Thái Offline" in status.text
    assert "PRINTER_URL" in status.text


@pytest.mark.asyncio
async def test_printer_moonraker_success():
    client = PrinterClient(url="http://192.168.1.50:7125", api_type="moonraker")
    mock_json = {
        "result": {
            "status": {
                "print_stats": {
                    "state": "printing",
                    "filename": "benchy.gcode",
                    "print_duration": 1800.0,
                    "total_duration": 3600.0,
                },
                "extruder": {"temperature": 209.8, "target": 210.0},
                "heater_bed": {"temperature": 60.1, "target": 60.0},
            }
        }
    }

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value=mock_json)
    mock_resp.__aenter__.return_value = mock_resp
    mock_resp.__aexit__.return_value = None

    mock_session = MagicMock()
    mock_session.get.return_value = mock_resp
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__.return_value = None

    with patch("aiohttp.ClientSession", return_value=mock_session):
        status = await client.get_status()
        assert status.is_connected is True
        assert status.state == "Printing"
        assert status.nozzle_temp == 209.8
        assert status.bed_temp == 60.1
        assert status.progress_percent == 50.0
        assert "Anycubic Kobra - Klipper Telemetry" in status.text
        assert "50.0%" in status.text


@pytest.mark.asyncio
async def test_printer_connection_error():
    client = PrinterClient(url="http://127.0.0.1:9999", api_type="moonraker")
    with patch("aiohttp.ClientSession.get", side_effect=Exception("Connection refused")):
        status = await client.get_status()
        assert status.is_connected is False
        assert "Không thể kết nối" in status.text
