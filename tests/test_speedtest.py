import pytest
from unittest.mock import MagicMock, patch
from core.speedtest_manager import _run_speedtest_sync, run_vps_speedtest

def test_run_speedtest_mock():
    mock_st_instance = MagicMock()
    mock_st_instance.download.return_value = 150_000_000 # 150 Mbps
    mock_st_instance.upload.return_value = 80_000_000    # 80 Mbps
    mock_st_instance.results.dict.return_value = {
        "ping": 12.5,
        "server": {
            "name": "Ho Chi Minh City",
            "country": "Vietnam",
            "sponsor": "VNPT"
        }
    }

    with patch("speedtest.Speedtest", return_value=mock_st_instance):
        result = _run_speedtest_sync()
        assert result.error is None
        assert result.download_mbps == 150.0
        assert result.upload_mbps == 80.0
        assert result.ping_ms == 12.5
        assert "VNPT" in result.text
        assert "150.0 Mbps" in result.text

def test_run_speedtest_error():
    with patch("speedtest.Speedtest", side_effect=Exception("Network unreachable")):
        result = _run_speedtest_sync()
        assert result.error == "Network unreachable"
        assert "Không thể thực hiện" in result.text
