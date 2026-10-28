import pytest
from pathlib import Path
from core.gcode_parser import parse_gcode_file


def test_parse_orcaslicer_gcode(tmp_path):
    gcode_file = tmp_path / "benchy.gcode"
    content = """
; OrcaSlicer 2.1.1
; estimated printing time (normal mode) = 1h 24m 12s
; filament used [mm] = 4850.5
; filament used [g] = 14.6
; filament_type = PLA
; nozzle_temperature = 210
; bed_temperature = 60
; layer_height = 0.20
; total layers count = 240
G28 ; Home
M109 S210
"""
    gcode_file.write_text(content, encoding="utf-8")
    report = parse_gcode_file(gcode_file)

    assert report.slicer == "OrcaSlicer"
    assert report.print_time == "1h 24m 12s"
    assert report.filament_length_m == 4.85
    assert report.filament_weight_g == 14.6
    assert report.filament_type == "PLA"
    assert report.nozzle_temp == 210.0
    assert report.bed_temp == 60.0
    assert report.layer_height == 0.20
    assert report.total_layers == 240
    assert "Sẵn sàng copy vào thẻ nhớ/USB" in report.text


def test_parse_cura_gcode(tmp_path):
    gcode_file = tmp_path / "cura_model.gcode"
    content = """
;Generated with Cura_SteamEngine 5.6.0
;TIME:3600
;Filament used: 3.2m
;Layer height: 0.28
;LAYER_COUNT:100
M104 S200
M140 S55
"""
    gcode_file.write_text(content, encoding="utf-8")
    report = parse_gcode_file(gcode_file)

    assert report.slicer == "UltiMaker Cura"
    assert report.print_time == "1h 0m"
    assert report.filament_length_m == 3.2
    assert report.layer_height == 0.28
    assert report.total_layers == 100
    assert report.nozzle_temp == 200.0
    assert report.bed_temp == 55.0


def test_parse_missing_gcode(tmp_path):
    missing_file = tmp_path / "missing.gcode"
    report = parse_gcode_file(missing_file)
    assert report.error == "File not found"
    assert "Không tìm thấy tệp" in report.text
