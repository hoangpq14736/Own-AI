import pytest
import struct
from pathlib import Path
from core.dfam_analyzer import analyze_stl_file, is_binary_stl


def create_mock_binary_stl(tmp_path: Path, filename="cube.stl", scale=20.0) -> Path:
    """Creates a valid 2-triangle flat rectangular surface in binary STL."""
    file_path = tmp_path / filename
    header = b"Mock Binary STL for Anycubic Kobra testing" + b"\x00" * (80 - 42)
    # 2 triangles
    num_triangles = 2

    # Triangle 1: Normal pointing up (0, 0, 1), vertices (0,0,0), (scale,0,0), (0,scale,0)
    t1 = struct.pack(
        "<3f3f3f3fH",
        0.0, 0.0, 1.0,  # normal
        0.0, 0.0, 0.0,  # v1
        scale, 0.0, 0.0,  # v2
        0.0, scale, 0.0,  # v3
        0,
    )
    # Triangle 2: Normal pointing up (0, 0, 1), vertices (scale,scale,0), (0,scale,0), (scale,0,0)
    t2 = struct.pack(
        "<3f3f3f3fH",
        0.0, 0.0, 1.0,  # normal
        scale, scale, 0.0,  # v1
        0.0, scale, 0.0,  # v2
        scale, 0.0, 0.0,  # v3
        0,
    )

    with open(file_path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", num_triangles))
        f.write(t1)
        f.write(t2)

    return file_path


def create_mock_ascii_stl(tmp_path: Path, filename="triangle.stl") -> Path:
    file_path = tmp_path / filename
    content = """solid test_ascii
facet normal 0.0 0.0 1.0
  outer loop
    vertex 0.0 0.0 0.0
    vertex 50.0 0.0 0.0
    vertex 0.0 50.0 0.0
  endloop
endfacet
endsolid test_ascii
"""
    file_path.write_text(content, encoding="utf-8")
    return file_path


def test_is_binary_stl(tmp_path):
    bin_file = create_mock_binary_stl(tmp_path)
    asc_file = create_mock_ascii_stl(tmp_path)

    with open(bin_file, "rb") as f:
        assert is_binary_stl(f.read()) is True

    with open(asc_file, "rb") as f:
        assert is_binary_stl(f.read()) is False


def test_analyze_binary_stl_success(tmp_path):
    bin_file = create_mock_binary_stl(tmp_path, scale=30.0)
    report = analyze_stl_file(bin_file, bed_x=220.0, bed_y=220.0, bed_z=250.0)

    assert report.triangles == 2
    assert report.size_x == 30.0
    assert report.size_y == 30.0
    assert report.fits_bed is True
    assert "Vừa vặn bàn in" in report.text
    assert "Anycubic Kobra" in report.text


def test_analyze_ascii_stl_success(tmp_path):
    asc_file = create_mock_ascii_stl(tmp_path)
    report = analyze_stl_file(asc_file)

    assert report.triangles == 1
    assert report.size_x == 50.0
    assert report.fits_bed is True
    assert "Anycubic Kobra" in report.text


def test_analyze_oversized_model(tmp_path):
    # Model larger than Kobra bed (e.g. 300mm on X)
    oversized_file = create_mock_binary_stl(tmp_path, filename="big.stl", scale=300.0)
    report = analyze_stl_file(oversized_file, bed_x=220.0, bed_y=220.0, bed_z=250.0)

    assert report.fits_bed is False
    assert "Vượt quá bàn in Anycubic Kobra" in report.text


def test_analyze_missing_file(tmp_path):
    missing_file = tmp_path / "non_existent.stl"
    report = analyze_stl_file(missing_file)
    assert report.error == "File not found"
    assert "Không tìm thấy tệp" in report.text
