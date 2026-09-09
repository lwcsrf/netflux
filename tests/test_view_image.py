import base64
from dataclasses import FrozenInstanceError
import importlib
from io import BytesIO
from pathlib import Path
from random import Random
from unittest.mock import patch

from PIL import Image, ImageCms
import pytest

from ..func_lib import view_image
from ..func_lib.view_image import ImageResult
from ..runtime import Runtime


@pytest.fixture
def image_module():
    return Image


def assert_prepared_image(result, path, source_data):
    """Check the delivered encoding as well as the immutable source snapshot."""
    assert isinstance(result, ImageResult)
    assert result.source_path == str(path.resolve())
    assert path.read_bytes() == source_data
    assert base64.b64decode(result.base64_data, validate=True) == result.data
    assert result.mime_type in result.status
    assert f"{result.width}x{result.height}" in result.status
    assert f"{len(result.data)} bytes" in result.status
    assert result.base64_data not in result.status
    with Image.open(BytesIO(result.data)) as decoded:
        assert result.mime_type == {
            "PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp",
        }[decoded.format]
        assert decoded.size == (result.width, result.height)
        assert getattr(decoded, "n_frames", 1) == 1
        decoded.load()
        return decoded.copy()


@pytest.fixture
def image_path(image_module, tmp_path):
    path = tmp_path / "image.png"
    image_module.new("RGB", (12, 8), "red").save(path)
    return path


@pytest.mark.parametrize("fmt,mime", [
    ("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp"),
])
def test_runtime_returns_validated_snapshot(image_module, tmp_path, fmt, mime):
    # Identify content, regardless of extension; retain it after the file changes.
    path = tmp_path / "misleading.txt"
    image_module.new("RGB", (12, 8), "red").save(path, format=fmt)
    data = path.read_bytes()
    runtime = Runtime([view_image], client_factories={})
    node = runtime.invoke(None, view_image, {"path": str(path)})
    assert node.done.wait(5)
    result = node.result()
    path.write_bytes(b"overwritten")
    path.unlink()

    assert isinstance(result, ImageResult)
    assert (result.mime_type, result.width, result.height) == (mime, 12, 8)
    assert result.source_path == str(path.resolve())
    assert result.data == data
    assert base64.b64decode(result.base64_data, validate=True) == data
    assert runtime.get_view(node.id).outputs is result
    assert result.base64_data not in str(result)
    assert result.base64_data not in repr(result)
    assert "12x8" in str(result)
    assert result.status.endswith("; unchanged.")
    with pytest.raises(FrozenInstanceError):
        result.width = 1


@pytest.mark.parametrize("kind", ["missing", "directory", "unreadable"])
def test_file_errors_raise(image_path, tmp_path, kind):
    if kind == "missing":
        path = tmp_path / "missing.png"
        error = FileNotFoundError
    elif kind == "directory":
        path, error = tmp_path, ValueError
    else:
        with patch.object(Path, "open", side_effect=PermissionError("unreadable")):
            with pytest.raises(PermissionError, match="unreadable"):
                view_image.callable(None, path=str(image_path))
        return
    with pytest.raises(error) as exc_info:
        view_image.callable(None, path=str(path))
    if kind == "missing":
        assert str(exc_info.value) == "no file at that absolute filepath"


@pytest.mark.parametrize("contents", [b"", b"not an image", b"\x89PNG\r\n\x1a\n"])
def test_invalid_data_raises(image_module, tmp_path, contents):
    path = tmp_path / "invalid.png"
    path.write_bytes(contents)
    with pytest.raises(ValueError, match="Cannot load image"):
        view_image.callable(None, path=str(path))


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_truncated_image_raises(image_module, tmp_path, fmt):
    path = tmp_path / "truncated"
    image_module.new("RGB", (40, 30), "red").save(path, format=fmt)
    path.write_bytes(path.read_bytes()[:-10])
    with pytest.raises(ValueError, match="Cannot load image"):
        view_image.callable(None, path=str(path))


@pytest.mark.parametrize("fmt", ["GIF", "BMP", "DIB", "TIFF", "ICO", "AVIF", "JPEG2000", "TGA", "QOI", "PCX"])
def test_other_formats_are_converted(image_module, tmp_path, fmt):
    image_module.init()
    if fmt not in image_module.OPEN or fmt not in image_module.SAVE:
        pytest.skip(f"Pillow build does not support {fmt}")
    path = tmp_path / "image.png"
    size = (32, 32) if fmt == "ICO" else (12, 8)
    image_module.new("RGB", size, "red").save(path, format=fmt, **({"sizes": [size]} if fmt == "ICO" else {}))
    source_data = path.read_bytes()
    with image_module.open(path) as original:
        expected_pixels = original.convert("RGBA").tobytes()

    result = view_image.callable(None, path=str(path))

    decoded = assert_prepared_image(result, path, source_data)
    assert result.mime_type == "image/png"
    assert decoded.size == size
    assert decoded.convert("RGBA").tobytes() == expected_pixels
    assert f"re-encoded {fmt} -> PNG" in result.status


@pytest.mark.parametrize("fmt", ["PNG", "WEBP", "GIF", "TIFF", "AVIF"])
def test_animation_and_multipage_images_raise(image_module, tmp_path, fmt):
    path = tmp_path / "animated"
    image_module.new("RGB", (12, 8), "red").save(
        path, format=fmt, save_all=True,
        append_images=[image_module.new("RGB", (12, 8), "blue")], duration=100,
    )
    source_data = path.read_bytes()
    with image_module.open(path) as original:
        assert original.n_frames == 2
    runtime = Runtime([view_image], client_factories={})
    node = runtime.invoke(None, view_image, {"path": str(path)})
    assert node.done.wait(5)
    with pytest.raises(ValueError, match="Animated|multipage"):
        node.result()
    assert path.read_bytes() == source_data


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_exif_orientation_is_applied(image_module, tmp_path, fmt):
    path = tmp_path / "rotated"
    original = image_module.new("RGB", (12, 8), "red")
    original.paste("blue", (0, 0, 6, 8))
    exif = image_module.Exif()
    exif[274] = 6
    original.save(path, format=fmt, exif=exif)
    source_data = path.read_bytes()
    with image_module.open(path) as encoded:
        expected = encoded.transpose(image_module.Transpose.ROTATE_270).convert("RGB")

    result = view_image.callable(None, path=str(path))

    decoded = assert_prepared_image(result, path, source_data)
    assert decoded.size == (8, 12)
    assert decoded.convert("RGB").tobytes() == expected.tobytes()
    assert decoded.getexif().get(274, 1) == 1
    assert "re-encoded" in result.status
    assert "unchanged" not in result.status


@pytest.mark.parametrize("kind", ["rgba", "palette"])
def test_conversion_preserves_transparency(image_module, tmp_path, kind):
    path = tmp_path / "transparent"
    if kind == "rgba":
        original = image_module.new("RGBA", (12, 8), (255, 0, 0, 0))
        original.paste((0, 0, 255, 128), (6, 0, 12, 8))
        original.save(path, format="TIFF")
    else:
        original = image_module.new("P", (12, 8), 0)
        original.putpalette([255, 0, 0, 0, 0, 255] + [0] * 762)
        original.paste(1, (6, 0, 12, 8))
        original.save(path, format="GIF", transparency=0)
    source_data = path.read_bytes()
    with image_module.open(path) as encoded:
        expected_pixels = encoded.convert("RGBA").tobytes()

    result = view_image.callable(None, path=str(path))

    decoded = assert_prepared_image(result, path, source_data)
    assert decoded.convert("RGBA").tobytes() == expected_pixels
    assert decoded.convert("RGBA").getpixel((0, 0))[3] == 0
    assert decoded.convert("RGBA").getpixel((11, 0))[3] == (128 if kind == "rgba" else 255)


def test_16_bit_grayscale_preserves_transparency_during_normalization(tmp_path):
    path = tmp_path / "transparent16.png"
    original = Image.frombytes("I;16", (3, 1), b"\x00\x00\x00\x80\xff\xff")
    exif = Image.Exif()
    exif[274] = 3
    original.save(path, transparency=32768, exif=exif)
    source_data = path.read_bytes()

    result = view_image.callable(None, path=str(path))

    decoded = assert_prepared_image(result, path, source_data).convert("RGBA")
    assert decoded.getpixel((0, 0)) == (255, 255, 255, 255)
    assert decoded.getpixel((1, 0)) == (127, 127, 127, 0)
    assert decoded.getpixel((2, 0)) == (0, 0, 0, 255)


def test_cmyk_tiff_is_converted_to_rgb(image_module, tmp_path):
    path = tmp_path / "cmyk.tiff"
    image_module.new("CMYK", (12, 8), (0, 255, 255, 0)).save(path)
    source_data = path.read_bytes()

    result = view_image.callable(None, path=str(path))

    decoded = assert_prepared_image(result, path, source_data)
    assert decoded.convert("RGB").getpixel((0, 0)) == (255, 0, 0)
    assert "color CMYK -> RGB" in result.status


@pytest.mark.parametrize("size,pixel_limit", [
    ((400, 200), 5_000), ((200, 400), 5_000), ((1_000, 1), 100), ((1, 1_000), 100),
])
def test_pixel_limit_resizes_with_aspect_ratio(image_module, tmp_path, monkeypatch, size, pixel_limit):
    module = importlib.import_module("netflux.func_lib.view_image")
    monkeypatch.setattr(module, "MAX_PIXELS", pixel_limit)
    path = tmp_path / "large.png"
    image_module.new("RGB", size, "red").save(path)
    source_data = path.read_bytes()

    result = view_image.callable(None, path=str(path))

    decoded = assert_prepared_image(result, path, source_data)
    assert 0 < result.width <= size[0]
    assert 0 < result.height <= size[1]
    assert result.width * result.height <= pixel_limit
    assert len(result.data) <= module.MAX_BYTES
    assert abs(result.width * size[1] - result.height * size[0]) <= max(size)
    assert decoded.convert("RGB").getpixel((0, 0)) == (255, 0, 0)
    assert f"downsized {size[0]}x{size[1]}" in result.status


@pytest.mark.parametrize("pixel_limit", [2_000_000, 700])
def test_byte_limit_reduces_noisy_image_and_retains_alpha(image_module, tmp_path, monkeypatch, pixel_limit):
    module = importlib.import_module("netflux.func_lib.view_image")
    monkeypatch.setattr(module, "MAX_BYTES", 2_000)
    monkeypatch.setattr(module, "MAX_PIXELS", pixel_limit)
    path = tmp_path / "noisy.png"
    size = (128, 96)
    original = image_module.frombytes("RGB", size, Random(0).randbytes(size[0] * size[1] * 3))
    alpha_row = bytes([0] * 42 + [128] * 44 + [255] * 42)
    original.putalpha(image_module.frombytes("L", size, alpha_row * size[1]))
    original.save(path)
    source_data = path.read_bytes()
    assert len(source_data) > module.MAX_BYTES

    result = view_image.callable(None, path=str(path))

    decoded = assert_prepared_image(result, path, source_data)
    assert 0 < result.width <= size[0]
    assert 0 < result.height <= size[1]
    assert result.width * result.height < size[0] * size[1]
    assert result.width * result.height <= pixel_limit
    assert len(result.data) <= module.MAX_BYTES
    assert abs(result.width * size[1] - result.height * size[0]) <= max(size)
    assert decoded.convert("RGBA").getchannel("A").getextrema() == (0, 255)
    assert "downsized" in result.status
    assert "lossy" not in result.status


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_exact_limits_preserve_original_bytes(image_module, tmp_path, monkeypatch, fmt):
    module = importlib.import_module("netflux.func_lib.view_image")
    path = tmp_path / "exact"
    image_module.new("RGB", (12, 8), "red").save(path, format=fmt)
    source_data = path.read_bytes()
    monkeypatch.setattr(module, "MAX_BYTES", len(source_data))
    monkeypatch.setattr(module, "MAX_PIXELS", 12 * 8)
    monkeypatch.setattr(module, "MAX_SIDE", 12)

    result = view_image.callable(None, path=str(path))

    assert_prepared_image(result, path, source_data)
    assert result.data == source_data


@pytest.mark.parametrize("size", [(1, 1), (1, 2), (2, 1), (3, 2)])
def test_impossible_byte_limit_raises(image_module, tmp_path, monkeypatch, size):
    module = importlib.import_module("netflux.func_lib.view_image")
    monkeypatch.setattr(module, "MAX_BYTES", 1)
    path = tmp_path / "single_pixel.png"
    image_module.new("RGB", size, "red").save(path)
    source_data = path.read_bytes()
    with pytest.raises(ValueError, match="byte limit|fit|limit"):
        view_image.callable(None, path=str(path))
    assert path.read_bytes() == source_data


def test_pillow_decompression_bomb_protection_remains_enabled(image_module, image_path, monkeypatch):
    monkeypatch.setattr(image_module, "MAX_IMAGE_PIXELS", 1)
    with pytest.raises(ValueError, match="decompression bomb"):
        view_image.callable(None, path=str(image_path))


def test_pillow_decompression_bomb_warning_remains_enabled(image_module, image_path, monkeypatch):
    source_data = image_path.read_bytes()
    monkeypatch.setattr(image_module, "MAX_IMAGE_PIXELS", 50)
    with pytest.warns(image_module.DecompressionBombWarning, match="decompression bomb"):
        result = view_image.callable(None, path=str(image_path))
        assert_prepared_image(result, image_path, source_data)
    assert result.data == source_data


def test_encoding_failure_is_a_failed_node(image_path):
    with patch("netflux.func_lib.view_image.base64.b64encode", side_effect=ValueError("encoding failed")):
        runtime = Runtime([view_image], client_factories={})
        node = runtime.invoke(None, view_image, {"path": str(image_path)})
        assert node.done.wait(5)
        with pytest.raises(ValueError, match="encoding failed"):
            node.result()


@pytest.mark.parametrize("path", [
    "image.png", "./image.png", "missing.png", "", "C:image.png", r"\image.png",
])
def test_relative_paths_are_rejected(image_path, monkeypatch, path):
    monkeypatch.chdir(image_path.parent)
    with pytest.raises(FileNotFoundError) as exc_info:
        view_image.callable(None, path=path)
    assert str(exc_info.value) == "no file at that absolute filepath"


@pytest.mark.parametrize("fmt,mode", [
    ("PNG", "P"), ("PNG", "L"), ("PNG", "RGBA"), ("PNG", "I;16"),
    ("JPEG", "L"), ("JPEG", "CMYK"), ("WEBP", "RGBA"),
])
def test_native_modes_do_not_reencode(tmp_path, monkeypatch, fmt, mode):
    path = tmp_path / "native"
    Image.new(mode, (12, 8)).save(path, format=fmt)
    source_data = path.read_bytes()
    with patch.object(Image.Image, "save", side_effect=AssertionError("must not re-encode")):
        result = view_image.callable(None, path=str(path))
    assert_prepared_image(result, path, source_data)
    assert result.data == source_data
    assert result.status.endswith("; unchanged.")


def test_native_icc_profile_is_untouched(tmp_path):
    path = tmp_path / "profile.png"
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.new("RGB", (12, 8), "red").save(path, icc_profile=profile)
    source_data = path.read_bytes()
    result = view_image.callable(None, path=str(path))
    assert result.data == source_data
    assert "unchanged" in result.status


@pytest.mark.parametrize("size", [(2001, 2), (2, 2001)])
def test_long_edge_limit_applies_even_below_pixel_limit(tmp_path, size):
    path = tmp_path / "panorama.png"
    Image.new("RGB", size, "red").save(path)
    source_data = path.read_bytes()
    result = view_image.callable(None, path=str(path))
    assert_prepared_image(result, path, source_data)
    assert max(result.width, result.height) == 2000
    assert min(result.width, result.height) == 2
    assert "downsized" in result.status


def test_real_pixel_limit(tmp_path):
    path = tmp_path / "2mp.png"
    Image.new("L", (1600, 1251), "white").save(path)
    source_data = path.read_bytes()
    result = view_image.callable(None, path=str(path))
    assert_prepared_image(result, path, source_data)
    assert result.width * result.height <= 2_000_000
    assert max(result.width, result.height) <= 2000
    assert "downsized 1600x1251" in result.status


@pytest.mark.parametrize("size", [(2000, 1000), (1000, 2000)])
def test_exact_real_dimension_limits_preserve_original_bytes(tmp_path, size):
    path = tmp_path / "exact_dimensions.png"
    Image.new("RGB", size, "red").save(path)
    source_data = path.read_bytes()

    result = view_image.callable(None, path=str(path))

    assert_prepared_image(result, path, source_data)
    assert result.data == source_data
    assert (result.width, result.height) == size
    assert result.status.endswith("; unchanged.")


def test_antialiasing_filters_high_frequency_pixels(tmp_path, monkeypatch):
    module = importlib.import_module("netflux.func_lib.view_image")
    monkeypatch.setattr(module, "MAX_PIXELS", 16 * 16)
    path = tmp_path / "checkerboard.png"
    pixels = bytes(255 * ((x + y) % 2) for y in range(128) for x in range(128))
    Image.frombytes("L", (128, 128), pixels).save(path)
    source_data = path.read_bytes()
    result = view_image.callable(None, path=str(path))
    decoded = assert_prepared_image(result, path, source_data)
    assert decoded.size == (16, 16)
    # Nearest-neighbor would alias to black/white; low-pass filtering averages
    # the alternating black and white pixels to almost uniform middle gray.
    low, high = decoded.getextrema()
    assert 120 <= low <= high <= 135


def test_optimized_png_avoids_unnecessary_lossy_conversion(tmp_path, monkeypatch):
    module = importlib.import_module("netflux.func_lib.view_image")
    monkeypatch.setattr(module, "MAX_BYTES", 2000)
    path = tmp_path / "uncompressed.png"
    Image.new("RGB", (200, 100), "red").save(path, compress_level=0)
    source_data = path.read_bytes()
    assert len(source_data) > module.MAX_BYTES
    result = view_image.callable(None, path=str(path))
    assert_prepared_image(result, path, source_data)
    assert result.mime_type == "image/png"
    assert (result.width, result.height) == (200, 100)
    assert "re-encoded PNG -> PNG" in result.status
    assert "lossy" not in result.status


def test_lossy_jpeg_fallback_is_reported(tmp_path, monkeypatch):
    module = importlib.import_module("netflux.func_lib.view_image")
    monkeypatch.setattr(module, "MAX_BYTES", 25_000)
    path = tmp_path / "noise.png"
    Image.frombytes("RGB", (100, 100), Random(0).randbytes(30_000)).save(path)
    source_data = path.read_bytes()
    assert len(source_data) > module.MAX_BYTES
    result = view_image.callable(None, path=str(path))
    assert_prepared_image(result, path, source_data)
    assert result.mime_type == "image/jpeg"
    assert len(result.data) <= module.MAX_BYTES
    assert (result.width, result.height) == (100, 100)
    assert "re-encoded PNG -> JPEG (lossy JPEG)" in result.status


def test_real_byte_limit_caps_encoded_image_at_512_kib(tmp_path):
    path = tmp_path / "large-noise.png"
    Image.frombytes("RGB", (800, 600), Random(0).randbytes(800 * 600 * 3)).save(path)
    source_data = path.read_bytes()
    assert len(source_data) > 512 * 1024
    result = view_image.callable(None, path=str(path))
    assert_prepared_image(result, path, source_data)
    assert len(result.data) <= 512 * 1024
    assert len(result.base64_data) <= 4 * ((512 * 1024 + 2) // 3)
    assert result.width * result.height <= 2_000_000
    assert max(result.width, result.height) <= 2000
    assert "lossy JPEG" in result.status or "downsized" in result.status


@pytest.mark.parametrize("suffix", [".svg", ".txt"])
def test_svg_error_suggests_source_and_visual_inspection(tmp_path, suffix):
    path = tmp_path / f"drawing{suffix}"
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg"><circle r="1"/></svg>')
    source_data = path.read_bytes()
    with pytest.raises(ValueError, match="Read its source.*render it to PNG"):
        view_image.callable(None, path=str(path))
    assert path.read_bytes() == source_data
    assert "SVG" not in view_image.desc


def test_external_renderer_formats_are_rejected(tmp_path):
    path = tmp_path / "image.eps"
    Image.new("RGB", (12, 8), "red").save(path)
    with patch("PIL.EpsImagePlugin.Ghostscript", side_effect=AssertionError("external renderer")):
        with pytest.raises(ValueError, match="unsupported raster"):
            view_image.callable(None, path=str(path))


@pytest.mark.parametrize("mode", ["I", "F"])
def test_ambiguous_high_dynamic_range_is_not_silently_clipped(tmp_path, mode):
    path = tmp_path / "hdr.tiff"
    Image.new(mode, (12, 8), 1000).save(path)
    with pytest.raises(ValueError, match="Unsupported color mode.*export an 8-bit raster"):
        view_image.callable(None, path=str(path))


def test_reencoding_failure_is_a_failed_node(tmp_path):
    path = tmp_path / "convert.bmp"
    Image.new("RGB", (12, 8), "red").save(path)
    source_data = path.read_bytes()
    with patch("netflux.func_lib.view_image._encode_image", side_effect=OSError("encoder failed")):
        runtime = Runtime([view_image], client_factories={})
        node = runtime.invoke(None, view_image, {"path": str(path)})
        assert node.done.wait(5)
        with pytest.raises(ValueError, match="Cannot load image.*encoder failed"):
            node.result()
    assert path.read_bytes() == source_data
