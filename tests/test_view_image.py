"""Image snapshots, conversions, output limits, and container regressions."""

import base64
from dataclasses import FrozenInstanceError
import importlib
from io import BytesIO
from random import Random
from struct import pack
from unittest.mock import patch

from PIL import Image, ImageCms, PngImagePlugin
import pytest

from ..func_lib import view_image
from ..func_lib.view_image import ImageResult
from ..runtime import Runtime


image_impl = importlib.import_module("netflux.func_lib.view_image")


def assert_prepared_image(result, path, source_data):
    """Check the delivered encoding as well as the immutable source snapshot."""
    assert isinstance(result, ImageResult)
    assert result.source_path == str(path.resolve())
    assert len(result.data) <= image_impl.MAX_BYTES
    assert result.width * result.height <= image_impl.MAX_PIXELS
    assert max(result.width, result.height) <= image_impl.MAX_SIDE
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
def image_path(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (12, 8), "red").save(path)
    return path


@pytest.mark.parametrize("fmt,mode,mime", [
    ("PNG", "RGBA", "image/png"),
    ("PNG", "P", "image/png"),
    ("PNG", "I;16", "image/png"),
    ("JPEG", "CMYK", "image/jpeg"),
    ("JPEG", "L", "image/jpeg"),
    ("WEBP", "RGB", "image/webp"),
    ("WEBP", "RGBA", "image/webp"),
])
def test_runtime_preserves_native_snapshot_at_exact_limits(tmp_path, monkeypatch, fmt, mode, mime):
    # Detect content regardless of extension; preserve native modes and metadata.
    path = tmp_path / "misleading.txt"
    options = {}
    if fmt == "PNG" and mode == "RGBA":
        options["icc_profile"] = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.new(mode, (12, 8)).save(path, format=fmt, **options)
    data = path.read_bytes()
    monkeypatch.setattr(image_impl, "MAX_BYTES", len(data))
    monkeypatch.setattr(image_impl, "MAX_PIXELS", 12 * 8)
    monkeypatch.setattr(image_impl, "MAX_SIDE", 12)

    runtime = Runtime([view_image], client_factories={})
    node = runtime.invoke(None, view_image, {"path": str(path)})
    assert node.done.wait(5)
    result = node.result()

    assert_prepared_image(result, path, data)
    path.unlink()
    assert (result.mime_type, result.width, result.height) == (mime, 12, 8)
    assert result.data == data
    assert base64.b64decode(result.base64_data, validate=True) == data
    assert runtime.get_view(node.id).outputs is result
    assert str(result) == result.status
    assert result.base64_data not in repr(result)
    assert result.status.endswith("; unchanged.")
    with pytest.raises(FrozenInstanceError):
        result.width = 1


def test_existing_relative_path_is_rejected(image_path, monkeypatch):
    monkeypatch.chdir(image_path.parent)
    with pytest.raises(FileNotFoundError, match="no file at that absolute filepath"):
        view_image.callable(None, path=image_path.name)


@pytest.mark.parametrize("kind,error,message", [
    ("missing", FileNotFoundError, "no file at that absolute filepath"),
    ("directory", ValueError, "not a regular file"),
])
def test_file_errors_raise(tmp_path, kind, error, message):
    path = tmp_path / "missing.png" if kind == "missing" else tmp_path
    with pytest.raises(error, match=message):
        view_image.callable(None, path=str(path))


def test_invalid_data_raises(tmp_path):
    path = tmp_path / "invalid.png"
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Cannot load image"):
        view_image.callable(None, path=str(path))


@pytest.mark.parametrize("fmt,missing_bytes", [("PNG", 1), ("JPEG", 10)])
def test_truncated_image_raises(tmp_path, fmt, missing_bytes):
    path = tmp_path / "truncated"
    Image.new("RGB", (40, 30), "red").save(path, format=fmt)
    # Pillow.verify() overlooks a missing final PNG checksum byte.
    source_data = path.read_bytes()[:-missing_bytes]
    path.write_bytes(source_data)
    with pytest.raises(ValueError, match="Cannot load image"):
        view_image.callable(None, path=str(path))
    assert path.read_bytes() == source_data


@pytest.mark.parametrize("suffix", [".svg", ".txt"])
def test_svg_error_suggests_source_and_visual_inspection(tmp_path, suffix):
    path = tmp_path / f"drawing{suffix}"
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg"><circle r="1"/></svg>')
    source_data = path.read_bytes()
    with pytest.raises(ValueError, match="Read its source.*render it to PNG"):
        view_image.callable(None, path=str(path))
    assert path.read_bytes() == source_data


def test_external_renderer_formats_are_rejected(tmp_path):
    path = tmp_path / "image.eps"
    Image.new("RGB", (12, 8), "red").save(path)
    with patch("PIL.EpsImagePlugin.Ghostscript", side_effect=AssertionError("external renderer")):
        with pytest.raises(ValueError, match="unsupported raster"):
            view_image.callable(None, path=str(path))


def test_concatenated_ppm_images_are_rejected(tmp_path):
    # PNM permits concatenated image sequences, but Pillow exposes only one.
    source = b"P6\n1 1\n255\n\xff\x00\x00P6\n1 1\n255\n\x00\x00\xff"
    path = tmp_path / "multiple.ppm"
    path.write_bytes(source)
    with Image.open(path) as image:
        assert image.format == "PPM"
        assert getattr(image, "n_frames", 1) == 1

    with pytest.raises(ValueError, match="unsupported raster"):
        view_image.callable(None, path=str(path))

    assert path.read_bytes() == source


@pytest.mark.parametrize("fmt", ["GIF", "TIFF"])
def test_animation_and_multipage_images_raise(tmp_path, fmt):
    path = tmp_path / "animated"
    Image.new("RGB", (12, 8), "red").save(
        path, format=fmt, save_all=True,
        append_images=[Image.new("RGB", (12, 8), "blue")], duration=100,
    )
    source_data = path.read_bytes()
    with Image.open(path) as original:
        assert original.n_frames == 2
    runtime = Runtime([view_image], client_factories={})
    node = runtime.invoke(None, view_image, {"path": str(path)})
    assert node.done.wait(5)
    with pytest.raises(ValueError, match="Animated|multipage"):
        node.result()
    assert path.read_bytes() == source_data


def test_single_frame_apng_container_is_rejected(tmp_path):
    metadata = PngImagePlugin.PngInfo()
    metadata.add(b"acTL", pack(">II", 1, 0))
    metadata.add(b"fcTL", pack(">IIIIIHHBB", 0, 20, 12, 0, 0, 100, 1000, 0, 0))
    path = tmp_path / "single_frame.apng"
    Image.new("RGB", (20, 12), "red").save(path, format="PNG", pnginfo=metadata)
    source = path.read_bytes()
    with Image.open(path) as image:
        assert image.n_frames == 1
        assert not image.is_animated
        assert image.custom_mimetype == "image/apng"
        image.load()

    with pytest.raises(ValueError, match="[Aa]nimat"):
        view_image.callable(None, path=str(path))

    assert path.read_bytes() == source


def test_single_frame_webp_animation_container_is_rejected(tmp_path):
    # Pillow defines is_animated as n_frames > 1, even when VP8X marks animation.
    def chunk(tag, data):
        return pack("<4sI", tag, len(data)) + data + b"\x00" * (len(data) % 2)

    with BytesIO() as buffer:
        Image.new("RGB", (20, 12), "red").save(buffer, format="WEBP", lossless=True)
        frame_payload = buffer.getvalue()[12:]
    dimensions = (19).to_bytes(3, "little") + (11).to_bytes(3, "little")
    frame = b"\x00" * 6 + dimensions + (100).to_bytes(3, "little") + b"\x00" + frame_payload
    body = (
        b"WEBP"
        + chunk(b"VP8X", b"\x02\x00\x00\x00" + dimensions)
        + chunk(b"ANIM", pack("<IH", 0, 1))
        + chunk(b"ANMF", frame)
    )
    source = b"RIFF" + pack("<I", len(body)) + body
    path = tmp_path / "single_frame_animation.webp"
    path.write_bytes(source)
    with Image.open(path) as image:
        assert image.n_frames == 1
        assert not image.is_animated
        assert image.info["loop"] == 1
        image.load()

    with pytest.raises(ValueError, match="[Aa]nimat"):
        view_image.callable(None, path=str(path))

    assert path.read_bytes() == source


def test_single_frame_looping_gif_is_rejected(tmp_path):
    path = tmp_path / "looping.gif"
    Image.new("RGB", (12, 8), "red").save(path, save_all=True, duration=100, loop=0)
    with Image.open(path) as image:
        assert image.n_frames == 1
        assert image.info["loop"] == 0
    with pytest.raises(ValueError, match="Animated"):
        view_image.callable(None, path=str(path))


def test_multiple_icon_entries_with_same_dimensions_are_rejected(tmp_path):
    # ICO can hold several distinct images at the same resolution. Pillow's
    # sizes() returns a set and therefore cannot establish a single-image file.
    payloads = []
    for color in ("red", "blue"):
        with BytesIO() as buffer:
            Image.new("RGB", (16, 16), color).save(buffer, format="PNG")
            payloads.append(buffer.getvalue())
    offset = 6 + 16 * len(payloads)
    entries = []
    for data in payloads:
        entries.append(pack("<BBBBHHII", 16, 16, 0, 0, 1, 32, len(data), offset))
        offset += len(data)
    source = pack("<HHH", 0, 1, len(payloads)) + b"".join(entries) + b"".join(payloads)
    path = tmp_path / "same_size_images.ico"
    path.write_bytes(source)
    with Image.open(path) as image:
        assert image.ico.nb_items == 2
        assert image.ico.sizes() == {(16, 16)}

    with pytest.raises(ValueError, match="[Mm]ulti|[Aa]nimat"):
        view_image.callable(None, path=str(path))

    assert path.read_bytes() == source


@pytest.mark.parametrize("fmt", ["BMP", "DIB", "ICO", "AVIF", "JPEG2000", "TGA", "QOI", "PCX"])
def test_other_formats_are_converted(tmp_path, fmt):
    # GIF and TIFF are covered by the transparency cases below. Each remaining
    # format exercises our decoder allowlist as well as the conversion path.
    Image.init()
    if fmt not in Image.OPEN or fmt not in Image.SAVE:
        pytest.skip(f"Pillow build does not support {fmt}")
    path = tmp_path / "image.png"
    size = (32, 32) if fmt == "ICO" else (12, 8)
    Image.new("RGB", size, "red").save(path, format=fmt, **({"sizes": [size]} if fmt == "ICO" else {}))
    source_data = path.read_bytes()
    with Image.open(path) as original:
        expected_pixels = original.convert("RGBA").tobytes()

    result = view_image.callable(None, path=str(path))

    decoded = assert_prepared_image(result, path, source_data)
    assert result.mime_type == "image/png"
    assert decoded.size == size
    assert decoded.convert("RGBA").tobytes() == expected_pixels
    assert f"re-encoded {fmt} -> PNG" in result.status


def test_exif_orientation_is_applied(tmp_path):
    path = tmp_path / "rotated"
    original = Image.new("RGB", (12, 8), "red")
    original.paste("blue", (0, 0, 6, 8))
    exif = Image.Exif()
    exif[274] = 6
    original.save(path, format="JPEG", exif=exif)
    source_data = path.read_bytes()
    with Image.open(path) as encoded:
        expected = encoded.transpose(Image.Transpose.ROTATE_270).convert("RGB")

    result = view_image.callable(None, path=str(path))

    decoded = assert_prepared_image(result, path, source_data)
    assert decoded.size == (8, 12)
    assert decoded.convert("RGB").tobytes() == expected.tobytes()
    assert decoded.getexif().get(274, 1) == 1
    assert "re-encoded" in result.status
    assert "unchanged" not in result.status


@pytest.mark.parametrize("kind", ["rgba", "palette", "grayscale_alpha"])
def test_conversion_preserves_transparency(tmp_path, kind):
    path = tmp_path / "transparent"
    if kind == "rgba":
        original = Image.new("RGBA", (12, 8), (255, 0, 0, 0))
        original.paste((0, 0, 255, 128), (6, 0, 12, 8))
        original.save(path, format="TIFF")
    elif kind == "grayscale_alpha":
        original = Image.new("LA", (12, 8), (100, 0))
        original.paste((200, 128), (6, 0, 12, 8))
        original.save(path, format="TIFF")
    else:
        original = Image.new("P", (12, 8), 0)
        original.putpalette([255, 0, 0, 0, 0, 255] + [0] * 762)
        original.paste(1, (6, 0, 12, 8))
        original.save(path, format="GIF", transparency=0)
    source_data = path.read_bytes()
    with Image.open(path) as encoded:
        expected_pixels = encoded.convert("RGBA").tobytes()

    result = view_image.callable(None, path=str(path))

    decoded = assert_prepared_image(result, path, source_data)
    assert decoded.convert("RGBA").tobytes() == expected_pixels
    assert decoded.convert("RGBA").getpixel((0, 0))[3] == 0
    assert decoded.convert("RGBA").getpixel((11, 0))[3] == (255 if kind == "palette" else 128)


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


def test_cmyk_tiff_is_converted_to_rgb(tmp_path):
    path = tmp_path / "cmyk.tiff"
    Image.new("CMYK", (12, 8), (0, 255, 255, 0)).save(path)
    source_data = path.read_bytes()

    result = view_image.callable(None, path=str(path))

    decoded = assert_prepared_image(result, path, source_data)
    assert decoded.convert("RGB").getpixel((0, 0)) == (255, 0, 0)
    assert "color CMYK -> RGB" in result.status


@pytest.mark.parametrize("mode", ["P", "RGB"])
def test_icc_conversion_resolves_palette_and_color_key_transparency(tmp_path, mode):
    image = Image.new(mode, (12, 8))
    if mode == "P":
        image.putpalette([255, 0, 0, 0, 0, 255] + [0] * 762)
        image.paste(1, (6, 0, 12, 8))
        image.info["transparency"] = 0
    else:
        image.paste((255, 0, 0), (0, 0, 6, 8))
        image.paste((0, 0, 255), (6, 0, 12, 8))
        image.info["transparency"] = (255, 0, 0)
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    orientation = Image.Exif()
    orientation[274] = 3  # Force conversion of an otherwise native PNG.
    path = tmp_path / "transparent_profile.png"
    image.save(path, icc_profile=profile, exif=orientation)
    source = path.read_bytes()

    result = view_image.callable(None, path=str(path))

    rgba = assert_prepared_image(result, path, source).convert("RGBA")
    assert rgba.getpixel((0, 0)) == (0, 0, 255, 255)
    assert rgba.getpixel((11, 0)) == (255, 0, 0, 0)
    assert "sRGB" in result.status


@pytest.mark.parametrize("mode", ["I", "F"])
def test_ambiguous_high_dynamic_range_is_not_silently_clipped(tmp_path, mode):
    path = tmp_path / "hdr.tiff"
    Image.new(mode, (12, 8), 1000).save(path)
    with pytest.raises(ValueError, match="Unsupported color mode.*export an 8-bit raster"):
        view_image.callable(None, path=str(path))


# Exercise the shipped limits too: tests with substituted caps cannot catch
# accidental changes to the defaults or unnecessary resizing at their boundary.
def test_exact_real_dimension_limits_preserve_original_bytes(tmp_path):
    size = (2000, 1000)
    path = tmp_path / "exact_dimensions.png"
    Image.new("RGB", size, "red").save(path)
    source_data = path.read_bytes()

    result = view_image.callable(None, path=str(path))

    assert_prepared_image(result, path, source_data)
    assert result.data == source_data
    assert (result.width, result.height) == size
    assert result.status.endswith("; unchanged.")


def test_real_pixel_limit(tmp_path):
    path = tmp_path / "2mp.png"
    Image.new("L", (1600, 1251), "white").save(path)
    source_data = path.read_bytes()
    result = view_image.callable(None, path=str(path))
    assert_prepared_image(result, path, source_data)
    assert result.width * result.height <= 2_000_000
    assert max(result.width, result.height) <= 2000
    assert "downsized 1600x1251" in result.status


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


@pytest.mark.parametrize("size,pixel_limit", [
    ((400, 200), 5_000), ((200, 400), 5_000), ((1_000, 1), 100), ((1, 1_000), 100),
])
def test_pixel_limit_resizes_with_aspect_ratio(tmp_path, monkeypatch, size, pixel_limit):
    monkeypatch.setattr(image_impl, "MAX_PIXELS", pixel_limit)
    path = tmp_path / "large.png"
    Image.new("RGB", size, "red").save(path)
    source_data = path.read_bytes()

    result = view_image.callable(None, path=str(path))

    decoded = assert_prepared_image(result, path, source_data)
    assert 0 < result.width <= size[0]
    assert 0 < result.height <= size[1]
    assert result.width * result.height <= pixel_limit
    assert len(result.data) <= image_impl.MAX_BYTES
    assert abs(result.width * size[1] - result.height * size[0]) <= max(size)
    assert decoded.convert("RGB").getpixel((0, 0)) == (255, 0, 0)
    assert f"downsized {size[0]}x{size[1]}" in result.status


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


def test_antialiasing_filters_high_frequency_pixels(tmp_path, monkeypatch):
    monkeypatch.setattr(image_impl, "MAX_PIXELS", 16 * 16)
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


@pytest.mark.parametrize("pixel_limit", [2_000_000, 700])
def test_byte_limit_reduces_noisy_image_and_retains_alpha(tmp_path, monkeypatch, pixel_limit):
    monkeypatch.setattr(image_impl, "MAX_BYTES", 2_000)
    monkeypatch.setattr(image_impl, "MAX_PIXELS", pixel_limit)
    path = tmp_path / "noisy.png"
    size = (128, 96)
    original = Image.frombytes("RGB", size, Random(0).randbytes(size[0] * size[1] * 3))
    alpha_row = bytes([0] * 42 + [128] * 44 + [255] * 42)
    original.putalpha(Image.frombytes("L", size, alpha_row * size[1]))
    original.save(path)
    source_data = path.read_bytes()
    assert len(source_data) > image_impl.MAX_BYTES

    result = view_image.callable(None, path=str(path))

    decoded = assert_prepared_image(result, path, source_data)
    assert 0 < result.width <= size[0]
    assert 0 < result.height <= size[1]
    assert result.width * result.height < size[0] * size[1]
    assert result.width * result.height <= pixel_limit
    assert len(result.data) <= image_impl.MAX_BYTES
    assert abs(result.width * size[1] - result.height * size[0]) <= max(size)
    assert decoded.convert("RGBA").getchannel("A").getextrema() == (0, 255)
    assert "downsized" in result.status
    assert "lossy" not in result.status


def test_optimized_png_avoids_unnecessary_lossy_conversion(tmp_path, monkeypatch):
    monkeypatch.setattr(image_impl, "MAX_BYTES", 2000)
    path = tmp_path / "uncompressed.png"
    Image.new("RGB", (200, 100), "red").save(path, compress_level=0)
    source_data = path.read_bytes()
    assert len(source_data) > image_impl.MAX_BYTES
    result = view_image.callable(None, path=str(path))
    assert_prepared_image(result, path, source_data)
    assert result.mime_type == "image/png"
    assert (result.width, result.height) == (200, 100)
    assert "re-encoded PNG -> PNG" in result.status
    assert "lossy" not in result.status


def test_lossy_jpeg_fallback_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(image_impl, "MAX_BYTES", 25_000)
    path = tmp_path / "noise.png"
    Image.frombytes("RGB", (100, 100), Random(0).randbytes(30_000)).save(path)
    source_data = path.read_bytes()
    assert len(source_data) > image_impl.MAX_BYTES
    result = view_image.callable(None, path=str(path))
    assert_prepared_image(result, path, source_data)
    assert result.mime_type == "image/jpeg"
    assert len(result.data) <= image_impl.MAX_BYTES
    assert (result.width, result.height) == (100, 100)
    assert "re-encoded PNG -> JPEG (lossy JPEG)" in result.status


def test_impossible_byte_limit_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(image_impl, "MAX_BYTES", 1)
    path = tmp_path / "single_pixel.png"
    Image.new("RGB", (1, 2), "red").save(path)
    source_data = path.read_bytes()
    with pytest.raises(ValueError, match="Cannot fit image within"):
        view_image.callable(None, path=str(path))
    assert path.read_bytes() == source_data


def test_pillow_decompression_bomb_protection_remains_enabled(image_path, monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)
    with pytest.raises(ValueError, match="decompression bomb"):
        view_image.callable(None, path=str(image_path))


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
