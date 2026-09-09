"""Regression cases found while checking image container and color edge cases."""

from io import BytesIO
from struct import pack

from PIL import Image, ImageCms, PngImagePlugin
import pytest

from ..func_lib import view_image


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

    with Image.open(BytesIO(result.data)) as output:
        rgba = output.convert("RGBA")
        assert rgba.getpixel((0, 0)) == (0, 0, 255, 255)
        assert rgba.getpixel((11, 0)) == (255, 0, 0, 0)
    assert "sRGB" in result.status
    assert path.read_bytes() == source


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


@pytest.mark.parametrize("missing_bytes", [1, 4])
def test_png_missing_iend_checksum_is_rejected(tmp_path, missing_bytes):
    path = tmp_path / "truncated_checksum.png"
    Image.new("RGB", (12, 8), "red").save(path)
    source = path.read_bytes()[:-missing_bytes]
    path.write_bytes(source)

    with pytest.raises(ValueError, match="Cannot load image"):
        view_image.callable(None, path=str(path))

    assert path.read_bytes() == source


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


def test_grayscale_alpha_conversion_stays_compact(tmp_path):
    path = tmp_path / "grayscale_alpha.tiff"
    Image.new("LA", (12, 8), (100, 128)).save(path)
    source = path.read_bytes()
    result = view_image.callable(None, path=str(path))
    with Image.open(BytesIO(result.data)) as image:
        assert image.mode == "LA"
        assert image.getpixel((0, 0)) == (100, 128)
    assert path.read_bytes() == source
