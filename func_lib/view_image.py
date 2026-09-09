"""
Optional `view_image` builtin (add to agent's `uses` list).

* Absolute paths only
* Return a frozen ImageResult or raise. Sources are read only;
  conversion and snapshots stay in memory.
* Reject corrupt, animated, multipage/multi-image and unsupported inputs.
* PNG/JPEG/still WebP are native to all providers; keep fitting originals
  byte-for-byte unless EXIF orientation requires normalization or need downsizing.
* `transform_image` runs only when format, size or EXIF orientation requires a change
  and always re-encodes. Bypassing it preserves exact original file bytes;
  validation and base64 encoding still occur in both paths.
* Output caps: 512 KiB (before base64), 2M pixels, 2,000px per edge.
  These conservative caps stay below provider limits and minimize context overhead;
  every cap must be satisfied.
  Downsize proportionally with LANCZOS antialiasing.
* Re-encoding prefers optimized PNG, then high-quality JPEG for opaque images
  if smaller, then downsizing. Preserves alpha; normalize modes and ICC profiles
  to sRGB during conversion. Providers document no shared required color mode.
* `ImageResult.status` reports delivered MIME type, dimensions, byte count and "unchanged" or
  re-encoding/color/resizing/lossy-compression notes.
* Successful `ToolResultPart.outputs` (netflux transcript type) retains the original
  `ImageResult` with `is_error=False`; exceptions use ordinary provider error text
  with `is_error=True`.
* `node.result()` returns the `ImageResult` on success or raises on failure.
  Provider replay extracts status and payload into native tool-result content.
* Recommend TUIs display `ImageResult.status` and a media placeholder
  without decoding the image payload.
* Provider transcript replay still has whole-request limits.
* For SVG, rejected. Read the source for structure or render to PNG to inspect appearance.
"""

import base64
from dataclasses import dataclass, field
from io import BytesIO
from math import sqrt
from pathlib import Path
from typing import BinaryIO, Final, Literal, cast

from PIL import IcoImagePlugin, Image, ImageCms, ImageOps, UnidentifiedImageError

from ..core import CodeFunction, FunctionArg, RunContext


ImageMime = Literal["image/png", "image/jpeg", "image/webp"]
ImageEncoding = Literal["PNG", "JPEG"]

# Preserve native encodings when the validated source fits every output cap.
MIME_TYPES: Final[dict[str, ImageMime]] = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
}

# Explicit raster decoders avoid formats that launch external renderers (EPS,
# for example), or whose layers/embedded previews would need special treatment.
# PPM is omitted: files can concatenate images that Pillow silently ignores.
INPUT_FORMATS: Final[tuple[str, ...]] = (
    "PNG", "JPEG", "WEBP", "GIF", "BMP", "DIB", "TIFF", "ICO", "AVIF",
    "JPEG2000", "TGA", "QOI", "PCX",
)

# Bound encoded image bytes before base64 expansion, total pixels and each edge.
MAX_BYTES: Final[int] = 512 * 1024
MAX_PIXELS: Final[int] = 2_000_000
MAX_SIDE: Final[int] = 2_000


@dataclass(frozen=True)
class ImageResult:
    """Immutable image already encoded and downsized if needed, ready for model ingestion."""

    source_path: str
    mime_type: ImageMime
    width: int
    height: int
    status: str
    data: bytes = field(repr=False)
    base64_data: str = field(repr=False)

    def __str__(self) -> str:
        return self.status


@dataclass
class _PreparedImage:
    """Image within output limits, awaiting base64 encoding and ImageResult construction."""

    mime_type: ImageMime
    width: int
    height: int
    notes: list[str]
    data: bytes = field(repr=False)


def _encode_image(image: Image.Image, encoding: ImageEncoding) -> bytes:
    """Return optimized PNG bytes or quality-90 JPEG bytes without chroma subsampling."""

    # Encode entirely in memory, retaining fine detail when JPEG is necessary.
    with BytesIO() as buffer:
        if encoding == "JPEG":
            image.save(buffer, format=encoding, quality=90, subsampling=0)
        else:
            image.save(buffer, format=encoding, optimize=True)

        return buffer.getvalue()


def _resize(image: Image.Image, scale: float) -> Image.Image:
    """Return a proportional LANCZOS resize within pixel and edge limits."""

    # Round thin images rather than truncating their thickness; clamp rounded
    # dimensions to the area/edge caps, including one-pixel-wide panoramas.
    width = max(1, min(MAX_SIDE, MAX_PIXELS, round(image.width * scale)))
    height = max(1, min(MAX_SIDE, MAX_PIXELS // width, round(image.height * scale)))

    # Use the same scale on both axes and filter out detail above the new resolution.
    scale = min(width / image.width, height / image.height)
    return image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )


def transform_image(image: Image.Image) -> _PreparedImage:
    """Apply required orientation/color fixes and resizing + antialias; always re-encode as PNG/JPEG."""

    # Retain source details for the status message and normalize orientation first.
    source_format: str | None = image.format
    source_mode: str = image.mode
    notes: list[str] = []
    ImageOps.exif_transpose(image, in_place=True)

    # Normalize modes that PNG/JPEG cannot encode, preserving 16-bit transparency.
    profile: bytes | None = image.info.get("icc_profile")
    alpha: Image.Image | None = None
    if image.mode.startswith("I;16"):
        if "transparency" in image.info:
            # Match the original 16-bit color key before quantizing grayscale.
            alpha_table: list[int] = [255] * 65_536
            alpha_table[image.info["transparency"]] = 0
            alpha = image.convert("I").point(alpha_table, "L")

        # Map the full 16-bit grayscale range to 8 bits instead of clipping at 255.
        image = image.convert("F").point(lambda value: value / 257).convert("L")
        if alpha is not None:
            image.putalpha(alpha)

    elif image.mode in ("I", "F"):
        raise ValueError(
            f"Unsupported color mode {image.mode!r}; export an 8-bit raster first"
        )

    elif image.mode == "1" and profile:
        # Keep monochrome pixels compatible with their grayscale ICC profile.
        image = image.convert("L")

    elif image.mode not in ("RGB", "RGBA", "L", "LA", "CMYK", "LAB"):
        image = image.convert("RGBA" if image.has_transparency_data else "RGB")

    # Convert embedded color profiles to sRGB, carrying transparency separately.
    if profile:
        # CMS does not interpret color-key transparency or palette indices.
        alpha = image.convert("RGBA").getchannel("A") if image.has_transparency_data else None
        color_mode = {"RGBA": "RGB", "LA": "L"}.get(image.mode, image.mode)
        color: Image.Image = image.convert(color_mode)
        converted: Image.Image | None = ImageCms.profileToProfile(
            color,
            ImageCms.ImageCmsProfile(BytesIO(profile)),
            ImageCms.createProfile("sRGB"),
            outputMode="RGB",
        )
        if converted is None:
            raise ValueError("Cannot convert image color profile")

        image = converted
        if alpha is not None:
            image.putalpha(alpha)
        notes.append("color profile converted to sRGB")

    elif image.mode not in ("RGB", "RGBA", "L", "LA") or "transparency" in image.info:
        image = image.convert("RGBA" if image.has_transparency_data else "RGB")

    # Report color changes and discard metadata before encoding the payload.
    if image.mode != source_mode:
        notes.append(f"color {source_mode} -> {image.mode}")
    image.info.clear()
    original: Image.Image = image

    # Satisfy the pixel and edge caps with a proportional, antialiased resize.
    width, height = image.size
    scale = min(1.0, sqrt(MAX_PIXELS / (width * height)), MAX_SIDE / max(width, height))
    if scale < 1:
        image = _resize(original, scale)

    # Prefer optimized PNG; use JPEG only when opaque content becomes smaller.
    encoding: ImageEncoding = "PNG"
    data: bytes = _encode_image(image, encoding)
    if len(data) > MAX_BYTES and not image.has_transparency_data:
        jpeg: bytes = _encode_image(image, "JPEG")
        if len(jpeg) < len(data):
            data = jpeg
            encoding = "JPEG"

    # Resize from the original pixels until the encoding fits the byte cap.
    while len(data) > MAX_BYTES:
        if image.size == (1, 1):
            raise ValueError(f"Cannot fit image within the {MAX_BYTES}-byte limit")

        # Reduce at least one pixel even when rounding very small dimensions.
        current_scale = min(image.width / original.width, image.height / original.height)
        reduction = min(
            0.9,
            1 - 1 / max(image.size),
            sqrt(MAX_BYTES / len(data)) * 0.95,
        )
        image = _resize(original, current_scale * reduction)
        data = _encode_image(image, encoding)

    # Describe every re-encoding, lossy compression and dimension change.
    encoding_note = f"re-encoded {source_format} -> {encoding}"
    if encoding == "JPEG":
        encoding_note += " (lossy JPEG)"
    notes.insert(0, encoding_note)

    if image.size != original.size:
        notes.append(
            f"downsized {original.width}x{original.height} -> {image.width}x{image.height}"
        )

    return _PreparedImage(
        mime_type=MIME_TYPES[encoding],
        width=image.width,
        height=image.height,
        notes=notes,
        data=data,
    )


def validate_is_still(image: Image.Image) -> None:
    """Raise ValueError for animation or multiple frames, pages or icon entries."""

    # Reject animation containers even when they happen to contain a single frame.
    if (
        getattr(image, "is_animated", False)
        or getattr(image, "n_frames", 1) != 1
        or getattr(image, "custom_mimetype", None) == "image/apng"
        or (image.format == "GIF" and "loop" in image.info)
    ):
        raise ValueError(
            "Animated or multipage images are unsupported; export one still frame/page first"
        )

    # Count icon entries directly; distinct images may share the same dimensions.
    if image.format == "ICO":
        icon = cast(IcoImagePlugin.IcoImageFile, image)
        if icon.ico.nb_items != 1:
            raise ValueError("Multi-image icons are unsupported; export one icon as PNG first")


def _view_image(ctx: RunContext, *, path: str) -> ImageResult:
    """Read and validate an image file, enforce output limits (re-encode, downsize if needed)
    and return an ImageResult."""

    # Resolve an absolute path to a regular file before opening it read-only.
    source = Path(path)
    if not source.is_absolute():
        raise FileNotFoundError("no file at that absolute filepath")

    try:
        source = source.resolve(strict=True)
        if not source.is_file():
            raise ValueError(f"Image path is not a regular file: {source}")
    except FileNotFoundError:
        raise FileNotFoundError("no file at that absolute filepath") from None

    # Snapshot small inputs; decode larger files without first copying all bytes.
    with source.open("rb") as stream:
        data: bytes = stream.read(MAX_BYTES + 1)
        image_stream: BinaryIO = BytesIO(data) if len(data) <= MAX_BYTES else stream

        try:
            # Pillow labels one-frame animated WebP as still. VP8X's animation
            # flag is authoritative even for that case:
            # https://developers.google.com/speed/webp/docs/riff_container#extended_file_format
            if (
                len(data) > 20
                and data[:4] == b"RIFF"
                and data[8:16] == b"WEBPVP8X"
                and data[20] & 2
            ):
                raise ValueError("Animated WebP is unsupported; export a still image first")

            # Validate the still-image container before decoding its pixels.
            image_stream.seek(0)
            with Image.open(image_stream, formats=INPUT_FORMATS) as image:
                validate_is_still(image)
                image.verify()

                if image.format == "GIF":
                    # Require the final sub-block terminator and GIF trailer that Pillow may ignore.
                    image_stream.seek(-2, 2)
                    if image_stream.read() != b"\x00;":
                        raise ValueError("Corrupt or truncated GIF end marker")
                if image.format == "PNG":
                    # Pillow.verify() stops before validating IEND's CRC.
                    image_stream.seek(-12, 2)
                    if image_stream.read() != b"\x00\x00\x00\x00IEND\xaeB\x60\x82":
                        raise ValueError("Corrupt or truncated PNG end marker")

            # verify() checks structure; reopen to fully decode the still image.
            image_stream.seek(0)
            with Image.open(image_stream, formats=INPUT_FORMATS) as image:
                image.load()

                # Passing every check preserves the exact original file bytes.
                prepared: _PreparedImage
                if (
                    image.format in MIME_TYPES
                    and len(data) <= MAX_BYTES
                    and image.width * image.height <= MAX_PIXELS
                    and max(image.size) <= MAX_SIDE
                    and image.getexif().get(274, 1) not in range(2, 9)
                ):
                    prepared = _PreparedImage(
                        mime_type=MIME_TYPES[image.format],
                        width=image.width,
                        height=image.height,
                        notes=["unchanged"],
                        data=data,
                    )
                else:
                    # At least one change is required; this path always re-encodes.
                    prepared = transform_image(image)

        # Return actionable SVG advice through the normal tool error path.
        except UnidentifiedImageError as exc:
            if source.suffix.lower() in (".svg", ".svgz") or b"<svg" in data[:4096].lower():
                hint = (
                    "SVG is not a supported raster image. Read its source for structure, "
                    "or render it to PNG and call view_image for visual inspection."
                )
            else:
                hint = "Invalid or unsupported raster image."

            raise ValueError(f"Cannot load image {str(source)!r}: {hint}") from exc

        # Include the source path when validation, decoding or conversion fails.
        except (
            OSError, SyntaxError, ValueError,
            Image.DecompressionBombError, ImageCms.PyCMSError,
        ) as exc:
            raise ValueError(f"Cannot load image {str(source)!r}: {exc}") from exc

    # Freeze the delivered image and concise status for provider replay and TUIs.
    return ImageResult(
        source_path=str(source),
        mime_type=prepared.mime_type,
        width=prepared.width,
        height=prepared.height,
        status=(
            f"{prepared.mime_type}, {prepared.width}x{prepared.height}, "
            f"{len(prepared.data)} bytes; {'; '.join(prepared.notes)}."
        ),
        data=prepared.data,
        base64_data=base64.b64encode(prepared.data).decode("ascii"),
    )


# Expose the opt-in builtin with a concise description of its output and failures.
view_image: Final[CodeFunction] = CodeFunction(
    name="view_image",
    desc=(
        "View an image. Returns image content and reports any downsizing or re-encoding. "
        "Handles typical raster formats. "
        "Downsizes to caps: 512 KiB, 2M pixels, 2,000px per edge. "
        "Raises on unreadable, invalid, animated or multipage inputs."
    ),
    args=[FunctionArg("path", str, desc="Absolute path to image file")],
    callable=_view_image,
)
