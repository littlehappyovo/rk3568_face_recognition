import io
import warnings

from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

from .store import StoreValidationError


def checked_jpeg(data: bytes, *, remote=False) -> bytes:
    maximum = 256 * 1024 if remote else 8 * 1024 * 1024
    dimensions = (512, 512) if remote else (4096, 4096)
    if len(data) > maximum:
        raise RequestEntityTooLarge()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in (("JPEG",) if remote else ("JPEG", "PNG")):
                    raise StoreValidationError("只支持JPEG/PNG照片")
                if not 0 < image.width <= dimensions[0] or not 0 < image.height <= dimensions[1]:
                    raise StoreValidationError("照片尺寸超过限制")
                image.load()
                output = io.BytesIO()
                image.convert("RGB").save(output, format="JPEG", quality=85)
                normalized = output.getvalue()
                if len(normalized) > maximum:
                    raise RequestEntityTooLarge()
                return normalized
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise StoreValidationError("照片无法解码") from error
