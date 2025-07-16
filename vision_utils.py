import os
import logging
from typing import Union

try:
    from google.cloud import vision  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    vision = None  # type: ignore

try:
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - optional
    pytesseract = None  # type: ignore
    Image = None  # type: ignore


logger = logging.getLogger(__name__)


class OCRError(RuntimeError):
    """Raised when OCR backends fail and detection is critical."""


def detect_text(image: Union[str, bytes], *, critical: bool | None = None) -> str:
    """Return detected text using Google Vision or Tesseract."""
    critical = critical if critical is not None else os.getenv("OCR_CRITICAL") == "1"

    errors: list[str] = []

    if vision and os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        try:
            client = vision.ImageAnnotatorClient()
            if isinstance(image, str):
                with open(image, "rb") as f:
                    content = f.read()
            else:
                content = image
            img = vision.Image(content=content)
            resp = client.text_detection(image=img)
            if resp.error.message:
                errors.append(resp.error.message)
            else:
                return " ".join(t.description for t in resp.text_annotations)
        except Exception as exc:
            errors.append(f"vision: {exc}")
    else:
        if not vision:
            errors.append("google vision unavailable")
        else:
            errors.append("no credentials for vision")

    if pytesseract and Image:
        try:
            if isinstance(image, str):
                img = Image.open(image)
            else:
                from io import BytesIO
                img = Image.open(BytesIO(image))
            return pytesseract.image_to_string(img)
        except Exception as exc:
            errors.append(f"tesseract: {exc}")
    else:
        errors.append("pytesseract unavailable")

    msg = "; ".join(errors) if errors else "text detection failed"
    logger.warning("text detection failed: %s", msg)
    if critical:
        raise OCRError(msg)
    return ""
