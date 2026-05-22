from typing import Dict, Optional, Any
import math
import random
from dataclasses import dataclass
from io import BytesIO

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class _ImageAugmentationConfig:
    enabled: bool
    seed_offset: int
    probability: float
    rotation_degrees: float
    jpeg_quality_min: int
    jpeg_quality_max: int
    gaussian_blur_max: float
    contrast_min: float
    contrast_max: float
    brightness_min: float
    brightness_max: float
    sharpness_min: float
    sharpness_max: float
    grayscale_probability: float
    noise_std_max: float
    erase_probability: float
    erase_area_min: float
    erase_area_max: float
    ink_bleed_probability: float
    ink_bleed_strength: float


class _ImageAugmenter:
    def __init__(
        self,
        config: _ImageAugmentationConfig,
        seed: int,
    ) -> None:
        self.config = config
        self.rng = random.Random(seed + config.seed_offset)

    def __call__(
        self,
        image: Image.Image,
    ) -> Image.Image:
        if self.rng.random() > self.config.probability:
            return image

        augmented = image.convert("RGB")
        augmented = self._rotate_image(image=augmented)
        augmented = self._adjust_tone(image=augmented)
        augmented = self._apply_blur(image=augmented)
        augmented = self._apply_noise(image=augmented)
        augmented = self._apply_erasure(image=augmented)
        augmented = self._apply_ink_bleed(image=augmented)
        augmented = self._apply_grayscale(image=augmented)
        augmented = self._apply_jpeg_compression(image=augmented)
        return augmented

    def _rotate_image(
        self,
        image: Image.Image,
    ) -> Image.Image:
        if self.config.rotation_degrees <= 0:
            return image

        angle = self.rng.uniform(
            -self.config.rotation_degrees,
            self.config.rotation_degrees,
        )
        return image.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=(255, 255, 255),
        )

    def _adjust_tone(
        self,
        image: Image.Image,
    ) -> Image.Image:
        image = ImageEnhance.Contrast(image).enhance(
            self.rng.uniform(
                self.config.contrast_min,
                self.config.contrast_max,
            )
        )
        image = ImageEnhance.Brightness(image).enhance(
            self.rng.uniform(
                self.config.brightness_min,
                self.config.brightness_max,
            )
        )
        image = ImageEnhance.Sharpness(image).enhance(
            self.rng.uniform(
                self.config.sharpness_min,
                self.config.sharpness_max,
            )
        )
        return image

    def _apply_blur(
        self,
        image: Image.Image,
    ) -> Image.Image:
        if self.config.gaussian_blur_max <= 0:
            return image

        radius = self.rng.uniform(
            0.0,
            self.config.gaussian_blur_max,
        )
        if radius == 0:
            return image
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    def _apply_noise(
        self,
        image: Image.Image,
    ) -> Image.Image:
        if self.config.noise_std_max <= 0:
            return image

        sigma = self.rng.uniform(
            0.0,
            self.config.noise_std_max,
        )
        if sigma == 0:
            return image

        noise = Image.effect_noise(
            image.size,
            sigma,
        ).convert("RGB")
        return Image.blend(
            image,
            noise,
            alpha=min(0.25, sigma / 255.0),
        )

    def _apply_erasure(
        self,
        image: Image.Image,
    ) -> Image.Image:
        if self.config.erase_probability <= 0:
            return image
        if self.rng.random() > self.config.erase_probability:
            return image

        width, height = image.size
        area_ratio = self.rng.uniform(
            self.config.erase_area_min,
            self.config.erase_area_max,
        )
        side_ratio = math.sqrt(area_ratio)
        patch_width = max(1, int(width * side_ratio))
        patch_height = max(1, int(height * side_ratio))
        x0 = self.rng.randint(0, max(0, width - patch_width))
        y0 = self.rng.randint(0, max(0, height - patch_height))
        fill = self.rng.randint(230, 255)

        erased = image.copy()
        draw = ImageDraw.Draw(erased)
        draw.rectangle(
            [x0, y0, x0 + patch_width, y0 + patch_height],
            fill=(fill, fill, fill),
        )
        return erased

    def _apply_ink_bleed(
        self,
        image: Image.Image,
    ) -> Image.Image:
        if self.config.ink_bleed_probability <= 0:
            return image
        if self.rng.random() > self.config.ink_bleed_probability:
            return image

        bled = image.filter(ImageFilter.MinFilter(size=3))
        return Image.blend(
            image,
            bled,
            alpha=self.config.ink_bleed_strength,
        )

    def _apply_grayscale(
        self,
        image: Image.Image,
    ) -> Image.Image:
        if self.config.grayscale_probability <= 0:
            return image
        if self.rng.random() > self.config.grayscale_probability:
            return image
        return image.convert("L").convert("RGB")

    def _apply_jpeg_compression(
        self,
        image: Image.Image,
    ) -> Image.Image:
        if self.config.jpeg_quality_min > self.config.jpeg_quality_max:
            return image

        quality = self.rng.randint(
            self.config.jpeg_quality_min,
            self.config.jpeg_quality_max,
        )
        buffer = BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=quality,
        )
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


def _build_image_augmenter(
    config: Dict[str, Any],
    seed: int,
) -> Optional[_ImageAugmenter]:
    augmentation_config = _ImageAugmentationConfig(
        enabled=bool(config["enabled"]),
        seed_offset=int(config["seed_offset"]),
        probability=float(config["probability"]),
        rotation_degrees=float(config["rotation_degrees"]),
        jpeg_quality_min=int(config["jpeg_quality_min"]),
        jpeg_quality_max=int(config["jpeg_quality_max"]),
        gaussian_blur_max=float(config["gaussian_blur_max"]),
        contrast_min=float(config["contrast_min"]),
        contrast_max=float(config["contrast_max"]),
        brightness_min=float(config["brightness_min"]),
        brightness_max=float(config["brightness_max"]),
        sharpness_min=float(config["sharpness_min"]),
        sharpness_max=float(config["sharpness_max"]),
        grayscale_probability=float(config["grayscale_probability"]),
        noise_std_max=float(config["noise_std_max"]),
        erase_probability=float(config["erase_probability"]),
        erase_area_min=float(config["erase_area_min"]),
        erase_area_max=float(config["erase_area_max"]),
        ink_bleed_probability=float(config["ink_bleed_probability"]),
        ink_bleed_strength=float(config["ink_bleed_strength"]),
    )
    if not augmentation_config.enabled:
        return None
    return _ImageAugmenter(
        config=augmentation_config,
        seed=seed,
    )
