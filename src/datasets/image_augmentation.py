from typing import Dict, List, Tuple, Optional, Protocol, Any
import os
import math
import random
from dataclasses import dataclass
from io import BytesIO

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


def build_image_augmenter(
    config: Dict[str, Any],
    seed: int,
) -> Optional["_ImageAugmenterProtocol"]:
    _validate_required_keys(
        config=config,
        keys=[
            "enabled",
            "seed_offset",
            "backend",
            "probability",
        ],
        prefix="image_augmentation",
    )
    enabled = bool(config["enabled"])
    if not enabled:
        return None

    backend = str(config["backend"])
    if backend == "pil":
        return _PILImageAugmenter(
            config=_build_pil_config(config=config),
            seed=seed,
        )
    if backend == "albumentations":
        return _AlbumentationsImageAugmenter(
            config=_build_albumentations_config(config=config),
            seed=seed,
        )
    raise ValueError("image_augmentation.backend must be one of: pil, albumentations.")


class _ImageAugmenterProtocol(Protocol):
    def __call__(
        self,
        image: Image.Image,
    ) -> Image.Image: ...


@dataclass(frozen=True)
class _PILImageAugmentationConfig:
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


@dataclass(frozen=True)
class _AlbumentationsImageAugmentationConfig:
    enabled: bool
    seed_offset: int
    probability: float
    resize: Dict[str, Any]
    rotate: Dict[str, Any]
    blur: Dict[str, Any]
    noise: Dict[str, Any]
    seasoning: Dict[str, Any]
    coarse_dropout: Dict[str, Any]
    scan: Dict[str, Any]
    color: Dict[str, Any]
    hsv: Dict[str, Any]
    rgb_shift: Dict[str, Any]
    jpeg: Dict[str, Any]
    weather: Dict[str, Any]


class _PILImageAugmenter:
    def __init__(
        self,
        config: _PILImageAugmentationConfig,
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
            alpha=min(
                0.25,
                sigma / 255.0,
            ),
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
        patch_width = max(
            1,
            int(width * side_ratio),
        )
        patch_height = max(
            1,
            int(height * side_ratio),
        )
        x0 = self.rng.randint(
            0,
            max(
                0,
                width - patch_width,
            ),
        )
        y0 = self.rng.randint(
            0,
            max(
                0,
                height - patch_height,
            ),
        )
        fill = self.rng.randint(
            230,
            255,
        )

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


class _AspectRatioPreservingResize:
    def __init__(
        self,
        cv2_module: Any,
        rng: random.Random,
        scale_percent_range: Tuple[float, float],
    ) -> None:
        self.cv2 = cv2_module
        self.rng = rng
        self.scale_percent_range = scale_percent_range
        self.interpolations = [
            self.cv2.INTER_LINEAR,
            self.cv2.INTER_AREA,
            self.cv2.INTER_CUBIC,
            self.cv2.INTER_LANCZOS4,
        ]

    def __call__(
        self,
        image: Any,
        **params: Any,
    ) -> Any:
        height, width = image.shape[:2]
        scale_percent = self.rng.uniform(*self.scale_percent_range)
        scale_factor = scale_percent / 100.0
        resized_height = max(
            1,
            int(height * scale_factor),
        )
        resized_width = max(
            1,
            int(width * scale_factor),
        )
        return self.cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=self.rng.choice(self.interpolations),
        )


class _LowResBlur:
    def __init__(
        self,
        cv2_module: Any,
        rng: random.Random,
        factor_range: Tuple[float, float],
    ) -> None:
        self.cv2 = cv2_module
        self.rng = rng
        self.factor_range = factor_range
        self.interpolations = [
            self.cv2.INTER_LINEAR,
            self.cv2.INTER_AREA,
            self.cv2.INTER_CUBIC,
            self.cv2.INTER_LANCZOS4,
        ]

    def __call__(
        self,
        image: Any,
        **params: Any,
    ) -> Any:
        factor_x = self.rng.uniform(*self.factor_range)
        factor_y = self.rng.uniform(*self.factor_range)
        image_small = self.cv2.resize(
            src=image,
            dsize=(0, 0),
            dst=None,
            fx=factor_x,
            fy=factor_y,
            interpolation=self.rng.choice(self.interpolations),
        )
        return self.cv2.resize(
            src=image_small,
            dsize=(image.shape[1], image.shape[0]),
            interpolation=self.rng.choice(self.interpolations),
        )


class _RandomSeasoning:
    def __init__(
        self,
        rng: random.Random,
        seasoning_range: Tuple[float, float],
    ) -> None:
        self.rng = rng
        self.seasoning_range = seasoning_range

    def __call__(
        self,
        image: Any,
        **params: Any,
    ) -> Any:
        height, width = image.shape[:2]
        total_pixels = height * width
        amount = self.rng.uniform(*self.seasoning_range)
        total_holes = max(
            1,
            int(total_pixels * amount),
        )
        num_white = self.rng.randint(
            0,
            total_holes,
        )
        num_black = total_holes - num_white
        output = image.copy()
        self._apply_points(
            image=output,
            count=num_white,
            value=255,
        )
        self._apply_points(
            image=output,
            count=num_black,
            value=0,
        )
        return output

    def _apply_points(
        self,
        image: Any,
        count: int,
        value: int,
    ) -> None:
        height, width = image.shape[:2]
        for _ in range(count):
            y = self.rng.randrange(height)
            x = self.rng.randrange(width)
            image[y, x] = value


class _RandomCoarseDropout:
    def __init__(
        self,
        rng: random.Random,
        num_holes_range: Tuple[int, int],
        hole_height_range: Tuple[int, int],
        hole_width_range: Tuple[int, int],
    ) -> None:
        self.rng = rng
        self.num_holes_range = num_holes_range
        self.hole_height_range = hole_height_range
        self.hole_width_range = hole_width_range

    def __call__(
        self,
        image: Any,
        **params: Any,
    ) -> Any:
        output = image.copy()
        total_holes = self.rng.randint(*self.num_holes_range)
        num_white = self.rng.randint(
            0,
            total_holes,
        )
        num_black = total_holes - num_white
        self._apply_rectangles(
            image=output,
            count=num_white,
            value=255,
        )
        self._apply_rectangles(
            image=output,
            count=num_black,
            value=0,
        )
        return output

    def _apply_rectangles(
        self,
        image: Any,
        count: int,
        value: int,
    ) -> None:
        height, width = image.shape[:2]
        for _ in range(count):
            hole_height = min(
                height,
                self.rng.randint(*self.hole_height_range),
            )
            hole_width = min(
                width,
                self.rng.randint(*self.hole_width_range),
            )
            y0 = self.rng.randint(
                0,
                max(
                    0,
                    height - hole_height,
                ),
            )
            x0 = self.rng.randint(
                0,
                max(
                    0,
                    width - hole_width,
                ),
            )
            image[y0 : y0 + hole_height, x0 : x0 + hole_width] = value


class _FaxLikeNoise:
    def __init__(
        self,
        cv2_module: Any,
        np_module: Any,
        rng: random.Random,
        seed: int,
    ) -> None:
        self.cv2 = cv2_module
        self.np = np_module
        self.rng = rng
        self.np_rng = np_module.random.default_rng(seed)
        self.fax_noise_params = [
            {
                "stripe_noise_scale": 50,
                "uniform_noise_scale": 150,
                "binary_threshold": 150,
            },
            {
                "stripe_noise_scale": 80,
                "uniform_noise_scale": 80,
                "binary_threshold": 160,
            },
            {
                "stripe_noise_scale": 127,
                "uniform_noise_scale": 127,
                "binary_threshold": 127,
            },
        ]

    def __call__(
        self,
        image: Any,
        **params: Any,
    ) -> Any:
        height, width = image.shape[:2]
        noise = self.rng.choice(self.fax_noise_params)
        stripe_noise_stride = self.rng.randint(
            2,
            5,
        )
        threshold = noise["binary_threshold"]
        stripe_noise_scale = noise["stripe_noise_scale"]
        uniform_noise_scale = noise["uniform_noise_scale"]
        scale_factor = 255 / (255 + stripe_noise_scale + uniform_noise_scale)
        stripe_noise_scale = int(scale_factor * stripe_noise_scale)
        uniform_noise_scale = int(scale_factor * uniform_noise_scale)

        noise_stripe = self.np.full(
            (height, width),
            stripe_noise_scale,
            dtype=self.np.uint8,
        )
        if self.rng.random() < 0.75:
            noise_stripe[:, ::stripe_noise_stride] = 0
        else:
            noise_stripe[::stripe_noise_stride, :] = 0

        shift_mask = self.np.array(
            [self.rng.random() > 0.8 for _ in range(height)],
            dtype=bool,
        )
        if shift_mask.any():
            noise_stripe[shift_mask, :] = self.np.tile(
                self.np.roll(
                    noise_stripe[0],
                    1,
                ),
                (int(shift_mask.sum()), 1),
            )

        image_gray = self.cv2.cvtColor(
            image,
            self.cv2.COLOR_RGB2GRAY,
        ).astype(self.np.float32)
        random_noise = self.np_rng.uniform(
            0,
            uniform_noise_scale,
            size=(height, width),
        )
        image_with_noise = self.np.clip(
            image_gray * scale_factor + random_noise + noise_stripe,
            0,
            255,
        ).astype(self.np.uint8)
        _, thresholded = self.cv2.threshold(
            image_with_noise,
            threshold,
            255,
            self.cv2.THRESH_BINARY,
        )
        return self.np.repeat(
            thresholded[:, :, None],
            3,
            axis=2,
        )


class _AdaptiveThreshold:
    def __init__(
        self,
        cv2_module: Any,
        block_size: int,
        c: int,
    ) -> None:
        self.cv2 = cv2_module
        self.block_size = block_size
        self.c = c

    def __call__(
        self,
        image: Any,
        **params: Any,
    ) -> Any:
        image_gray = self.cv2.cvtColor(
            image,
            self.cv2.COLOR_RGB2GRAY,
        )
        image_thresh = self.cv2.adaptiveThreshold(
            image_gray,
            255,
            self.cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            self.cv2.THRESH_BINARY,
            self.block_size,
            self.c,
        )
        return self.cv2.merge((image_thresh, image_thresh, image_thresh))


class _AlbumentationsImageAugmenter:
    def __init__(
        self,
        config: _AlbumentationsImageAugmentationConfig,
        seed: int,
    ) -> None:
        self.config = config
        self.seed = seed + config.seed_offset
        self.rng = random.Random(self.seed)
        self.A, self.cv2, self.np = _load_albumentations_modules()
        self.transform = self._build_transform()

    def __call__(
        self,
        image: Image.Image,
    ) -> Image.Image:
        image = image.convert("RGB")
        if self.rng.random() > self.config.probability:
            return image
        if self.transform is None:
            return image

        image_array = self.np.array(image)
        augmented = self.transform(image=image_array)
        return Image.fromarray(augmented["image"]).convert("RGB")

    def _build_transform(
        self,
    ) -> Optional[Any]:
        transforms = []
        self._append_resize_transform(transforms=transforms)
        self._append_rotate_transform(transforms=transforms)
        self._append_blur_transform(transforms=transforms)
        self._append_noise_transform(transforms=transforms)
        self._append_seasoning_transform(transforms=transforms)
        self._append_coarse_dropout_transform(transforms=transforms)
        self._append_scan_transforms(transforms=transforms)
        self._append_color_transform(transforms=transforms)
        self._append_hsv_transform(transforms=transforms)
        self._append_rgb_shift_transform(transforms=transforms)
        self._append_jpeg_transform(transforms=transforms)
        self._append_weather_transform(transforms=transforms)
        if not transforms:
            return None
        return self.A.Compose(
            transforms=transforms,
            p=1.0,
            seed=self.seed,
        )

    def _append_resize_transform(
        self,
        transforms: List[Any],
    ) -> None:
        resize_config = self.config.resize
        probability = _validate_probability(
            value=resize_config["probability"],
            name="image_augmentation.albumentations.resize.probability",
        )
        if probability <= 0:
            return
        transforms.append(
            self.A.Lambda(
                image=_AspectRatioPreservingResize(
                    cv2_module=self.cv2,
                    rng=random.Random(self._next_seed()),
                    scale_percent_range=_to_float_range(
                        value=resize_config["scale_percent_range"],
                        name="image_augmentation.albumentations.resize.scale_percent_range",
                    ),
                ),
                name="aspect_ratio_preserving_resize",
                p=probability,
            )
        )

    def _append_rotate_transform(
        self,
        transforms: List[Any],
    ) -> None:
        rotate_config = self.config.rotate
        probability = _validate_probability(
            value=rotate_config["probability"],
            name="image_augmentation.albumentations.rotate.probability",
        )
        if probability <= 0:
            return
        transforms.append(
            self.A.Rotate(
                limit=_to_float_range(
                    value=rotate_config["limit"],
                    name="image_augmentation.albumentations.rotate.limit",
                ),
                border_mode=self.cv2.BORDER_CONSTANT,
                fill=(255, 255, 255),
                p=probability,
            )
        )

    def _append_blur_transform(
        self,
        transforms: List[Any],
    ) -> None:
        blur_config = self.config.blur
        probability = _validate_probability(
            value=blur_config["probability"],
            name="image_augmentation.albumentations.blur.probability",
        )
        if probability <= 0:
            return

        blur_transforms = []
        blur_types = _validate_names(
            values=blur_config["types"],
            allowed=["average", "median", "motion", "gaussian", "lowres"],
            name="image_augmentation.albumentations.blur.types",
        )
        if "average" in blur_types:
            blur_transforms.append(
                self.A.Blur(
                    blur_limit=_to_int_range(
                        value=blur_config["average"],
                        name="image_augmentation.albumentations.blur.average",
                    ),
                    p=1.0,
                )
            )
        if "median" in blur_types:
            blur_transforms.append(
                self.A.MedianBlur(
                    blur_limit=_to_int_range(
                        value=blur_config["median"],
                        name="image_augmentation.albumentations.blur.median",
                    ),
                    p=1.0,
                )
            )
        if "motion" in blur_types:
            blur_transforms.append(
                self.A.MotionBlur(
                    blur_limit=_to_int_range(
                        value=blur_config["motion"],
                        name="image_augmentation.albumentations.blur.motion",
                    ),
                    p=1.0,
                )
            )
        if "gaussian" in blur_types:
            blur_transforms.append(
                self.A.GaussianBlur(
                    blur_limit=0,
                    sigma_limit=_to_float_range(
                        value=blur_config["gaussian_sigma"],
                        name="image_augmentation.albumentations.blur.gaussian_sigma",
                    ),
                    p=1.0,
                )
            )
        if "lowres" in blur_types:
            blur_transforms.append(
                self.A.Lambda(
                    image=_LowResBlur(
                        cv2_module=self.cv2,
                        rng=random.Random(self._next_seed()),
                        factor_range=_to_float_range(
                            value=blur_config["lowres_factor"],
                            name="image_augmentation.albumentations.blur.lowres_factor",
                        ),
                    ),
                    name="lowres_blur",
                    p=1.0,
                )
            )
        if blur_transforms:
            transforms.append(
                self.A.OneOf(
                    transforms=blur_transforms,
                    p=probability,
                )
            )

    def _append_noise_transform(
        self,
        transforms: List[Any],
    ) -> None:
        noise_config = self.config.noise
        probability = _validate_probability(
            value=noise_config["probability"],
            name="image_augmentation.albumentations.noise.probability",
        )
        if probability <= 0:
            return

        noise_transforms = []
        noise_types = _validate_names(
            values=noise_config["types"],
            allowed=["gaussian", "laplace", "poisson"],
            name="image_augmentation.albumentations.noise.types",
        )
        if "gaussian" in noise_types:
            noise_transforms.append(
                self.A.GaussNoise(
                    std_range=_to_float_range(
                        value=noise_config["gaussian_std"],
                        name="image_augmentation.albumentations.noise.gaussian_std",
                    ),
                    p=1.0,
                )
            )
        if "laplace" in noise_types:
            noise_transforms.append(
                self.A.ISONoise(
                    color_shift=_to_float_range(
                        value=noise_config["laplace_color_shift"],
                        name="image_augmentation.albumentations.noise.laplace_color_shift",
                    ),
                    intensity=_to_float_range(
                        value=noise_config["laplace_intensity"],
                        name="image_augmentation.albumentations.noise.laplace_intensity",
                    ),
                    p=1.0,
                )
            )
        if "poisson" in noise_types:
            noise_transforms.append(
                self.A.MultiplicativeNoise(
                    multiplier=_to_float_range(
                        value=noise_config["poisson_multiplier"],
                        name="image_augmentation.albumentations.noise.poisson_multiplier",
                    ),
                    p=1.0,
                )
            )
        if noise_transforms:
            transforms.append(
                self.A.OneOf(
                    transforms=noise_transforms,
                    p=probability,
                )
            )

    def _append_seasoning_transform(
        self,
        transforms: List[Any],
    ) -> None:
        seasoning_config = self.config.seasoning
        probability = _validate_probability(
            value=seasoning_config["probability"],
            name="image_augmentation.albumentations.seasoning.probability",
        )
        if probability <= 0:
            return
        transforms.append(
            self.A.Lambda(
                image=_RandomSeasoning(
                    rng=random.Random(self._next_seed()),
                    seasoning_range=_to_float_range(
                        value=seasoning_config["seasoning_range"],
                        name="image_augmentation.albumentations.seasoning.seasoning_range",
                    ),
                ),
                name="random_seasoning",
                p=probability,
            )
        )

    def _append_coarse_dropout_transform(
        self,
        transforms: List[Any],
    ) -> None:
        coarse_dropout_config = self.config.coarse_dropout
        probability = _validate_probability(
            value=coarse_dropout_config["probability"],
            name="image_augmentation.albumentations.coarse_dropout.probability",
        )
        if probability <= 0:
            return
        transforms.append(
            self.A.Lambda(
                image=_RandomCoarseDropout(
                    rng=random.Random(self._next_seed()),
                    num_holes_range=_to_int_range(
                        value=coarse_dropout_config["num_holes_range"],
                        name="image_augmentation.albumentations.coarse_dropout.num_holes_range",
                    ),
                    hole_height_range=_to_int_range(
                        value=coarse_dropout_config["hole_height_range"],
                        name="image_augmentation.albumentations.coarse_dropout.hole_height_range",
                    ),
                    hole_width_range=_to_int_range(
                        value=coarse_dropout_config["hole_width_range"],
                        name="image_augmentation.albumentations.coarse_dropout.hole_width_range",
                    ),
                ),
                name="random_coarse_dropout",
                p=probability,
            )
        )

    def _append_scan_transforms(
        self,
        transforms: List[Any],
    ) -> None:
        scan_config = self.config.scan
        fax_probability = _validate_probability(
            value=scan_config["fax_probability"],
            name="image_augmentation.albumentations.scan.fax_probability",
        )
        if fax_probability > 0:
            transforms.append(
                self.A.Lambda(
                    image=_FaxLikeNoise(
                        cv2_module=self.cv2,
                        np_module=self.np,
                        rng=random.Random(self._next_seed()),
                        seed=self._next_seed(),
                    ),
                    name="fax_like_noise",
                    p=fax_probability,
                )
            )

        black_white_probability = _validate_probability(
            value=scan_config["black_white_probability"],
            name="image_augmentation.albumentations.scan.black_white_probability",
        )
        if black_white_probability <= 0:
            return
        transforms.append(
            self.A.Lambda(
                image=_AdaptiveThreshold(
                    cv2_module=self.cv2,
                    block_size=_to_odd_int(
                        value=scan_config["adaptive_threshold_block_size"],
                        name="image_augmentation.albumentations.scan.adaptive_threshold_block_size",
                    ),
                    c=int(scan_config["adaptive_threshold_c"]),
                ),
                name="adaptive_threshold",
                p=black_white_probability,
            )
        )

    def _append_color_transform(
        self,
        transforms: List[Any],
    ) -> None:
        color_config = self.config.color
        probability = _validate_probability(
            value=color_config["probability"],
            name="image_augmentation.albumentations.color.probability",
        )
        if probability <= 0:
            return

        color_transforms = []
        color_types = _validate_names(
            values=color_config["types"],
            allowed=[
                "grayscale",
                "quantization",
                "hue",
                "saturation",
            ],
            name="image_augmentation.albumentations.color.types",
        )
        if "grayscale" in color_types:
            color_transforms.append(self.A.ToGray(p=1.0))
        if "quantization" in color_types:
            color_transforms.append(
                self.A.Posterize(
                    num_bits=int(color_config["posterize_num_bits"]),
                    p=1.0,
                )
            )
        if "hue" in color_types:
            color_transforms.append(
                self.A.HueSaturationValue(
                    hue_shift_limit=float(color_config["hue_shift_limit"]),
                    sat_shift_limit=0,
                    val_shift_limit=0,
                    p=1.0,
                )
            )
        if "saturation" in color_types:
            color_transforms.append(
                self.A.HueSaturationValue(
                    hue_shift_limit=0,
                    sat_shift_limit=float(color_config["saturation_shift_limit"]),
                    val_shift_limit=0,
                    p=1.0,
                )
            )
        if color_transforms:
            transforms.append(
                self.A.OneOf(
                    transforms=color_transforms,
                    p=probability,
                )
            )

    def _append_hsv_transform(
        self,
        transforms: List[Any],
    ) -> None:
        hsv_config = self.config.hsv
        probability = _validate_probability(
            value=hsv_config["probability"],
            name="image_augmentation.albumentations.hsv.probability",
        )
        if probability <= 0:
            return
        transforms.append(
            self.A.HueSaturationValue(
                hue_shift_limit=_to_float_range(
                    value=hsv_config["hue_shift"],
                    name="image_augmentation.albumentations.hsv.hue_shift",
                ),
                sat_shift_limit=_to_float_range(
                    value=hsv_config["saturation_shift"],
                    name="image_augmentation.albumentations.hsv.saturation_shift",
                ),
                val_shift_limit=_to_float_range(
                    value=hsv_config["value_shift"],
                    name="image_augmentation.albumentations.hsv.value_shift",
                ),
                p=probability,
            )
        )

    def _append_rgb_shift_transform(
        self,
        transforms: List[Any],
    ) -> None:
        rgb_shift_config = self.config.rgb_shift
        probability = _validate_probability(
            value=rgb_shift_config["probability"],
            name="image_augmentation.albumentations.rgb_shift.probability",
        )
        if probability <= 0:
            return
        transforms.append(
            self.A.RGBShift(
                r_shift_limit=_to_float_range(
                    value=rgb_shift_config["red_shift"],
                    name="image_augmentation.albumentations.rgb_shift.red_shift",
                ),
                g_shift_limit=_to_float_range(
                    value=rgb_shift_config["green_shift"],
                    name="image_augmentation.albumentations.rgb_shift.green_shift",
                ),
                b_shift_limit=_to_float_range(
                    value=rgb_shift_config["blue_shift"],
                    name="image_augmentation.albumentations.rgb_shift.blue_shift",
                ),
                p=probability,
            )
        )

    def _append_jpeg_transform(
        self,
        transforms: List[Any],
    ) -> None:
        jpeg_config = self.config.jpeg
        probability = _validate_probability(
            value=jpeg_config["probability"],
            name="image_augmentation.albumentations.jpeg.probability",
        )
        if probability <= 0:
            return
        transforms.append(
            self.A.ImageCompression(
                compression_type="jpeg",
                quality_range=_to_int_range(
                    value=jpeg_config["quality_range"],
                    name="image_augmentation.albumentations.jpeg.quality_range",
                ),
                p=probability,
            )
        )

    def _append_weather_transform(
        self,
        transforms: List[Any],
    ) -> None:
        weather_config = self.config.weather
        probability = _validate_probability(
            value=weather_config["probability"],
            name="image_augmentation.albumentations.weather.probability",
        )
        if probability <= 0:
            return

        weather_transforms = []
        weather_types = _validate_names(
            values=weather_config["types"],
            allowed=["rain", "fog", "snow"],
            name="image_augmentation.albumentations.weather.types",
        )
        if "rain" in weather_types:
            weather_transforms.append(self.A.RandomRain(p=1.0))
        if "fog" in weather_types:
            weather_transforms.append(self.A.RandomFog(p=1.0))
        if "snow" in weather_types:
            weather_transforms.append(self.A.RandomSnow(p=1.0))
        if weather_transforms:
            transforms.append(
                self.A.OneOf(
                    transforms=weather_transforms,
                    p=probability,
                )
            )

    def _next_seed(
        self,
    ) -> int:
        return self.rng.randint(
            0,
            2**32 - 1,
        )


def _build_pil_config(
    config: Dict[str, Any],
) -> _PILImageAugmentationConfig:
    _validate_required_keys(
        config=config,
        keys=[
            "rotation_degrees",
            "jpeg_quality_min",
            "jpeg_quality_max",
            "gaussian_blur_max",
            "contrast_min",
            "contrast_max",
            "brightness_min",
            "brightness_max",
            "sharpness_min",
            "sharpness_max",
            "grayscale_probability",
            "noise_std_max",
            "erase_probability",
            "erase_area_min",
            "erase_area_max",
            "ink_bleed_probability",
            "ink_bleed_strength",
        ],
        prefix="image_augmentation",
    )
    return _PILImageAugmentationConfig(
        enabled=bool(config["enabled"]),
        seed_offset=int(config["seed_offset"]),
        probability=_validate_probability(
            value=config["probability"],
            name="image_augmentation.probability",
        ),
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
        grayscale_probability=_validate_probability(
            value=config["grayscale_probability"],
            name="image_augmentation.grayscale_probability",
        ),
        noise_std_max=float(config["noise_std_max"]),
        erase_probability=_validate_probability(
            value=config["erase_probability"],
            name="image_augmentation.erase_probability",
        ),
        erase_area_min=float(config["erase_area_min"]),
        erase_area_max=float(config["erase_area_max"]),
        ink_bleed_probability=_validate_probability(
            value=config["ink_bleed_probability"],
            name="image_augmentation.ink_bleed_probability",
        ),
        ink_bleed_strength=float(config["ink_bleed_strength"]),
    )


def _build_albumentations_config(
    config: Dict[str, Any],
) -> _AlbumentationsImageAugmentationConfig:
    _validate_required_keys(
        config=config,
        keys=["albumentations"],
        prefix="image_augmentation",
    )
    albumentations_config = config["albumentations"]
    _validate_required_keys(
        config=albumentations_config,
        keys=[
            "resize",
            "rotate",
            "blur",
            "noise",
            "seasoning",
            "coarse_dropout",
            "scan",
            "color",
            "hsv",
            "rgb_shift",
            "jpeg",
            "weather",
        ],
        prefix="image_augmentation.albumentations",
    )
    _validate_albumentations_sections(config=albumentations_config)
    return _AlbumentationsImageAugmentationConfig(
        enabled=bool(config["enabled"]),
        seed_offset=int(config["seed_offset"]),
        probability=_validate_probability(
            value=config["probability"],
            name="image_augmentation.probability",
        ),
        resize=albumentations_config["resize"],
        rotate=albumentations_config["rotate"],
        blur=albumentations_config["blur"],
        noise=albumentations_config["noise"],
        seasoning=albumentations_config["seasoning"],
        coarse_dropout=albumentations_config["coarse_dropout"],
        scan=albumentations_config["scan"],
        color=albumentations_config["color"],
        hsv=albumentations_config["hsv"],
        rgb_shift=albumentations_config["rgb_shift"],
        jpeg=albumentations_config["jpeg"],
        weather=albumentations_config["weather"],
    )


def _validate_albumentations_sections(
    config: Dict[str, Any],
) -> None:
    required_by_section = {
        "resize": ["probability", "scale_percent_range"],
        "rotate": ["probability", "limit"],
        "blur": [
            "probability",
            "types",
            "average",
            "median",
            "motion",
            "gaussian_sigma",
            "lowres_factor",
        ],
        "noise": [
            "probability",
            "types",
            "gaussian_std",
            "laplace_color_shift",
            "laplace_intensity",
            "poisson_multiplier",
        ],
        "seasoning": ["probability", "seasoning_range"],
        "coarse_dropout": [
            "probability",
            "num_holes_range",
            "hole_height_range",
            "hole_width_range",
        ],
        "scan": [
            "fax_probability",
            "black_white_probability",
            "adaptive_threshold_block_size",
            "adaptive_threshold_c",
        ],
        "color": [
            "probability",
            "types",
            "posterize_num_bits",
            "hue_shift_limit",
            "saturation_shift_limit",
        ],
        "hsv": ["probability", "hue_shift", "saturation_shift", "value_shift"],
        "rgb_shift": ["probability", "red_shift", "green_shift", "blue_shift"],
        "jpeg": ["probability", "quality_range"],
        "weather": ["probability", "types"],
    }
    for section_name, keys in required_by_section.items():
        _validate_required_keys(
            config=config[section_name],
            keys=keys,
            prefix=f"image_augmentation.albumentations.{section_name}",
        )


def _validate_required_keys(
    config: Dict[str, Any],
    keys: List[str],
    prefix: str,
) -> None:
    missing_keys = [key for key in keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing {prefix} config keys: {missing_keys}")


def _validate_probability(
    value: Any,
    name: str,
) -> float:
    probability = float(value)
    if probability < 0.0 or probability > 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0].")
    return probability


def _validate_names(
    values: Any,
    allowed: List[str],
    name: str,
) -> List[str]:
    names = [
        str(value)
        for value in _to_list(
            value=values,
            name=name,
        )
    ]
    invalid = [value for value in names if value not in allowed]
    if invalid:
        raise ValueError(f"{name} contains unsupported values: {invalid}")
    return names


def _to_float_range(
    value: Any,
    name: str,
) -> Tuple[float, float]:
    values = _to_list(
        value=value,
        name=name,
    )
    if len(values) != 2:
        raise ValueError(f"{name} must be a 2-item range.")
    start = float(values[0])
    end = float(values[1])
    if start > end:
        raise ValueError(f"{name} min must be <= max.")
    return start, end


def _to_int_range(
    value: Any,
    name: str,
) -> Tuple[int, int]:
    start, end = _to_float_range(
        value=value,
        name=name,
    )
    int_start = int(start)
    int_end = int(end)
    if int_start > int_end:
        raise ValueError(f"{name} min must be <= max.")
    return int_start, int_end


def _to_odd_int(
    value: Any,
    name: str,
) -> int:
    int_value = int(value)
    if int_value <= 1 or int_value % 2 == 0:
        raise ValueError(f"{name} must be an odd integer greater than 1.")
    return int_value


def _to_list(
    value: Any,
    name: str,
) -> List[Any]:
    if isinstance(value, str):
        raise ValueError(f"{name} must be a list.")
    try:
        return list(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a list.") from exc


def _load_albumentations_modules() -> Tuple[Any, Any, Any]:
    try:
        os.environ.setdefault(
            "NO_ALBUMENTATIONS_UPDATE",
            "1",
        )
        import albumentations as albumentations_module
        import cv2 as cv2_module
        import numpy as numpy_module
    except ImportError as exc:
        raise ImportError(
            "image_augmentation.backend=albumentations requires albumentations "
            "and opencv-python-headless."
        ) from exc
    return albumentations_module, cv2_module, numpy_module
