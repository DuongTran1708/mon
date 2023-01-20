#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements functions that manipulate pixel intensity in images.

There are many implementations of these functions, but we choose to follow the
same implementation approaches as in :mod:`torchvision`.
"""

from __future__ import annotations

__all__ = [
    "adjust_brightness", "adjust_contrast", "adjust_gamma", "adjust_hue",
    "adjust_saturation", "adjust_sharpness", "autocontrast", "blend",
    "equalize", "erase", "invert", "posterize", "solarize",
]

import torch
from torchvision.transforms import functional_tensor as functional_t

from mon.coreimage import color, util


# region Basic Adjustment

def blend(
    image1: torch.Tensor,
    image2: torch.Tensor,
    alpha : float,
    gamma : float = 0.0
) -> torch.Tensor:
    """Blends 2 images together:
        output = :param:`image1` * alpha + :param:`image2` * beta + gamma

    Args:
        image1: Source image of shape [..., C, H, W].
        image2: Second image of shape [..., C, H, W] that we want to blend on
            top of :param:`image1`.
        alpha: Alpha transparency of the overlay.
        gamma: Scalar added to each sum. Defaults to 0.0.

    Returns:
        Blended image of shape [..., C, H, W].
    """
    image = util.add_weighted(
        image1 = image2,
        alpha  = alpha,
        image2 = image1,
        beta   = 1.0 - alpha,
        gamma  = gamma,
    )
    return image


def adjust_brightness(
    image            : torch.Tensor,
    brightness_factor: float,
) -> torch.Tensor:
    """Adjusts the brightness of the given :param:`image`.

    Args:
        image: Image of shape [..., 1 or 3, H, W] to be adjusted, where ...
            means it can have an arbitrary number of leading dimensions.
        brightness_factor: How much to adjust the brightness. Can be any
            non-negative number. 0 gives a black image, 1 gives the original
            image while 2 increases the brightness by a factor of 2.
        
    Returns:
        Brightness adjusted image of shape [..., 1 or 3, H, W].
    """
    assert brightness_factor >= 0
    assert isinstance(image, torch.Tensor) and image.shape[-3] in [1, 3]
    image = image.clone()
    image = blend(
	    image1 = image,
	    alpha  = brightness_factor,
	    image2 = torch.zeros_like(image),
    )
    return image


def adjust_contrast(
    image          : torch.Tensor,
    contrast_factor: float,
) -> torch.Tensor:
    """Adjusts the contrast of the given :param:`image`.

    Args:
        image: Image of shape [..., 1 or 3, H, W] to be adjusted, where ...
            means it can have an arbitrary number of leading dimensions.
        contrast_factor: How much to adjust the contrast. Can be any
            non-negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        Contrast adjusted image of shape [..., 1 or 3, H, W].
    """
    assert contrast_factor >= 0
    assert isinstance(image, torch.Tensor) and image.shape[-3] in [1, 3]
    c     = util.get_num_channels(image=image)
    dtype = image.dtype if torch.is_floating_point(image) else torch.float32
    if c == 3:
        mean = torch.mean(
            color.rgb_to_grayscale(image=image).to(dtype),
            dim     = (-3, -2, -1),
            keepdim = True,
        )
    else:
        mean = torch.mean(image.to(dtype), dim=(-3, -2, -1), keepdim=True)
    image = image.clone()
    image = blend(image1=image, alpha=contrast_factor, image2=mean)
    return image


def adjust_gamma(
    image: torch.Tensor,
    gamma: float,
    gain : float = 1.0,
) -> torch.Tensor:
    """Adjusts the gamma of the given :param:`image`.

    Args:
        image: Image of shape [..., 1 or 3, H, W] to be adjusted, where ...
            means it can have an arbitrary number of leading dimensions.
        gamma: How much to adjust the gamma. Can be any non-negative number. 0
            gives a black image, 1 gives the original image while 2 increases
            the brightness by a factor of 2.
        gain: Default to 1.0.
        
    Returns:
        Gamma adjusted image of shape [..., 1 or 3, H, W].
    """
    assert gamma >= 0
    assert isinstance(image, torch.Tensor) and image.shape[-3] in [1, 3]
    image  = image.clone()
    dtype  = image.dtype
    if not torch.is_floating_point(image):
        image = functional_t.convert_image_dtype(image, torch.float32)
    image = (gain * image ** gamma).clamp(0, 1)
    image = functional_t.convert_image_dtype(image, dtype)
    return image


def adjust_hue(image: torch.Tensor, hue_factor: float) -> torch.Tensor:
    """Adjusts the hue of the given :param:`image`.

    The image hue is adjusted by converting the image to HSV and cyclically
    shifting the intensities in the hue channel (H). The image is then converted
    back to original image mode.

    :param:`hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See `Hue`_ for more details.

    .. _Hue: https://en.wikipedia.org/wiki/Hue

    Args:
        image: Image of shape [..., 1 or 3, H, W] to be adjusted, where ...
            means it can have an arbitrary number of leading dimensions.
        hue_factor: How much to shift the hue channel. Should be in [-0.5, 0.5].
            0.5 and -0.5 give complete reversal of hue channel in HSV space in
            positive and negative direction respectively. 0 means no shift.
            Therefore, both -0.5 and 0.5 will give an image with complementary
            colors while 0 gives the original image.

    Returns:
        Hue adjusted image of shape [..., 1 or 3, H, W].
    """
    assert -0.5 <= hue_factor <= 0.5
    assert isinstance(image, torch.Tensor) and image.shape[-3] in [1, 3]
    image = image.clone()
    
    dtype = image.dtype
    if image.dtype == torch.uint8:
        image = image.to(dtype=torch.float32) / 255.0

    image   = functional_t._rgb2hsv(image)
    h, s, v = image.unbind(dim=-3)
    h       = (h + hue_factor) % 1.0
    image   = torch.stack((h, s, v), dim=-3)
    image   = functional_t._hsv2rgb(image)
    
    if dtype == torch.uint8:
        image = (image * 255.0).to(dtype=dtype)
    return image


def adjust_saturation(
    image            : torch.Tensor,
    saturation_factor: float,
) -> torch.Tensor:
    """Adjusts the color saturation of the given :param:`image`.

    Args:
        image: Image of shape [..., 1 or 3, H, W] to be adjusted, where ...
            means it can have an arbitrary number of leading dimensions.
        saturation_factor: How much to adjust the saturation. 0 will give a
            black and white image, 1 will give the original image while 2 will
            enhance the saturation by a factor of 2.

    Returns:
        Saturation adjusted image of shape [..., 1 or 3, H, W].
    """
    assert saturation_factor >= 0
    assert isinstance(image, torch.Tensor) and image.shape[-3] in [1, 3]
    if util.get_num_channels(image) == 1:
        return image
    image = image.clone()
    image = blend(
        image1 = image,
        alpha  = saturation_factor,
        image2 = color.rgb_to_grayscale(image=image)
    )
    return image


def adjust_sharpness(
    image           : torch.Tensor,
    sharpness_factor: float,
) -> torch.Tensor:
    """Adjusts the sharpness of the given :param:`image`.
    
    Args:
        image: Image of shape [..., 1 or 3, H, W] to be adjusted , where ...
            means it can have an arbitrary number of leading dimensions.
        sharpness_factor: How much to adjust the sharpness. 0 will give a black
            and white image, 1 will give the original image while 2 will enhance
            the saturation by a factor of 2.
    
    Returns:
        Sharpness adjusted image of shape [..., 1 or 3, H, W].
    """
    assert sharpness_factor >= 0
    assert isinstance(image, torch.Tensor) and image.shape[-3] in [1, 3]
    
    if image.shape[-1] <= 2 or image.shape[-2] <= 2:
        return image
    
    image = blend(
        image1 = image,
        image2 = functional_t._blurred_degenerate_image(img=image),
        alpha  = sharpness_factor,
    )
    return image

# endregion


# region Advanced Adjustment

def autocontrast(image: torch.Tensor) -> torch.Tensor:
    """Maximizes contrast of an image by remapping its pixels per channel so
    that the lowest becomes black and the lightest becomes white.
    
    Args:
        image: Image of shape [..., 1 or 3, H, W] to be adjusted, where ...
        means it can have an arbitrary number of leading dimensions.
    
    Returns:
        Auto-contrast adjusted image of shape [..., 1 or 3, H, W].
    """
    assert isinstance(image, torch.Tensor) and image.shape[-3] in [1, 3] and image.ndim >= 3
    bound            = 1.0 if image.is_floating_point() else 255.0
    dtype            = image.dtype if torch.is_floating_point(image) else torch.float32
    minimum          = image.amin(dim=(-2, -1), keepdim=True).to(dtype)
    maximum          = image.amax(dim=(-2, -1), keepdim=True).to(dtype)
    scale            = bound / (maximum - minimum)
    eq_idxs          = torch.isfinite(scale).logical_not()
    minimum[eq_idxs] = 0
    scale[eq_idxs]   = 1
    return ((image - minimum) * scale).clamp(0, bound).to(image.dtype)


def erase(
    image  : torch.Tensor,
    i      : int,
    j      : int,
    h      : int,
    w      : int,
    v      : torch.Tensor,
) -> torch.Tensor:
    """Erases value in :param:`image` with given value.

    Args:
        image: Image of shape [..., C, H, W] to be adjusted, where ... means it
            can have an arbitrary number of leading dimensions.
        i: i in (i,j) i.e coordinates of the upper left corner.
        j: j in (i,j) i.e coordinates of the upper left corner.
        h: Height of the erased region.
        w: Width of the erased region.
        v: Erasing value.

    Returns:
        Erased image of shape [..., C, H, W].
    """
    assert isinstance(image, torch.Tensor) and image.ndim >= 3
    image = image.clone()
    image[..., i: i + h, j: j + w] = v
    return image


def equalize(image: torch.Tensor) -> torch.Tensor:
    """Equalizes the histogram of the given :param:`image` by applying a non-linear
    mapping to the input in order to create a uniform distribution of grayscale
    values in the output.
    
    Args:
        image: Image of shape [..., 1 or 3, H, W] to be adjusted, where ...
            means it can have an arbitrary number of leading dimensions.
    
    Returns:
        Equalized image of shape [..., 1 or 3, H, W].
    """
    assert isinstance(image, torch.Tensor) \
           and image.shape[-3] in [1, 3] \
           and 3 <= image.ndim <= 4
    
    if image.dtype != torch.uint8:
        raise TypeError(
            f":param:`image` must be a `torch.uint8`. But got: {image.dtype}."
        )
    if image.ndim == 3:
        return functional_t._equalize_single_image(image)
    image = torch.stack([functional_t._equalize_single_image(x) for x in image])
    return image


def invert(image: torch.Tensor) -> torch.Tensor:
    """Inverts the colors of an RGB/grayscale image.
    
    Args:
        image: Image of shape [..., 1 or 3, H, W] to be transformed, where ...
            means it can have an arbitrary number of leading dimensions.
      
    Returns:
        Inverted image of shape [..., 1 or 3, H, W].
    """
    assert isinstance(image, torch.Tensor) and image.shape[-3] in [1, 3]
    image = image.clone()
    bound = torch.tensor(
        data   = 1 if image.is_floating_point() else 255,
        dtype  = image.dtype,
        device = image.device
    )
    image = bound - image
    return image


def posterize(image: torch.Tensor, bits: int) -> torch.Tensor:
    """Posterizes an image by reducing the number of bits for each color
    channel.
    
    Args:
        image: Image of shape [..., C, H, W] to be transformed, where .. means
            it can have an arbitrary number of leading dimensions.
        bits: Number of bits to keep for each channel (0-8).
        
    Returns:
        Posterized image of shape [..., C, H, W].
    """
    assert isinstance(image, torch.Tensor) and image.shape[-3] in [1, 3]
    image = image.clone()
    if image.dtype != torch.uint8:
        raise TypeError(
            f":param:`image` must be a `torch.uint8`. But got: {image.dtype}."
        )
    mask  = -int(2 ** (8 - bits))  # JIT-friendly for: ~(2 ** (8 - bits) - 1)
    image = image & mask
    return image


def solarize(image: torch.Tensor, threshold: float) -> torch.Tensor:
    """Solarizes an RGB/grayscale image by inverting all pixel values above a
    threshold.

    Args:
        image:Image of shape [..., C, H, W] to be transformed, where ... means
            it can have an arbitrary number of leading dimensions.
        threshold: All pixels equal or above this value are inverted.
        
    Returns:
        Solarized image of shape [..., C, H, W].
    """
    assert isinstance(image, torch.Tensor) and image.shape[-3] in [1, 3]
    image = image.clone()
    bound = 1 if image.is_floating_point() else 255
    if threshold > bound:
        raise TypeError(
            f":param:`threshold` must be large than :param:`bound`."
        )
    inverted_img = invert(image)
    image        = torch.where(image >= threshold, inverted_img, image)
    return image

# endregion
