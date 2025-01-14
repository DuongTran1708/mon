#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Affine transformation.
"""

from __future__ import annotations

import torchvision.transforms.functional as F
from torch import Tensor
from torchvision.transforms import functional_tensor as F_t

from one.constants import TRANSFORMS
from one.core import *
from one.vision.acquisition import get_image_center4
from one.vision.acquisition import get_image_size
from one.vision.acquisition import is_channel_last
from one.vision.shape import affine_box
from one.vision.shape import horizontal_flip_box
from one.vision.shape import vertical_flip_box


# H1: - Affine ---------------------------------------------------------------

def affine(
    image        : Tensor,
    angle        : float,
    translate    : Ints,
    scale        : float,
    shear        : Floats,
    center       : Ints | None        = None,
    interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
    keep_shape   : bool               = True,
    fill         : Floats             = 0.0,
    padding_mode : PaddingMode_       = PaddingMode.CONSTANT,
    inplace      : bool               = False,
) -> Tensor:
    """
    Apply affine transformation on the image keeping image center invariant.
    
    Args:
        image (Tensor): Image of shape [..., C, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
        angle (float): Rotation angle in degrees between -180 and 180,
            clockwise direction.
        translate (Ints): Horizontal and vertical translations (post-rotation
            translation).
        scale (float): Overall scale.
        shear (Floats): Shear angle value in degrees between -180 to 180,
            clockwise direction. If a sequence is specified, the first value
            corresponds to a shear parallel to the x-axis, while the second
            value corresponds to a shear parallel to the y-axis.
        center (Ints | None): Center of affine transformation.  If None, use 
            the center of the image. Defaults to None.
        interpolation (InterpolationMode_): Desired interpolation mode.
            Default to InterpolationMode.BILINEAR.
        keep_shape (bool): If True, expands the output image to  make it large 
            enough to hold the entire rotated image. If False or omitted, make 
            the output image the same size as the input image. Note that the 
            `keep_shape` flag assumes rotation around the center and no 
            translation. Defaults to True.
        fill (Floats): Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): Desired padding mode.
            Defaults to PaddingMode.CONSTANT.
        inplace (bool): If True, make this operation inplace. Defaults to False.
            
    Returns:
        Transformed image of shape [..., C, H, W].
    """
    assert_tensor(image)
    
    assert_number(angle)
    if isinstance(angle, int):
        angle = float(angle)
    
    translate = to_list(translate)
    assert_sequence_of_length(translate, 2)
   
    if isinstance(scale, int):
        scale = float(scale)
    assert_positive_number(scale)
    
    if isinstance(shear, (int, float)):
        shear = [shear, 0.0]
    if isinstance(shear, tuple):
        shear = list(shear)
    if len(shear) == 1:
        shear = [shear[0], shear[0]]
    assert_sequence_of_length(shear, 2)
    
    if not isinstance(interpolation, InterpolationMode):
        interpolation = InterpolationMode.from_value(interpolation)
    interpolation_value = interpolation.value
    
    if not inplace:
        image = image.clone()
        
    h, w   = get_image_size(image)
    center = (h * 0.5, w * 0.5) if center is None else center  # H, W
    center = list(center[::-1])  # W, H
    
    # If keep shape, find the new width and height bounds
    if not keep_shape:
        matrix  = F._get_inverse_affine_matrix(
            center    = [0, 0],
            angle     = angle,
            translate = [0, 0],
            scale     = 1.0,
            shear     = [0.0, 0.0]
        )
        abs_cos = abs(matrix[0])
        abs_sin = abs(matrix[1])
        new_h   = int(h * abs_cos + w * abs_sin)
        new_w   = int(h * abs_sin + w * abs_cos)
        pad_h   = (new_h - h) / 2
        pad_w   = (new_w - w) / 2
        image   = pad(
            image        = image,
            padding      = (pad_h, pad_w),
            fill         = fill,
            padding_mode = padding_mode
        )
    
    translate_f = [1.0 * t for t in translate]
    matrix      = F._get_inverse_affine_matrix(
        center    = [0, 0],
        angle     = angle,
        translate = translate_f,
        scale     = scale,
        shear     = shear
    )
    return F_t.affine(
        img           = image,
        matrix        = matrix,
        interpolation = interpolation.value,
        fill          = to_list(fill)
    )


def affine_image_box(
    image        : Tensor,
    box          : Tensor,
    angle        : float,
    translate    : Ints,
    scale        : float,
    shear        : Floats,
    center       : Ints | None        = None,
    interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
    keep_shape   : bool               = True,
    fill         : Floats             = 0.0,
    padding_mode : PaddingMode_       = PaddingMode.CONSTANT,
    drop_ratio   : float              = 0.0,
    inplace      : bool               = False,
) -> tuple[Tensor, Tensor]:
    """
    Apply affine transformation on the image keeping image center invariant.
    
    Args:
        image (Tensor): Image of shape [..., C, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
        box (Tensor): Bounding boxes of shape [N, 4]. They are expected to be in
            (x1, y1, x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
        angle (float): Rotation angle in degrees between -180 and 180,
            clockwise direction.
        translate (Ints): Horizontal and vertical translations (post-rotation
            translation).
        scale (float): Overall scale.
        shear (Floats): Shear angle value in degrees between -180 to 180,
            clockwise direction. If a sequence is specified, the first value
            corresponds to a shear parallel to the x-axis, while the second
            value corresponds to a shear parallel to the y-axis.
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationMode_): Desired interpolation mode.
            Default to InterpolationMode.BILINEAR.
        keep_shape (bool): If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill (Floats): Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): Desired padding mode. Defaults to
            PaddingMode.CONSTANT.
        drop_ratio (float): If the fraction of a bounding box left in the image
            after being clipped is less than `drop_ratio` the bounding box is
            dropped. If `drop_ratio==0`, don't drop any bounding boxes.
            Defaults to 0.0.
        inplace (bool): If True, make this operation inplace. Defaults to False.
            
    Returns:
        Transformed image of shape [..., C, H, W].
        Transformed box of shape [N, 4].
    """
    assert_tensor_of_ndim(box, 2)
    image_size = get_image_size(image)
    return \
        affine(
            image         = image,
            angle         = angle,
            translate     = translate,
            scale         = scale,
            shear         = shear,
            center        = center,
            interpolation = interpolation,
            keep_shape    = keep_shape,
            fill          = fill,
            padding_mode  = padding_mode,
            inplace       = inplace,
        ), \
        affine_box(
            box        = box,
            image_size = image_size,
            angle      = angle,
            translate  = translate,
            scale      = scale,
            shear      = shear,
            center     = center,
            drop_ratio = drop_ratio,
            inplace    = inplace,
        )


@TRANSFORMS.register(name="affine")
class Affine(Transform):
    """
    Apply affine transformation on the image keeping image center invariant.
    
    Args:
        angle (float): Rotation angle in degrees between -180 and 180,
            clockwise direction.
        translate (Ints): Horizontal and vertical translations (post-rotation
            translation).
        scale (float): Overall scale.
        shear (Floats): Shear angle value in degrees between -180 to 180,
            clockwise direction. If a sequence is specified, the first value
            corresponds to a shear parallel to the x-axis, while the second
            value corresponds to a shear parallel to the y-axis.
        center (Ints | None): Center of affine transformation.  If None, use
            the center of the image. Defaults to None.
        interpolation (InterpolationMode_): Desired interpolation mode.
            Default to InterpolationMode.BILINEAR.
        keep_shape (bool): If True, expands the output image to  make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill (Floats): Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): Desired padding mode.
            Defaults to PaddingMode.CONSTANT.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        angle        : float,
        translate    : Ints,
        scale        : float,
        shear        : Floats,
        center       : Ints | None        = None,
        interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
        keep_shape   : bool               = True,
        fill         : Floats             = 0.0,
        padding_mode : PaddingMode_       = PaddingMode.CONSTANT,
        inplace      : bool               = False,
        p            : float | None       = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.angle         = angle
        self.translate     = translate
        self.scale         = scale
        self.shear         = shear
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        self.inplace       = inplace
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return \
            affine(
                image         = input,
                angle         = self.angle,
                translate     = self.translate,
                scale         = self.scale,
                shear         = self.shear,
                center        = self.center,
                interpolation = self.interpolation,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ), \
            affine(
                image         = target,
                angle         = self.angle,
                translate     = self.translate,
                scale         = self.scale,
                shear         = self.shear,
                center        = self.center,
                interpolation = self.interpolation,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace
            ) if target is not None else None


# H1: - Crop -----------------------------------------------------------------

def center_crop(
    image      : Tensor,
    output_size: Ints,
    inplace    : bool = False
) -> Tensor:
    """
    Crops an image at the center. If image size is smaller than output size
    along any edge, image is padded with 0 and then center cropped.

    Args:
        image (Tensor): Image of shape [..., C, H, W] to be cropped, where ...
            means an arbitrary number of leading dimensions.
        output_size (Ints): Desired output size of the crop. If size is an int
            instead of sequence like (h, w), a square crop (size, size) is made.
            If provided a sequence of length 1, it will be interpreted as
            (size[0], size[0]).
        inplace (bool): If True, make this operation inplace. Defaults to False.
            
    Returns:
        Cropped image of shape [..., C, H, W].
    """
    output_size      = to_size(output_size)
    image_h, image_w = get_image_size(image)
    crop_h,  crop_w  = output_size
    
    if not inplace:
        image = image.clone()
    
    if crop_w > image_w or crop_h > image_h:
        padding_ltrb = [
            (crop_w - image_w)     // 2 if crop_w > image_w else 0,
            (crop_h - image_h)     // 2 if crop_h > image_h else 0,
            (crop_w - image_w + 1) // 2 if crop_w > image_w else 0,
            (crop_h - image_h + 1) // 2 if crop_h > image_h else 0,
        ]
        image = pad(image, padding_ltrb, fill=0)  # PIL uses fill value 0
        _, image_h, image_w = get_image_size(image)
        if crop_w == image_w and crop_h == image_h:
            return image

    crop_top  = int(round((image_h - crop_h) / 2.0))
    crop_left = int(round((image_w - crop_w) / 2.0))
    return crop(
        image   = image,
        top     = crop_top,
        left    = crop_left,
        height  = crop_h,
        width   = crop_w,
        inplace = inplace
    )


def crop_tblr(
    image  : Tensor,
    top    : int,
    bottom : int,
    left   : int,
    right  : int,
    inplace: bool = False
) -> Tensor:
    """
    Crops an image with top + bottom + left + right value.
    
    Args:
        image (Tensor): Image of shape [..., C, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
        top (int): Top padding.
        bottom (int): Bottom padding.
        left (int): Left padding.
        right (int): Right padding.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        
    Returns:
        Cropped image of shape [..., C, H, W].
    """
    bottom = -bottom if bottom > 0 else bottom
    right  = -right  if right  > 0 else right
    
    if not inplace:
        image = image.clone()

    return image[..., top:bottom, left:right]


def crop(
    image  : Tensor,
    top    : int,
    left   : int,
    height : int,
    width  : int,
    inplace: bool = False
) -> Tensor:
    """
    Crop an image at specified location and output size.
    
    Args:
        image (Tensor): Image of shape [..., C, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
        inplace (bool): If True, make this operation inplace. Defaults to False.
            
    Returns:
        Cropped image of shape [..., C, H, W].
    """
    assert_tensor(image)
    h, w   = get_image_size(image)
    right  = left + width
    bottom = top  + height
    
    if not inplace:
        image = image.clone()
    
    if left < 0 or top < 0 or right > w or bottom > h:
        padding_ltrb = [
            max(-left, 0), max(-top, 0), max(right - w, 0), max(bottom - h, 0)
        ]
        return pad(
            image        = image[..., max(top, 0) : bottom, max(left, 0) : right],
            padding      = padding_ltrb,
            fill         = 0,
            padding_mode = PaddingMode.CONSTANT,
            inplace      = inplace,
        )
    return image[..., top:bottom, left:right]


def crop_zero_region(image: Tensor) -> Tensor:
    """
    Crop the zero region around the non-zero region in image.
    
    Args:
        image (Tensor): Image of shape [C, H, W]to with zeros background.
            
    Returns:
        Cropped image of shape [C, H, W].
    """
    assert_tensor(image)
    if is_channel_last(image):
        cols       = torch.any(image, dim=0)
        rows       = torch.any(image, dim=1)
        xmin, xmax = torch.where(cols)[0][[0, -1]]
        ymin, ymax = torch.where(rows)[0][[0, -1]]
        image      = image[ymin:ymax + 1, xmin:xmax + 1]
    else:
        cols       = torch.any(image, dim=1)
        rows       = torch.any(image, dim=2)
        xmin, xmax = torch.where(cols)[0][[0, -1]]
        ymin, ymax = torch.where(rows)[0][[0, -1]]
        image      = image[:, ymin:ymax + 1, xmin:xmax + 1]
    return image


def five_crop(
    image: Tensor, size: Ints
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Crop an image into four corners and the central crop.
    
    Notes:
        This transform returns a tuple of images and there may be a mismatch in
        the number of inputs and targets your `Dataset` returns.
    
    Args:
        image (Tensor): Image of shape [..., C, H, W] to be cropped, where ...
            means an arbitrary number of leading dimensions.
        size (Ints): Desired output size of the crop. If size is an int instead
            of sequence like (h, w), a square crop (size, size) is made.
            If provided a sequence of length 1, it will be interpreted
            as (size[0], size[0]).

    Returns:
        Tuple of corresponding top left, top right, bottom left, bottom right
        and center crop.
    """
    size = to_size(size)
    assert_sequence_of_length(size, 2)

    image_h, image_w = get_image_size(image)
    crop_h,  crop_w  = size
    if crop_w > image_w or crop_h > image_h:
        msg = "Requested crop size {} is bigger than input size {}"
        raise ValueError(msg.format(size, (image_h, image_w)))

    tl = crop(image, 0, 0, crop_h, crop_w)
    tr = crop(image, 0, image_w - crop_w, crop_h, crop_w)
    bl = crop(image, image_h - crop_h, 0, crop_h, crop_w)
    br = crop(image, image_h - crop_h, image_w - crop_w, crop_h, crop_w)

    center = center_crop(image, [crop_h, crop_w])
    return tl, tr, bl, br, center


def ten_crop(image: Tensor, size: Ints, vflip: bool = False) -> Tensors:
    """
    Generate ten cropped images from the given image. Crop the given image
    into four corners and the central crop plus the flipped version of these
    (horizontal flipping is used by default).
   
    Notes:
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your `Dataset` returns.

    Args:
        image (Tensor): Image of shape [..., C, H, W] to be cropped, where ...
            means an arbitrary number of leading dimensions.
        size (Ints): Desired output size of the crop. If size is an int instead
            of sequence like (h, w), a square crop (size, size) is made.
            If provided a sequence of length 1, it will be interpreted
            as (size[0], size[0]).
        vflip (bool): Use vertical flipping instead of horizontal.
            Defaults to False.

    Returns:
        Corresponding top left, top right, bottom left, bottom right and center
            crop and same for the flipped image.
    """
    size = to_size(size)
    assert_sequence_of_length(size, 2)
    
    first_five = five_crop(image, size)

    if vflip:
        image = vertical_flip(image=image)
    else:
        image = horizontal_flip(image=image)

    second_five = five_crop(image, size)
    return first_five + second_five


@TRANSFORMS.register(name="center_crop")
class CenterCrop(Transform):
    """
    Crops an image at the center.

    Args:
        output_size (Ints): Desired output size of the crop. If size is an int
            instead of sequence like (h, w), a square crop (size, size) is made.
            If provided a sequence of length 1, it will be interpreted as
            (size[0], size[0]).
        inplace (bool): If True, make this operation inplace. Defaults to False.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        output_size: Ints,
        inplace    : bool         = False,
        p          : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.output_size = to_size(size=output_size)
        self.inplace     = inplace
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return \
            center_crop(
                image       = input,
                output_size = self.output_size,
                inplace     = self.inplace,
            ), \
            center_crop(
                image       = target,
                output_size = self.output_size,
                inplace     = self.inplace,
            ) if target is not None else None
    
    
@TRANSFORMS.register(name="crop")
class Crop(Transform):
    """
    Crop an image at specified location and output size.
    
    Args:
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        top    : int,
        left   : int,
        height : int,
        width  : int,
        inplace: bool         = False,
        p      : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.top     = top
        self.left    = left
        self.height  = height
        self.width   = width
        self.inplace = inplace
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return \
            crop(
                image   = input,
                top     = self.top,
                left    = self.left,
                height  = self.height,
                width   = self.width,
                inplace = self.inplace,
            ), \
            crop(
                image   = target,
                top     = self.top,
                left    = self.left,
                height  = self.height,
                width   = self.width,
                inplace = self.inplace,
            ) if target is not None else None


@TRANSFORMS.register(name="five_crop")
class FiveCrop(Transform):
    """
    Crop an image into four corners and the central crop.

    Args:
        size (Ints): Desired output size of the crop. If size is an int instead
            of sequence like (h, w), a square crop (size, size) is made.
            If provided a sequence of length 1, it will be interpreted
            as (size[0], size[0]).
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        size : Ints,
        p    : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.size = to_size(size=size)
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | None
    ]:
        return five_crop(image=input , size=self.size), \
               five_crop(image=target, size=self.size) if target is not None else None
    

@TRANSFORMS.register(name="ten_crop")
class TenCrop(Transform):
    """
    Generate ten cropped images from the given image. Crop the given image
    into four corners and the central crop plus the flipped version of these
    (horizontal flipping is used by default).
   
    Notes:
        This transform returns a tuple of images and there may be a mismatch
        in the number of inputs and targets your `Dataset` returns.

    Args:
        size (Ints): Desired output size of the crop. If size is an int instead
            of sequence like (h, w), a square crop (size, size) is made.
            If provided a sequence of length 1, it will be interpreted
            as (size[0], size[0]).
        vflip (bool): Use vertical flipping instead of horizontal.
            Defaults to False.
        p (float | None): Probability of the image being adjusted. Defaults to
            None means process as normal.
    """
    
    def __init__(
        self,
        size : Ints,
        vflip: bool         = False,
        p    : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.size  = to_size(size=size)
        self.vflip = vflip
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensors, Tensors | None]:
        return ten_crop(image=input,  size=self.size, vflip=self.vflip), \
               ten_crop(image=target, size=self.size, vflip=self.vflip) \
                   if target is not None else None
    
    
# H1: - Flip -----------------------------------------------------------------

def horizontal_flip(image: Tensor, inplace: bool = False) -> Tensor:
    """
    Flip an image horizontally.
    
    Args:
        image (Tensor): Image of shape [..., C, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
        inplace (bool): If True, make this operation inplace. Defaults to False.
            
    Returns:
        Flipped imag eof shape [..., C, H, W].
    """
    assert_tensor(image)
    if not inplace:
        image = image.clone()
    return image.flip(-1)


def horizontal_flip_image_box(
    image: Tensor, box: Tensor, inplace: bool = False,
) -> tuple[Tensor, Tensor]:
    """
    Flip an image and a bounding box horizontally.
    
    Args:
        image (Tensor): Image of shape [..., C, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
        box (Tensor): Box of shape [N, 4] to be transformed.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        
    Returns:
        Flipped image of shape [..., C, H, W].
        Flipped box of shape [N, 4].
    """
    assert_tensor_of_ndim(box, 2)
    center = get_image_center4(image)
    return horizontal_flip(image=image, inplace=inplace), \
           horizontal_flip_box(box=box, image_center=center, inplace=inplace)


def vertical_flip(image: Tensor, inplace: bool = False) -> Tensor:
    """
    Flip an image vertically.
    
    Args:
        image (Tensor): Image of shape [..., C, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        
    Returns:
        Flipped image of shape [..., C, H, W].
    """
    assert_tensor(image)
    if not inplace:
        image = image.clone()
    return image.flip(-2)


def vertical_flip_image_box(
    image: Tensor, box: Tensor, inplace: bool = False
) -> tuple[Tensor, Tensor]:
    """
    Flip image and bounding box vertically.
    
    Args:
        image (Tensor): Image of shape [..., C, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
        box (Tensor): Box of shape [N, 4] to be transformed.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        
    Returns:
        Flipped image of shape [..., C, H, W].
        Flipped box of shape [N, 4].
    """
    assert_tensor_of_ndim(box, 2)
    center = get_image_center4(image)
    return vertical_flip(image=image, inplace=inplace), \
           vertical_flip_box(box=box, image_center=center, inplace=inplace)


@TRANSFORMS.register(name="hflip")
@TRANSFORMS.register(name="horizontal_flip")
class HorizontalFlip(Transform):
    """
    Flip an image horizontally.
    
    Args:
        inplace (bool): If True, make this operation inplace. Defaults to False.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        inplace: bool         = False,
        p      : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.inplace = inplace
        
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return horizontal_flip(image=input,  inplace=self.inplace), \
               horizontal_flip(image=target, inplace=self.inplace) \
                   if target is not None else None
    

@TRANSFORMS.register(name="hflip_image_box")
@TRANSFORMS.register(name="horizontal_flip_image_box")
class HorizontalFlipImageBox(Transform):
    """
    Flip an image and a bounding box horizontally.
    
    Args:
        inplace (bool): If True, make this operation inplace. Defaults to False.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        inplace: bool         = False,
        p      : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.inplace = inplace
    
    # noinspection PyMethodOverriding
    def forward(
        self,
        input : Tensor,
        target: Tensor,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor]:
        return horizontal_flip_image_box(
            image=input, box=target, inplace=self.inplace
        )
   

@TRANSFORMS.register(name="vflip")
@TRANSFORMS.register(name="vertical_flip")
class VerticalFlip(Transform):
    """
    Flip an image vertically.
    
    Args:
        inplace (bool): If True, make this operation inplace. Defaults to False.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        inplace: bool         = False,
        p      : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.inplace = inplace
      
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return vertical_flip(image=input,  inplace=self.inplace), \
               vertical_flip(image=target, inplace=self.inplace) \
                   if target is not None else None
    
    
@TRANSFORMS.register(name="vflip_image_box")
@TRANSFORMS.register(name="vertical_flip_image_box")
class VerticalFlipImageBox(Transform):
    """
    Flip an image and a bounding box vertically.
    
    Args:
        inplace (bool): If True, make this operation inplace. Defaults to False.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        inplace: bool         = False,
        p      : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.inplace = inplace
        
    # noinspection PyMethodOverriding
    def forward(
        self, input: Tensor, target: Tensor, *args, **kwargs
    ) -> tuple[Tensor, Tensor]:
        return vertical_flip_image_box(
            image=input, box=target, inplace=self.inplace
        )


# H1: - Pad ------------------------------------------------------------------

def pad(
    image       : Tensor,
    padding     : Floats,
    fill        : Floats       = 0.0,
    padding_mode: PaddingMode_ = PaddingMode.CONSTANT,
    inplace     : bool         = False,
) -> Tensor:
    """
    Pad an image with fill value.
    
    Args:
        image (Tensor): Image of shape [..., C, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
        padding (Float4T): Padding on each border. If a single int is provided
            this is used to pad all borders. If sequence of length 2 is provided
            this is the padding on left/right and top/bottom respectively.
            If a sequence of length 4 is provided this is the padding for the
            left, top, right and bottom borders respectively.
        fill (Floats): Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): One of: ["constant", "edge", "reflect",
            "symmetric"]
            - constant: pads with a constant value, this value is specified
              with fill.
            - edge: pads with the last value at the edge of the image. If input
              a 5D torch Tensor, the last 3 dimensions will be padded instead
              of the last 2.
            - reflect: pads with reflection of image without repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with 2
              elements on both sides in reflect mode will result in
              [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: pads with reflection of image repeating the last value
              on the edge. For example, padding [1, 2, 3, 4] with 2 elements on
              both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3].
             Default: `constant`.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        
    Returns:
        Padded image of shape [..., C, H, W].
    """
    assert_tensor(image)
    
    if not inplace:
        image = image.clone()
    
    if isinstance(fill, (tuple, list)):
        fill = fill[0]
    assert_number(fill)

    if not isinstance(padding, (int, tuple, list)):
        raise TypeError("Got inappropriate padding arg")
    if isinstance(padding, tuple):
        padding = list(padding)
    if isinstance(padding, list) and len(padding) not in [1, 2, 4]:
        raise ValueError(
            f"`padding` must be an int or a 1, 2, or 4 element tuple. "
            f"But got: {len(padding)}."
        )
    
    padding = F_t._parse_pad_padding(to_list(padding))

    if not isinstance(padding_mode, PaddingMode):
        padding_mode = PaddingMode.from_value(value=padding_mode)
    padding_mode = padding_mode.value
    assert_value_in_collection(padding_mode, ["constant", "edge", "reflect", "symmetric"])
    if padding_mode == "edge":
        padding_mode = "replicate"  # Remap padding_mode str
    elif padding_mode == "symmetric":
        # route to another implementation
        return F_t._pad_symmetric(image, padding)
    
    need_squeeze = False
    if image.ndim < 4:
        image        = image.unsqueeze(dim=0)
        need_squeeze = True
    out_dtype = image.dtype
    need_cast = False
    if (padding_mode != "constant") and image.dtype not in (torch.float32, torch.float64):
        # Here we temporary cast input tensor to float until pytorch issue is
        # resolved: https://github.com/pytorch/pytorch/issues/40763
        need_cast = True
        image     = image.to(torch.float32)
    if padding_mode in ("reflect", "replicate"):
        image = nn.functional.pad(image, padding, mode=padding_mode)
    else:
        image = nn.functional.pad(image, padding, mode=padding_mode, value=fill)
    if need_squeeze:
        image = image.squeeze(dim=0)
    if need_cast:
        image = image.to(out_dtype)
    return image


@TRANSFORMS.register(name="pad")
class Pad(Transform):
    """
    Pad an image with fill value.
    
    Args:
        padding (Float4T): Padding on each border. If a single int is provided
            this is used to pad all borders. If sequence of length 2 is provided
            this is the padding on left/right and top/bottom respectively.
            If a sequence of length 4 is provided this is the padding for the
            left, top, right and bottom borders respectively.
        fill (Floats): Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): One of: ["constant", "edge", "reflect",
            "symmetric"]
            - constant: pads with a constant value, this value is specified
              with fill.
            - edge: pads with the last value at the edge of the image. If input
              a 5D torch Tensor, the last 3 dimensions will be padded instead
              of the last 2.
            - reflect: pads with reflection of image without repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with 2
              elements on both sides in reflect mode will result in
              [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: pads with reflection of image repeating the last value
              on the edge. For example, padding [1, 2, 3, 4] with 2 elements on
              both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3].
             Default: `constant`.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        padding     : Ints,
        fill        : Floats       = 0.0,
        padding_mode: PaddingMode_ = PaddingMode.CONSTANT,
        inplace     : bool         = False,
        p           : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.padding      = padding
        self.fill         = fill
        self.padding_mode = padding_mode
        self.inplace      = inplace
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return \
            pad(
                image        = input,
                padding      = self.padding,
                fill         = self.fill,
                padding_mode = self.padding_mode,
                inplace      = self.inplace,
            ), \
            pad(
                image        = target,
                padding      = self.padding,
                fill         = self.fill,
                padding_mode = self.padding_mode,
                inplace      = self.inplace,
            ) if target is not None else None


# H1: - Resize ---------------------------------------------------------------

def letterbox_resize(
    image     : np.ndarray,
    size      : Ints | None = 768,
    stride    : int         = 32,
    color     : Color       = (114, 114, 114),
    auto      : bool        = True,
    scale_fill: bool        = False,
    scale_up  : bool        = True,
    inplace   : bool        = False,
):
    """
    Resize image to a `stride`-pixel-multiple rectangle.
    
    Notes:
        For YOLOv5, stride = 32.
        For Scaled-YOLOv4, stride = 128
    
    References:
        https://github.com/ultralytics/yolov3/issues/232
        
    Args:
        image:
        
        size:
        
        stride:
        
        color:
        
        auto:
        
        scale_fill:
        
        scale_up:
        
        inplace (bool): If True, make this operation inplace. Defaults to False.
        
    Returns:

    """
    old_size = get_image_size(image)
    
    if size is None:
        return image, None, None, None
    size = to_size(size)
    
    # Scale ratio (new / old)
    r = min(size[0] / old_size[0], size[1] / old_size[1])
    if not scale_up:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio     = r, r  # width, height ratios
    new_unpad = int(round(old_size[1] * r)), int(round(old_size[0] * r))
    dw, dh    = size[1] - new_unpad[0], size[0] - new_unpad[1]  # wh padding
    
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh    = 0.0, 0.0
        new_unpad = (size[1], size[0])
        ratio     = size[1] / old_size[1], size[0] / old_size[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    if old_size[::-1] != new_unpad:  # resize
        image = cv2.resize(
            src=image, dsize=new_unpad, interpolation=cv2.INTER_LINEAR
        )
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image       = cv2.copyMakeBorder(
        src        = image,
        top        = top,
        bottom     = bottom,
        left       = left,
        right      = right,
        borderType = cv2.BORDER_CONSTANT,
        value      = color
    )  # add border
    
    return image, ratio, (dw, dh)


def resize(
    image        : Tensor,
    size         : Ints               = None,
    interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
    antialias    : bool | None        = None,
    inplace      : bool               = False,
) -> Tensor:
    """
    Resize an image. Adapted from: `torchvision.transforms.functional.resize()`
    
    Args:
        image (Tensor): Image of shape [..., C, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
        size (Ints): Desired output size of shape [C, H, W].
        interpolation (InterpolationMode_): Interpolation method.
        antialias (bool | None): Defaults to None.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        
    Returns:
        Resized image of shape [..., C, H, W].
    """
    assert_tensor(image)
    size = (to_size(size))  # H, W
    
    if not inplace:
        image = image.clone()
    
    if not isinstance(interpolation, InterpolationMode):
        interpolation = InterpolationMode.from_value(interpolation)
    if interpolation is InterpolationMode.LINEAR:
        interpolation = InterpolationMode.BILINEAR
    
    return F_t.resize(
        img           = image,
        size          = to_list(size),  # H, W
        interpolation = str(interpolation.value),
        antialias     = antialias
    )


def resized_crop(
    image        : Tensor,
    top          : int,
    left         : int,
    height       : int,
    width        : int,
    size         : Ints | None        = None,
    interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
    inplace      : bool               = False,
) -> Tensor:
    """
    Crop and resize an image.
    
    Notably used in :class:`~torchvision.transforms.RandomResizedCrop`.

    Args:
        image (Tensor): Image of shape [..., C, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
        size (Ints | None): Desired output size of shape [C, H, W].
            Defaults to None.
        interpolation (InterpolationMode_): Interpolation method.
            Defaults to InterpolationMode.BILINEAR.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        
    Returns:
        Resized crop image of shape [..., C, H, W].
    """
    image = crop(
        image   = image,
        top     = top,
        left    = left,
        height  = height,
        width   = width,
        inplace = inplace,
    )
    image = resize(
        image         = image,
        size          = size,
        interpolation = interpolation,
        inplace       = inplace,
    )
    return image


@TRANSFORMS.register(name="resize")
class Resize(Transform):
    """
    Resize an image. Adapted from: `torchvision.transforms.functional.resize()`
    
    Args:
        image (Tensor): Image of shape [..., C, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
        size (Ints): Desired output size of shape [C, H, W].
        interpolation (InterpolationMode_): Interpolation method.
        antialias (bool, None): Defaults to None.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        size         : Ints,
        interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
        antialias    : bool  | None       = None,
        inplace      : bool               = False,
        p            : float | None       = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.size          = size
        self.interpolation = interpolation
        self.antialias     = antialias
        self.inplace       = inplace
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return \
            resize(
                image         = input,
                size          = self.size,
                interpolation = self.interpolation,
                antialias     = self.antialias,
                inplace       = self.inplace,
            ), \
            resize(
                image         = target,
                size          = self.size,
                interpolation = self.interpolation,
                antialias     = self.antialias,
                inplace       = self.inplace,
            ) if target is not None else None
    

@TRANSFORMS.register(name="resized_crop")
class ResizedCrop(Transform):
    """
    Crop and resize an image.
    
    Notably used in :class:`~torchvision.transforms.RandomResizedCrop`.

    Args:
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
        size (Ints | None): Desired output size of shape [C, H, W].
            Defaults to None.
        interpolation (InterpolationMode_): Interpolation method.
            Defaults to InterpolationMode.BILINEAR.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        top          : int,
        left         : int,
        height       : int,
        width        : int,
        size         : list[int],
        interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
        inplace      : bool               = False,
        p            : float | None       = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.top           = top
        self.left          = left
        self.height        = height
        self.width         = width
        self.size          = size
        self.interpolation = interpolation
        self.inplace       = inplace
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return \
            resized_crop(
                image= input,
                top           = self.top,
                left          = self.left,
                height        = self.height,
                width         = self.width,
                size          = self.size,
                interpolation = self.interpolation,
                inplace       = self.inplace,
            ), \
            resized_crop(
                image= target,
                top           = self.top,
                left          = self.left,
                height        = self.height,
                width         = self.width,
                size          = self.size,
                interpolation = self.interpolation,
                inplace       = self.inplace
            ) if target is not None else None


# H1: - Rotate ---------------------------------------------------------------

def rotate(
    image        : Tensor,
    angle        : float,
    center       : Ints | None        = None,
    interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
    keep_shape   : bool               = True,
    fill         : Floats             = 0.0,
    padding_mode : PaddingMode_       = PaddingMode.CONSTANT,
    inplace      : bool               = False,
) -> Tensor:
    """
    Rotate an image.
    
    Args:
        image (Tensor): Image of shape [..., C, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.
        angle (float): Angle to rotate the image.
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationMode_): Desired interpolation mode.
            Default to InterpolationMode.BILINEAR.
        keep_shape (bool): If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Defaults to True.
            Note that the `keep_shape` flag assumes rotation around the center
            and no translation.
        fill (Floats): Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): Desired padding mode.
            Defaults to PaddingMode.CONSTANT.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        
    Returns:
        image (Tensor[..., C, H, W]):
            Rotated image.
    """
    assert_tensor(image)
    return affine(
        image         = image,
        angle         = angle,
        translate     = [0, 0],
        scale         = 1.0,
        shear         = [0, 0],
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        fill          = fill,
        padding_mode  = padding_mode,
        inplace       = inplace
    )


def rotate_horizontal_flip(
    image        : Tensor,
    angle        : float,
    center       : Ints | None        = None,
    interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
    keep_shape   : bool               = True,
    fill         : Floats             = 0.0,
    padding_mode : PaddingMode_       = PaddingMode.CONSTANT,
    inplace      : bool               = False,
) -> Tensor:
    """
    Rotate an image, then flips it vertically.
    
    Args:
        image (Tensor): Image of shape [..., C, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
        angle (float): Angle to rotate the image.
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationMode_): Desired interpolation mode.
            Default to InterpolationMode.BILINEAR.
        keep_shape (bool): If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill (Floats): Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): Desired padding mode.
            Defaults to PaddingMode.CONSTANT.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        
    Returns:
        Rotated and flipped image of shape [..., C, H, W].
    """
    image = rotate(
        image         = image,
        angle         = angle,
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        fill          = fill,
        padding_mode  = padding_mode,
        inplace       = inplace,
    )
    return horizontal_flip(image=image, inplace=inplace)


def rotate_image_box(
    image        : Tensor,
    box          : Tensor,
    angle        : float,
    center       : Ints | None        = None,
    interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
    keep_shape   : bool               = True,
    fill         : Floats             = 0.0,
    padding_mode : PaddingMode_       = PaddingMode.CONSTANT,
    drop_ratio   : float              = 0.0,
    inplace      : bool               = False,
) -> tuple[Tensor, Tensor]:
    """
    Rotate an image and a bounding box.
    
    Args:
        image (Tensor): Image of shape [..., C, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
        box (Tensor[N, 4]): Bounding box of shape [N, 4] to be rotated.
        angle (float): Angle to rotate the image.
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationMode_): Desired interpolation mode.
            Default to InterpolationMode.BILINEAR.
        keep_shape (bool): If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill (Floats): Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): Desired padding mode.
            Defaults to PaddingMode.CONSTANT.
        drop_ratio (float): If the fraction of a bounding box left in the image
            after being clipped is less than `drop_ratio` the bounding box is
            dropped. If `drop_ratio==0`, don't drop any bounding boxes.
            Defaults to 0.0.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        
    Returns:
        Rotated image of shape [..., C, H, W].
        Rotated bounding box of shape [N, 4].
    """
    assert_tensor_of_ndim(box, 2)
    return affine_image_box(
        image         = image,
        box           = box,
        angle         = angle,
        translate     = [0, 0],
        scale         = 1.0,
        shear         = [0, 0],
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        fill          = fill,
        padding_mode  = padding_mode,
        drop_ratio    = drop_ratio,
        inplace       = inplace,
    )


def rotate_vertical_flip(
    image        : Tensor,
    angle        : float,
    center       : Ints | None        = None,
    interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
    keep_shape   : bool               = True,
    fill         : Floats             = 0.0,
    padding_mode : PaddingMode_       = PaddingMode.CONSTANT,
    inplace      : bool               = False,
) -> Tensor:
    """
    Rotate an image, then flips it vertically.
    
    Args:
        image (Tensor): Image of shape [..., C, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
        angle (float): Angle to rotate the image.
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationMode_): Desired interpolation mode.
            Default to InterpolationMode.BILINEAR.
        keep_shape (bool): If True, expands the output image to  make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill (int): Horizontal translation (post-rotation translation).
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): Desired padding mode.
            Defaults to PaddingMode.CONSTANT.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        
    Returns:
        Rotated and flipped image of shape [..., C, H, W].
    """
    image = rotate(
        image         = image,
        angle         = angle,
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        fill          = fill,
        padding_mode  = padding_mode,
        inplace       = inplace
    )
    return vertical_flip(image=image, inplace=inplace)


@TRANSFORMS.register(name="rotate")
class Rotate(Transform):
    """
    Rotate an image.
    
    Args:
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationMode_): Desired interpolation mode.
            Default to InterpolationMode.BILINEAR.
        keep_shape (bool): If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Defaults to True.
            Note that the `keep_shape` flag assumes rotation around the center
            and no translation.
        fill (Floats): Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): Desired padding mode.
            Defaults to PaddingMode.CONSTANT.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        angle        : float,
        center       : Ints | None  = None,
        interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
        keep_shape   : bool               = True,
        fill         : Floats             = 0.0,
        padding_mode : PaddingMode_       = PaddingMode.CONSTANT,
        inplace      : bool               = False,
        p            : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.angle         = angle
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        self.inplace       = inplace
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return \
            rotate(
                image         = input,
                angle         = self.angle,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ), \
            rotate(
                image         = target,
                angle         = self.angle,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ) if target is not None else None


@TRANSFORMS.register(name="rotate_hflip")
@TRANSFORMS.register(name="rotate_horizontal_flip")
class RotateHorizontalFlip(Transform):
    """
    Rotate an image, then flips it horizontally.
    
    Args:
        angle (float): Angle to rotate the image.
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationMode_): Desired interpolation mode.
            Default to InterpolationMode.BILINEAR.
        keep_shape (bool): If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill (Floats): Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): Desired padding mode.
            Defaults to PaddingMode.CONSTANT.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        angle        : float,
        center       : Ints | None  = None,
        interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
        keep_shape   : bool               = True,
        fill         : Floats             = 0.0,
        padding_mode : PaddingMode_       = PaddingMode.CONSTANT,
        inplace      : bool               = False,
        p            : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.angle         = angle
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        self.inplace       = inplace
        
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return \
            rotate_horizontal_flip(
                image         = input,
                angle         = self.angle,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ), \
            rotate_horizontal_flip(
                image         = target,
                angle         = self.angle,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ) if target is not None else None
    

@TRANSFORMS.register(name="rotate_vflip")
@TRANSFORMS.register(name="rotate_vertical_flip")
class RotateVerticalFlip(Transform):
    """
    Rotate an image, then flips it vertically.
    
    Args:
        image (Tensor): Image of shape [..., C, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
        angle (float): Angle to rotate the image.
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationMode_): Desired interpolation mode.
            Default to InterpolationMode.BILINEAR.
        keep_shape (bool): If True, expands the output image to  make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill (int): Horizontal translation (post-rotation translation).
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): Desired padding mode.
            Defaults to PaddingMode.CONSTANT.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        angle        : float,
        center       : Ints | None        = None,
        interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
        keep_shape   : bool               = True,
        fill         : Floats             = 0.0,
        padding_mode : PaddingMode_       = PaddingMode.CONSTANT,
        inplace      : bool               = False,
        p            : float | None       = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.angle         = angle
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        self.inplace       = inplace
        
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return \
            rotate_vertical_flip(
                image         = input,
                angle         = self.angle,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ), \
            rotate_vertical_flip(
                image         = target,
                angle         = self.angle,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ) if target is not None else None
    
    
# H1: - Shear ----------------------------------------------------------------

def horizontal_shear(
    image        : Tensor,
    magnitude    : float,
    center       : Ints | None        = None,
    interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
    keep_shape   : bool               = True,
    fill         : Floats             = 0.0,
    padding_mode : PaddingMode_       = PaddingMode.CONSTANT,
    inplace      : bool               = False,
) -> Tensor:
    """
    Shear an image horizontally.
    
    Args:
        image (Tensor): Image of shape [..., C, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
        magnitude (float): Shear angle value in degrees between -180 to 180,
            clockwise direction.
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationMode_): Desired interpolation mode.
            Default to InterpolationMode.BILINEAR.
        keep_shape (bool): If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill (Floats): Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): Desired padding mode. Defaults to
            PaddingMode.CONSTANT.
        inplace (bool): If True, make this operation inplace. Defaults to False.
            
    Returns:
        Transformed image of shape [..., C, H, W].
    """
    return affine(
        image         = image,
        angle         = 0.0,
        translate     = [0, 0],
        scale         = 1.0,
        shear         = [magnitude, 0.0],
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        fill          = fill,
        padding_mode  = padding_mode,
        inplace       = inplace,
    )


def shear(
    image        : Tensor,
    magnitude    : Floats,
    center       : Ints | None        = None,
    interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
    keep_shape   : bool               = True,
    fill         : Floats             = 0.0,
    padding_mode : PaddingMode_       = PaddingMode.CONSTANT,
    inplace      : bool               = False,
) -> Tensor:
    """
    Shear an image.
    
    Args:
        image (Tensor): Image of shape [..., C, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
        magnitude (Floats): Shear angle value in degrees between -180 to 180,
            clockwise direction. If a sequence is specified, the first value
            corresponds to a shear parallel to the x-axis, while the second
            value corresponds to a shear parallel to the y-axis.
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationMode_): Desired interpolation mode.
            Default to InterpolationMode.BILINEAR.
        keep_shape (bool): If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True
        fill (Floats): Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): Desired padding mode.
            Defaults to PaddingMode.CONSTANT.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        
    Returns:
        Transformed image of shape [..., C, H, W].
    """
    return affine(
        image         = image,
        angle         = 0.0,
        translate     = [0, 0],
        scale         = 1.0,
        shear         = magnitude,
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        fill          = fill,
        padding_mode  = padding_mode,
        inplace       = inplace,
    )


def shear_image_box(
    image        : Tensor,
    box          : Tensor,
    magnitude    : Floats,
    center       : Ints | None        = None,
    interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
    keep_shape   : bool               = True,
    fill         : Floats             = 0.0,
    padding_mode : PaddingMode_       = PaddingMode.CONSTANT,
    drop_ratio   : float              = 0.0,
    inplace      : bool               = False,
) -> tuple[Tensor, Tensor]:
    """
    Shear an image and a bounding box.
    
    Args:
        image (Tensor): Image of shape [..., C, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
        box (Tensor[N, 4]): Bounding box of shape [N, 4] to be sheared.
        magnitude (Floats): Shear angle value in degrees between -180 to 180,
            clockwise direction. If a sequence is specified, the first value
            corresponds to a shear parallel to the x-axis, while the second
            value corresponds to a shear parallel to the y-axis.
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationMode_): Desired interpolation mode.
            Default to InterpolationMode.BILINEAR.
        keep_shape (bool): If True, expands the output image to  make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill (Floats): Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): Desired padding mode.
            Defaults to PaddingMode.CONSTANT.
        drop_ratio (float): If the fraction of a bounding box left in the image
            after being clipped is less than `drop_ratio` the bounding box is
            dropped. If `drop_ratio==0`, don't drop any bounding boxes.
            Defaults to 0.0.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        
    Returns:
        Rotated image of shape [..., C, H, W].
        Rotated bounding box of shape [N, 4].
    """
    assert_tensor_of_ndim(box, 2)
    return affine_image_box(
        image         = image,
        box           = box,
        angle         = 0.0,
        translate     = [0, 0],
        scale         = 1.0,
        shear         = magnitude,
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        fill          = fill,
        padding_mode  = padding_mode,
        drop_ratio    = drop_ratio,
        inplace       = inplace,
    )


def vertical_shear(
    image        : Tensor,
    magnitude    : float,
    center       : Ints | None        = None,
    interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
    keep_shape   : bool               = True,
    fill         : Floats             = 0.0,
    padding_mode : PaddingMode_       = PaddingMode.CONSTANT,
    inplace      : bool               = False,
) -> Tensor:
    """
    Shear an image vertically.
    
    Args:
        image (Tensor): Image of shape [..., C, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
        magnitude (int): Shear angle value in degrees between -180 to 180,
            clockwise direction.
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationMode_): Desired interpolation mode.
            Default to InterpolationMode.BILINEAR.
        keep_shape (bool): If True, expands the output image to  make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill (Floats): Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): Desired padding mode.
            Defaults to PaddingMode.CONSTANT.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        
    Returns:
        Transformed image of shape [..., C, H, W].
    """
    return affine(
        image         = image,
        angle         = 0.0,
        translate     = [0, 0],
        scale         = 1.0,
        shear         = [0.0, magnitude],
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        fill          = fill,
        padding_mode  = padding_mode,
        inplace       = inplace,
    )


@TRANSFORMS.register(name="hshear")
@TRANSFORMS.register(name="horizontal_shear")
class HorizontalShear(Transform):
    """
    Shear an image horizontally.
    
    Args:
        magnitude (float): Shear angle value in degrees between -180 to 180,
            clockwise direction.
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationMode_): Desired interpolation mode.
            Default to InterpolationMode.BILINEAR.
        keep_shape (bool): If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill (Floats): Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): Desired padding mode. Defaults to
            PaddingMode.CONSTANT.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        magnitude    : float,
        center       : Ints | None        = None,
        interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
        keep_shape   : bool               = True,
        fill         : Floats             = 0.0,
        padding_mode : PaddingMode_       = PaddingMode.CONSTANT,
        inplace      : bool               = False,
        p            : float | None       = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        self.inplace       = inplace
        
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return \
            horizontal_shear(
                image         = input,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ), \
            horizontal_shear(
                image         = target,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ) if target is not None else None
    

@TRANSFORMS.register(name="shear")
class Shear(Transform):
    """
    Shear an image.
    
    Args:
        magnitude (Floats): Shear angle value in degrees between -180 to 180,
            clockwise direction. If a sequence is specified, the first value
            corresponds to a shear parallel to the x-axis, while the second
            value corresponds to a shear parallel to the y-axis.
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationMode_): Desired interpolation mode.
            Default to InterpolationMode.BILINEAR.
        keep_shape (bool): If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True
        fill (Floats): Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): Desired padding mode.
            Defaults to PaddingMode.CONSTANT.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        magnitude    : list[float],
        center       : Ints | None        = None,
        interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
        keep_shape   : bool               = True,
        fill         : Floats             = 0.0,
        padding_mode : PaddingMode_       = PaddingMode.CONSTANT,
        inplace      : bool               = False,
        p            : float | None       = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        self.inplace       = inplace
        
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return \
            shear(
                image         = input,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ), \
            shear(
                image         = target,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ) if target is not None else None


@TRANSFORMS.register(name="yshear")
@TRANSFORMS.register(name="vertical_shear")
class VerticalShear(Transform):
    """
    Shear an image vertically.
    
    Args:
        magnitude (int): Shear angle value in degrees between -180 to 180,
            clockwise direction.
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationMode_): Desired interpolation mode.
            Default to InterpolationMode.BILINEAR.
        keep_shape (bool): If True, expands the output image to  make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill (Floats): Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): Desired padding mode.
            Defaults to PaddingMode.CONSTANT.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        magnitude    : float,
        center       : Ints | None        = None,
        interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
        keep_shape   : bool               = True,
        fill         : Floats             = 0.0,
        padding_mode : PaddingMode_       = PaddingMode.CONSTANT,
        inplace      : bool               = False,
        p            : float | None       = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        self.inplace       = inplace
        
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return \
            vertical_shear(
                image         = input,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ), \
            vertical_shear(
                image         = target,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ) if target is not None else None

    
# H1: - Translate ------------------------------------------------------------

def horizontal_translate(
    image        : Tensor,
    magnitude    : int,
    center       : Ints | None        = None,
    interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
    keep_shape   : bool               = True,
    fill         : Floats             = 0.0,
    padding_mode : PaddingMode_       = PaddingMode.CONSTANT,
    inplace      : bool               = False,
) -> Tensor:
    """
    Translate an image in horizontal direction.
    
    Args:
        image (Tensor): Image of shape [..., C, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
        magnitude (int): Horizontal translation (post-rotation translation)
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationMode_): Desired interpolation mode.
            Default to InterpolationMode.BILINEAR.
        keep_shape (bool): If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill (Floats): Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): Desired padding mode. Defaults to
            PaddingMode.CONSTANT.
        inplace (bool): If True, make this operation inplace. Defaults to False.
            
    Returns:
        Transformed image of shape [..., C, H, W].
    """
    return affine(
        image         = image,
        angle         = 0.0,
        translate     = [magnitude, 0],
        scale         = 1.0,
        shear         = [0.0, 0.0],
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        fill          = fill,
        padding_mode  = padding_mode,
        inplace       = inplace,
    )


def horizontal_translate_image_box(
    image        : Tensor,
    box          : Tensor,
    magnitude    : int,
    center       : Ints | None        = None,
    interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
    keep_shape   : bool               = True,
    fill         : Floats             = 0.0,
    padding_mode : PaddingMode_       = PaddingMode.CONSTANT,
    drop_ratio   : float              = 0.0,
    inplace      : bool               = False,
) -> tuple[Tensor, Tensor]:
    """
    Translate an image and a bounding box in horizontal direction.
    
    References:
        https://blog.paperspace.com/data-augmentation-bounding-boxes-scaling-translation/
        
    Args:
        image (Tensor): Image of shape [..., C, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
        box (Tensor): Box of shape [N, 4] to be translated. They are expected 
            to be in (x1, y1, x2, y2) format with `0 <= x1 < x2` and 
            `0 <= y1 < y2`.
        magnitude (int): Horizontal translation (post-rotation translation).
        center (Ints | None): Center of affine transformation. If None, use the 
            center of the image. Defaults to None.
        interpolation (InterpolationMode_): Desired interpolation mode.
            Default to InterpolationMode.BILINEAR.
        keep_shape (bool): If True, expands the output image to make it large 
            enough to hold the entire rotated image. If False or omitted, make 
            the output image the same size as the input image. Note that the 
            `keep_shape` flag assumes rotation around the center and no 
            translation. Defaults to True.
        fill (Floats): Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): Desired padding mode. Defaults to
            PaddingMode.CONSTANT.
        drop_ratio (float): If the fraction of a bounding box left in the image 
            after being clipped is less than `drop_ratio` the bounding box is 
            dropped. If `drop_ratio==0`, don't drop any bounding boxes. 
            Defaults to 0.0.
        inplace (bool): If True, make this operation inplace. Defaults to False.
            
    Returns:
        Transformed image of shape [..., C, H, W].
        Translated box of shape [N, 4].
    """
    return affine_image_box(
        image         = image,
        box           = box,
        angle         = 0.0,
        translate     = [magnitude, 0],
        scale         = 1.0,
        shear         = [0, 0],
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        fill          = fill,
        padding_mode  = padding_mode,
        drop_ratio    = drop_ratio,
        inplace       = inplace,
    )


def translate(
    image        : Tensor,
    magnitude    : Ints,
    center       : Ints | None        = None,
    interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
    keep_shape   : bool               = True,
    fill         : Floats             = 0.0,
    padding_mode : PaddingMode_       = PaddingMode.CONSTANT,
    inplace      : bool               = False,
) -> Tensor:
    """
    Translate an image.
    
    Args:
        image (Tensor): Image of shape [..., C, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
        magnitude (Ints): Horizontal and vertical translations (post-rotation
            translation).
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationMode_): Desired interpolation mode.
            Default to InterpolationMode.BILINEAR.
        keep_shape (bool): If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill (Floats): Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): Desired padding mode.
            Defaults to PaddingMode.CONSTANT.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        
    Returns:
        Transformed image of shape [..., C, H, W].
    """
    return affine(
        image         = image,
        angle         = 0.0,
        translate     = magnitude,
        scale         = 1.0,
        shear         = [0.0, 0.0],
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        fill          = fill,
        padding_mode  = padding_mode,
        inplace       = inplace,
    )


def translate_image_box(
    image        : Tensor,
    box          : Tensor,
    magnitude    : Ints,
    center       : Ints | None        = None,
    interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
    keep_shape   : bool               = True,
    fill         : Floats             = 0.0,
    padding_mode : PaddingMode_       = PaddingMode.CONSTANT,
    drop_ratio   : float              = 0.0,
    inplace      : bool               = False,
) -> tuple[Tensor, Tensor]:
    """
    Translate an image and a bounding box.
    
    References:
        https://blog.paperspace.com/data-augmentation-bounding-boxes-scaling-translation/
        
    Args:
        image (Tensor): Image of shape [..., C, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
        box (Tensor): Box of shape [N, 4] to be translated. They are expected
            to be in (x1, y1, x2, y2) format with `0 <= x1 < x2` and
            `0 <= y1 < y2`.
        magnitude (Ints): Horizontal and vertical translations (post-rotation
            translation).
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationMode_): Desired interpolation mode.
            Default to InterpolationMode.BILINEAR.
        keep_shape (bool): If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill (Floats): Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): Desired padding mode.
            Defaults to PaddingMode.CONSTANT.
        drop_ratio (float): If the fraction of a bounding box left in the image
            after being clipped is less than `drop_ratio` the bounding box is
            dropped. If `drop_ratio==0`, don't drop any bounding boxes.
            Defaults to 0.0.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        
    Returns:
        Transformed image of shape [..., C, H, W].
        Translated box of shape [N, 4].
    """
    assert_tensor_of_ndim(box, 2)
    return affine_image_box(
        image         = image,
        box           = box,
        angle         = 0.0,
        translate     = magnitude,
        scale         = 1.0,
        shear         = [0, 0],
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        fill          = fill,
        padding_mode  = padding_mode,
        drop_ratio    = drop_ratio,
        inplace       = inplace,
    )


def vertical_translate(
    image        : Tensor,
    magnitude    : int,
    center       : Ints | None        = None,
    interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
    keep_shape   : bool               = True,
    fill         : Floats             = 0.0,
    padding_mode : PaddingMode_       = PaddingMode.CONSTANT,
    inplace      : bool               = False,
) -> Tensor:
    """
    Translate an image in vertical direction.
    
    Args:
        image (Tensor): Image of shape [..., C, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
        magnitude (int): Vertical translation (post-rotation translation)
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationMode_): Desired interpolation mode.
            Default to InterpolationMode.BILINEAR.
        keep_shape (bool): If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill (Floats): Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): Desired padding mode.
            Defaults to PaddingMode.CONSTANT.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        
    Returns:
        Transformed image of shape [..., C, H, W].
    """
    return affine(
        image         = image,
        angle         = 0.0,
        translate     = [0, magnitude],
        scale         = 1.0,
        shear         = [0.0, 0.0],
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        fill          = fill,
        padding_mode  = padding_mode,
        inplace       = inplace,
    )


def vertical_translate_image_box(
    image        : Tensor,
    box          : Tensor,
    magnitude    : int,
    center       : Ints | None        = None,
    interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
    keep_shape   : bool               = True,
    fill         : Floats             = 0.0,
    padding_mode : PaddingMode_       = PaddingMode.CONSTANT,
    drop_ratio   : float              = 0.0,
    inplace      : bool               = False,
) -> tuple[Tensor, Tensor]:
    """
    Translate an image and a bounding box in vertical direction.
    
    References:
        https://blog.paperspace.com/data-augmentation-bounding-boxes-scaling-translation/
        
    Args:
        image (Tensor): Image of shape [..., C, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
        box (Tensor): Box of shape [N, 4] to be translated. They are expected
            to be in (x1, y1, x2, y2) format with `0 <= x1 < x2` and
            `0 <= y1 < y2`.
        magnitude (int): Vertical translation (post-rotation translation).
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationMode_): Desired interpolation mode.
            Default to InterpolationMode.BILINEAR.
        keep_shape (bool): If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill (Floats): Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): Desired padding mode.
            Defaults to PaddingMode.CONSTANT.
        drop_ratio (float): If the fraction of a bounding box left in the image
            after being clipped is less than `drop_ratio` the bounding box is
            dropped. If `drop_ratio==0`, don't drop any bounding boxes.
            Defaults to 0.0.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        
    Returns:
        Transformed image of shape [..., C, H, W].
        Translated box of shape [N, 4].
    """
    assert_tensor_of_ndim(box, 2)
    return affine_image_box(
        image         = image,
        box           = box,
        angle         = 0.0,
        translate     = [0, magnitude],
        scale         = 1.0,
        shear         = [0, 0],
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        fill          = fill,
        padding_mode  = padding_mode,
        drop_ratio    = drop_ratio,
        inplace       = inplace,
    )


@TRANSFORMS.register(name="htranslate")
@TRANSFORMS.register(name="horizontal_translate")
class HorizontalTranslate(Transform):
    """
    Translate an image in horizontal direction.
    
    Args:
        magnitude (int): Horizontal translation (post-rotation translation)
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationMode_): Desired interpolation mode.
            Default to InterpolationMode.BILINEAR.
        keep_shape (bool): If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill (Floats): Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): Desired padding mode. Defaults to
            PaddingMode.CONSTANT.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        magnitude    : int,
        center       : Ints | None        = None,
        interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
        keep_shape   : bool               = True,
        fill         : Floats             = 0.0,
        padding_mode : PaddingMode_       = PaddingMode.CONSTANT,
        inplace      : bool               = False,
        p            : float | None       = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        self.inplace       = inplace
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return \
            horizontal_translate(
                image         = input,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ), \
            horizontal_translate(
                image         = target,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ) if target is not None else None
      

@TRANSFORMS.register(name="htranslate_image_box")
@TRANSFORMS.register(name="horizontal_translate_image_box")
class HorizontalTranslateImageBox(Transform):
    """
    Translate an image and a bounding box in horizontal direction.
    
    References:
        https://blog.paperspace.com/data-augmentation-bounding-boxes-scaling-translation/
        
    Args:
        magnitude (int): Horizontal translation (post-rotation translation).
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationMode_): Desired interpolation mode.
            Default to InterpolationMode.BILINEAR.
        keep_shape (bool): If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill (Floats): Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): Desired padding mode. Defaults to
            PaddingMode.CONSTANT.
        drop_ratio (float): If the fraction of a bounding box left in the image
            after being clipped is less than `drop_ratio` the bounding box is
            dropped. If `drop_ratio==0`, don't drop any bounding boxes.
            Defaults to 0.0.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        magnitude    : int,
        center       : Ints | None        = None,
        interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
        keep_shape   : bool               = True,
        fill         : Floats             = 0.0,
        padding_mode : PaddingMode_       = PaddingMode.CONSTANT,
        inplace      : bool               = False,
        p            : float | None       = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        self.inplace       = inplace
    
    # noinspection PyMethodOverriding
    def forward(
        self, input: Tensor, target: Tensor, *args, **kwargs
    ) -> tuple[Tensor, Tensor]:
        return horizontal_translate_image_box(
            image         = input,
            box           = target,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            fill          = self.fill,
            padding_mode  = self.padding_mode,
            inplace       = self.inplace,
        )


@TRANSFORMS.register(name="translate")
class Translate(Transform):
    """
    Translate an image.
    
    Args:
        magnitude (Ints): Horizontal and vertical translations (post-rotation
            translation).
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationMode_): Desired interpolation mode.
            Default to InterpolationMode.BILINEAR.
        keep_shape (bool): If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill (Floats): Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): Desired padding mode.
            Defaults to PaddingMode.CONSTANT.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        magnitude    : Ints,
        center       : Ints | None        = None,
        interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
        keep_shape   : bool               = True,
        fill         : Floats             = 0.0,
        padding_mode : PaddingMode_       = PaddingMode.CONSTANT,
        inplace      : bool               = False,
        p            : float | None       = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        self.inplace       = inplace
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return \
            translate(
                image         = input,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ), \
            translate(
                image         = input,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ) if target is not None else None


@TRANSFORMS.register(name="translate_image_box")
class TranslateImageBox(Transform):
    """
    Translate an image and a bounding box.
    
    References:
        https://blog.paperspace.com/data-augmentation-bounding-boxes-scaling-translation/
        
    Args:
        box (Tensor): Box of shape [N, 4] to be translated. They are expected
            to be in (x1, y1, x2, y2) format with `0 <= x1 < x2` and
            `0 <= y1 < y2`.
        magnitude (Ints): Horizontal and vertical translations (post-rotation
            translation).
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationMode_): Desired interpolation mode.
            Default to InterpolationMode.BILINEAR.
        keep_shape (bool): If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill (Floats): Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): Desired padding mode.
            Defaults to PaddingMode.CONSTANT.
        drop_ratio (float): If the fraction of a bounding box left in the image
            after being clipped is less than `drop_ratio` the bounding box is
            dropped. If `drop_ratio==0`, don't drop any bounding boxes.
            Defaults to 0.0.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        magnitude    : Ints,
        center       : Ints | None        = None,
        interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
        keep_shape   : bool               = True,
        fill         : Floats             = 0.0,
        padding_mode : PaddingMode_       = PaddingMode.CONSTANT,
        inplace      : bool               = False,
        p            : float | None       = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        self.inplace       = inplace
    
    # noinspection PyMethodOverriding
    def forward(
        self, input: Tensor, target: Tensor, *args, **kwargs
    ) -> tuple[Tensor, Tensor]:
        return translate_image_box(
            image         = input,
            box           = target,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            fill          = self.fill,
            padding_mode  = self.padding_mode,
            inplace       = self.inplace,
        )
    

@TRANSFORMS.register(name="vtranslate")
@TRANSFORMS.register(name="vertical_translate")
class VerticalTranslate(Transform):
    """
    Translate an image in vertical direction.
    
    Args:
        magnitude (int): Vertical translation (post-rotation translation)
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationMode_): Desired interpolation mode.
            Default to InterpolationMode.BILINEAR.
        keep_shape (bool): If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill (Floats): Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): Desired padding mode.
            Defaults to PaddingMode.CONSTANT.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        magnitude    : int,
        center       : Ints | None  = None,
        interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
        keep_shape   : bool               = True,
        fill         : Floats             = 0.0,
        padding_mode : PaddingMode_       = PaddingMode.CONSTANT,
        inplace      : bool               = False,
        p            : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        self.inplace       = inplace
        
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return \
            vertical_translate(
                image         = input,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ), \
            vertical_translate(
                image         = target,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ) if target is not None else None


@TRANSFORMS.register(name="vtranslate_image_box")
@TRANSFORMS.register(name="vertical_translate_image_box")
class VerticalTranslateImageBox(Transform):
    """
    Translate an image and a bounding box in vertical direction.
    
    References:
        https://blog.paperspace.com/data-augmentation-bounding-boxes-scaling-translation/
        
    Args:
        magnitude (int): Vertical translation (post-rotation translation).
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationMode_): Desired interpolation mode.
            Default to InterpolationMode.BILINEAR.
        keep_shape (bool): If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill (Floats): Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingMode_): Desired padding mode.
            Defaults to PaddingMode.CONSTANT.
        drop_ratio (float): If the fraction of a bounding box left in the image
            after being clipped is less than `drop_ratio` the bounding box is
            dropped. If `drop_ratio==0`, don't drop any bounding boxes.
            Defaults to 0.0.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        magnitude    : int,
        center       : Ints | None        = None,
        interpolation: InterpolationMode_ = InterpolationMode.BILINEAR,
        keep_shape   : bool               = True,
        fill         : Floats             = 0.0,
        padding_mode : PaddingMode_       = PaddingMode.CONSTANT,
        inplace      : bool               = False,
        p            : float | None       = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        self.inplace       = inplace
    
    # noinspection PyMethodOverriding
    def forward(
        self, input: Tensor, target: Tensor, *args, **kwargs
    ) -> tuple[Tensor, Tensor]:
        return vertical_translate_image_box(
            image         = input,
            box           = target,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            fill          = self.fill,
            padding_mode  = self.padding_mode,
            inplace       = self.inplace,
        )
