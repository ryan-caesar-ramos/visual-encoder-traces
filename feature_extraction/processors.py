from enum import Enum, auto
import io
from math import sqrt
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import numpy as np

class Processor(Enum):
    IDENTITY = auto()
    JPEG75_420 = auto()
    JPEG75_444 = auto()
    JPEG85_420 = auto()
    JPEG85_444 = auto()
    JPEG95_420 = auto()
    JPEG95_444 = auto()
    SHARPEN_2 = auto()
    SHARPEN_4 = auto()
    RESIZE_HALF = auto()
    RESIZE_DOUBLE = auto()
    INTERPOLATION_BILINEAR = auto()
    INTERPOLATION_BICUBIC = auto()
    INTERPOLATION_LANCZOS = auto()
    INTERPOLATION_BOX = auto()
    MASK_95 = auto()
    

    def get_processing(self):
        if self == Processor.IDENTITY:
            return Identity()
        elif self == Processor.JPEG75_420:
            return JPEGCompression(quality=75, subsampling="4:2:0")
        elif self == Processor.JPEG75_444:
            return JPEGCompression(quality=75, subsampling="4:4:4")
        elif self == Processor.JPEG85_420:
            return JPEGCompression(quality=85, subsampling="4:2:0")
        elif self == Processor.JPEG85_444:
            return JPEGCompression(quality=85, subsampling="4:4:4")
        elif self == Processor.JPEG95_420:
            return JPEGCompression(quality=95, subsampling="4:2:0")
        elif self == Processor.JPEG95_444:
            return JPEGCompression(quality=95, subsampling="4:4:4")
        elif self == Processor.SHARPEN_2:
            return transforms.Compose([Sharpen(sharpening_factor=2), JPEGCompression(quality=95, subsampling="4:4:4")])
        elif self == Processor.SHARPEN_4:
            return transforms.Compose([Sharpen(sharpening_factor=4), JPEGCompression(quality=95, subsampling="4:4:4")])
        elif self == Processor.RESIZE_HALF:
            return transforms.Compose([Rescale(scale_factor=0.5), JPEGCompression(quality=95, subsampling="4:4:4")])
        elif self == Processor.RESIZE_DOUBLE:
            return transforms.Compose([Rescale(scale_factor=2.0), JPEGCompression(quality=95, subsampling="4:4:4")])
        elif self == Processor.INTERPOLATION_BILINEAR:
            return transforms.Compose([Rescale(scale_factor=0.2, is_random=True, interpolation=transforms.InterpolationMode.BILINEAR), JPEGCompression(quality=95, subsampling="4:4:4")])
        elif self == Processor.INTERPOLATION_BICUBIC:
            return transforms.Compose([Rescale(scale_factor=0.2, is_random=True, interpolation=transforms.InterpolationMode.BICUBIC), JPEGCompression(quality=95, subsampling="4:4:4")])
        elif self == Processor.INTERPOLATION_LANCZOS:
            return transforms.Compose([Rescale(scale_factor=0.2, is_random=True, interpolation=transforms.InterpolationMode.LANCZOS), JPEGCompression(quality=95, subsampling="4:4:4")])
        elif self == Processor.INTERPOLATION_BOX:
            return transforms.Compose([Rescale(scale_factor=0.2, is_random=True, interpolation=transforms.InterpolationMode.BOX), JPEGCompression(quality=95, subsampling="4:4:4")])
        elif self == Processor.MASK_95:
            return transforms.Compose([JPEGCompression(quality=95, subsampling="4:4:4"), Mask(mask_ratio=0.95)])
        else:
            raise ValueError(f"Unknown processor: {self}")
        

class Identity:
    def __call__(self, img):
        return img

    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls}()"


class JPEGCompression:
    def __init__(self, quality=96, subsampling="4:4:4"):
        self.quality = quality
        self.subsampling = subsampling

    def __call__(self, img):
        outputIoStream = io.BytesIO()
        img.save(
            outputIoStream,
            "JPEG",
            quality=self.quality,
            subsampling=self.subsampling,
        )
        outputIoStream.seek(0)
        return Image.open(outputIoStream)

    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls}(quality={self.quality}, subsampling={self.subsampling})"
    

class Sharpen:
    def __init__(self, sharpening_factor=1):
        self.sharpening_factor = sharpening_factor

    def __call__(self, img):
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(self.sharpening_factor)
    
    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls}(sharpening_factor={self.sharpening_factor})"
    

class Rescale:
    def __init__(self, scale_factor=0.2, is_random=False, interpolation=transforms.InterpolationMode.BILINEAR):
        self.scale_factor = scale_factor
        self.is_random = is_random
        self.interpolation = interpolation
        self.rng = np.random.default_rng(seed=42)

    def __call__(self, img):
        W, H = img.size
        if self.is_random:
            scale = 1 + self.rng.uniform(-self.scale_factor, self.scale_factor)
        else:
            scale = self.scale_factor

        W = int(W * scale)
        H = int(H * scale)
        return transforms.functional.resize(img, (H, W), interpolation=self.interpolation)
    
    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls}(scale_factor={self.scale_factor}, is_random={self.is_random}, interpolation={self.interpolation})"


class Mask:
    def __init__(self, mask_ratio):
        self.mask_ratio = mask_ratio

    def __call__(self, img):
        width, height = img.size
        side_ratio = sqrt(self.mask_ratio)
        new_width, new_height = side_ratio * width, side_ratio * height
        y_margin = (height - new_height) / 2
        x_margin = (width - new_width) / 2
        y_min, y_max = round(y_margin), round(height - y_margin)
        x_min, x_max = round(x_margin), round(width - x_margin)
        img.paste(0, (x_min, y_min, x_max, y_max))
        return img

    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls}(mask_ratio={self.mask_ratio})"
