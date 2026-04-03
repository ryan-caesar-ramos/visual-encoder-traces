import os
import urllib.request
from functools import partial

import clip
import open_clip
import timm
import torch
import torchvision.models
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models.vision_transformer import VisionTransformer, _cfg
from torch import nn
from torchvision import transforms

registry = {}
variant_registry = {}
custom_variant_registry = {}


def validate_transforms(model_preprocessor):
    # assumption: preprocesses are always:
    #   Resize
    #   CenterCrop
    #   ... arbitrary transforms e.g. conversion to RGB ...
    #   ToTensor
    #   Normalize

    for transformation_idx, transformation in zip(
        (0, 1, -2, -1),
        (
            transforms.Resize,
            transforms.CenterCrop,
            transforms.ToTensor,
            transforms.Normalize,
        ),
    ):
        assert isinstance(
            model_preprocessor.transforms[transformation_idx], transformation
        ), (
            f"transformation at index {transformation_idx} ({model_preprocessor.transforms[transformation_idx]}) does not follow pattern: Resize → ... → ToTensor → Normalize"
        )


def _load_encoder(registry, encoder, *args, **kwargs):
    model_cls = registry[encoder]
    model = model_cls(*args, **kwargs)
    model.eval()
    validate_transforms(model.preprocess)
    model.preprocess.transforms[0].size = min(model.preprocess.transforms[1].size)
    return model


def load_encoder(encoder, variant, device):
    return _load_encoder(registry, encoder, variant, device=device)


def load_custom_encoder(encoder, variant, weights_dir, device):
    return _load_encoder(
        custom_variant_registry,
        encoder,
        variant,
        weights_dir=weights_dir,
        device=device
    )


def register(encoder_name, variants):
    def _(encoder_cls):
        registry[encoder_name] = encoder_cls
        variant_registry[encoder_name] = variants
        return encoder_cls

    return _


def register_custom_model(encoder_name, variants):
    def _(encoder_cls):
        registry[encoder_name] = encoder_cls
        custom_variant_registry[encoder_name] = variants
        return encoder_cls

    return _


@register(
    "clip",
    [
        "ViT-B-16",
        "ViT-B-32",
        "ViT-L-14",
        "ViT-L-14@336px",
        "RN50",
        "RN101",
        "RN50x4",
        "RN50x16",
        "RN50x64",
    ],
)
class CLIP(nn.Module):
    def __init__(self, variant, device):
        super().__init__()
        assert variant in variant_registry["clip"], (
            f"variant `{variant}` not recognized for model class `clip`"
        )

        if variant.startswith("ViT-"):
            arch, size, patch_size = variant.split("-")
            model_id = f"{arch}-{size}/{patch_size}"
        else:
            model_id = variant

        self._encoder, self.preprocess = clip.load(model_id, device=device)
        self.device = device

    def embed_images(self, preprocessed_images):
        return self._encoder.encode_image(preprocessed_images)


@register(
    "openclip",
    [
        "ViT-B-16-laion2B",
        "ViT-B-32-laion2B",
        "ViT-L-14-laion2B",
        "ViT-H-14-laion2B",
        "ViT-g-14-laion2B",
        "ViT-B-16-DataComp.XL",
        "ViT-B-32-DataComp.XL",
        "ViT-L-14-DataComp.XL",
        "convnext_base",
        "convnext_large",
        "convnext_xxlarge",
    ],
)
class OpenCLIP(nn.Module):
    def __init__(self, variant, device):
        super().__init__()
        assert variant in variant_registry["openclip"], (
            f"variant `{variant}` not recognized for model class `openclip`"
        )

        if "DataComp.XL" in variant:
            training_details = "s13B-b90K"
            model_id = f"hf-hub:laion/CLIP-{variant}-{training_details}"
        elif variant.startswith("ViT-") and "laion2B" in variant:
            if "B-16" in variant:
                training_details = "s34b-b88k"
            elif "B-32" in variant:
                training_details = "s34B-b79K"
            elif "L-14" in variant:
                training_details = "s32B-b82K"
            elif "H-14" in variant:
                training_details = "s32B-b79K"
            elif "g-14" in variant:
                training_details = "s34b-b88k"
            model_id = f"hf-hub:laion/CLIP-{variant}-{training_details}"
        elif variant.startswith("convnext_"):
            if variant.endswith("_base"):
                training_details = "_w-laion2b-s13B-b82K-augreg"
            elif variant.endswith("_large"):
                training_details = "_d_320.laion2b-s29b-b131k-ft-soup"
            elif variant.endswith("_xxlarge"):
                training_details = "-laion2b-s34b-b82k-augreg-soup"
            model_id = f"hf-hub:laion/CLIP-{variant}{training_details}"
        self._encoder, _, self.preprocess = open_clip.create_model_and_transforms(
            model_id, device=device
        )

    def embed_images(self, images):
        return self._encoder.encode_image(images)


@register(
    "siglip",
    [
        "base_256",
        "large_256",
    ],
)
class SigLIP(nn.Module):
    def __init__(self, variant, device):
        super().__init__()
        assert variant in variant_registry["siglip"], (
            f"variant `{variant}` not recognized for model class `siglip`"
        )

        vit_variant, image_size = variant.split("_")

        self._encoder = timm.create_model(
            model_name=f"vit_{vit_variant}_patch16_siglip_{image_size}.webli",
            pretrained=True,
            num_classes=0,
        ).to(device)
        self.preprocess = create_transform(
            **resolve_data_config({}, model=self._encoder)
        )
        self.device = device

    def embed_images(self, images):
        return self._encoder(images)


@register(
    "siglip2",
    [
        "base_256",
        "large_256",
    ],
)
class SigLIP2(nn.Module):
    def __init__(self, variant, device):
        super().__init__()
        assert variant in variant_registry["siglip"], (
            f"variant `{variant}` not recognized for model class `siglip`"
        )

        vit_variant, image_size = variant.split("_")

        self._encoder = timm.create_model(
            model_name=f"vit_{vit_variant}_patch16_siglip_{image_size}.v2_webli",
            pretrained=True,
            num_classes=0,
        ).to(device)
        self.preprocess = create_transform(
            **resolve_data_config({}, model=self._encoder)
        )
        self.device = device

    def embed_images(self, images):
        return self._encoder(images)


@register(
    "vit",
    [
        "base_patch16_224",
        "base_patch32_224",
        "large_patch16_224",
        "large_patch32_224",
        "huge_patch14_224",
    ],
)
class ViT(nn.Module):
    def __init__(self, variant, device):
        super().__init__()
        assert variant in variant_registry["vit"], (
            f"variant `{variant}` not recognized for model class `vit`"
        )

        self._encoder = timm.create_model(
            model_name=f"vit_{variant}.orig_in21k", pretrained=True, num_classes=0
        ).to(device)
        self.preprocess = create_transform(
            **resolve_data_config({}, model=self._encoder)
        )

    def embed_images(self, images):
        return self._encoder(images)


@register("resnet", ["50", "101"])
class ResNet(nn.Module):
    def __init__(self, variant, device):
        super().__init__()
        assert variant in variant_registry["resnet"], (
            f"variant `{variant}` not recognized for model class `resnet`"
        )

        self._encoder = timm.create_model(
            model_name=f"resnet{variant}", pretrained=True, num_classes=0
        ).to(device)
        self.preprocess = create_transform(
            **resolve_data_config({}, model=self._encoder)
        )

    def embed_images(self, images):
        return self._encoder(images)


@register(
    "convnext",
    [
        "tiny",
        "base",
        "large",
        "xlarge",
    ],
)
class ConvNeXt(nn.Module):
    def __init__(self, variant, device):
        super().__init__()
        assert variant in variant_registry["convnext"], (
            f"variant `{variant}` not recognized for model class `convnext`"
        )

        model_id = (
            f"convnext_{variant}.fb_in22k_ft_in1k_384"
            if "." not in variant
            else f"convnext_{variant}"
        )
        self._encoder = timm.create_model(
            model_name=model_id, pretrained=True, num_classes=0
        ).to(device)
        self.preprocess = create_transform(
            **resolve_data_config(self._encoder.pretrained_cfg, model=self._encoder)
        )

    def embed_images(self, images):
        return self._encoder(images)


@register("dino", ["vits8", "vits16", "vitb8", "vitb16", "resnet50"])
class DINO(nn.Module):
    def __init__(self, variant, device):
        super().__init__()
        assert variant in variant_registry["dino"], (
            f"variant `{variant}` not recognized for model class `dino`"
        )

        self._encoder = torch.hub.load(
            "facebookresearch/dino:main", f"dino_{variant}"
        ).to(device=device)
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256, interpolation=3),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def embed_images(self, images):
        return self._encoder(images)


@register(
    "dinov2",
    ["vits14_reg", "vitb14_reg", "vitl14_reg", "vitg14_reg"],
)
class DINOv2(nn.Module):
    def __init__(self, variant, device):
        super().__init__()
        assert variant in variant_registry["dinov2"], (
            f"variant `{variant}` not recognized for model class `dinov2`"
        )

        self._encoder = torch.hub.load(
            "facebookresearch/dinov2",
            f"dinov2_{variant}",
        ).to(device=device)
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def embed_images(self, images):
        return self._encoder(images)


@register("mocov3", ["vit_small", "vit_base", "resnet50"])
class MoCoV3(nn.Module):
    def __init__(self, variant, device):
        super().__init__()
        assert variant in variant_registry["mocov3"], (
            f"variant `{variant}` not recognized for model class `mocov3`"
        )

        if variant.startswith("vit"):
            kwargs = {
                "patch_size": 16,
                "depth": 12,
                "num_heads": 12,
                "mlp_ratio": 4,
                "qkv_bias": True,
                "norm_layer": partial(nn.LayerNorm, eps=1e-6),
                "num_classes": 0,
            }

            if variant.endswith("small"):
                kwargs.update({"embed_dim": 384})
            elif variant.endswith("base"):
                kwargs.update({"embed_dim": 768})

            self._encoder = VisionTransformer(**kwargs)
            self.default_cfg = _cfg()
        elif variant == "resnet50":
            self._encoder = torchvision.models.resnet50()
            self._encoder.fc = nn.Identity()

        self._encoder.to(device)

        if variant == "vit_small":
            checkpoint_id = "vit-s-300ep"
            head_name = "head"
        elif variant == "vit_base":
            checkpoint_id = "vit-b-300ep"
            head_name = "head"
        elif variant == "resnet50":
            checkpoint_id = "r-50-1000ep"
            head_name = "fc"

        checkpoint_path = os.path.join("/tmp", f"{checkpoint_id}.pth.tar")

        if not os.path.exists(checkpoint_path):
            url = f"https://dl.fbaipublicfiles.com/moco-v3/{checkpoint_id}/{checkpoint_id}.pth.tar"
            urllib.request.urlretrieve(url, checkpoint_path)

        checkpoint = torch.load(checkpoint_path)["state_dict"]
        checkpoint = {
            k.removeprefix("module.base_encoder."): v
            for k, v in checkpoint.items()
            if (
                k.startswith("module.base_encoder.")
                and not k.startswith(f"module.base_encoder.{head_name}.")
            )
        }

        self._encoder.load_state_dict(checkpoint)

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def embed_images(self, images):
        return self._encoder(images)


@register_custom_model(
    "custom", ["finetune_aug", "finetune_noaug", "scratch_aug", "scratch_noaug"]
)
class CustomOpenCLIP(nn.Module):
    def __init__(self, model_id, weights_dir, device):
        super().__init__()
        pretrained = os.path.join(weights_dir, f"{model_id}.pt")
        self._encoder, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=pretrained, device=device
        )

    def embed_images(self, images):
        return self._encoder.encode_image(images)
