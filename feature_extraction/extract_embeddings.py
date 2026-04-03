import argparse
import os
from itertools import chain

import torch
from safetensors.torch import save_file
from tqdm.auto import tqdm

from .encoders import (
    custom_variant_registry,
    load_custom_encoder,
    load_encoder,
    variant_registry,
)
from .datasets import get_dataloader
from .processors import Processor
import torchvision.transforms as transforms


def processing_type(name):
    try:
        return Processor[name]
    except KeyError:
        raise argparse.ArgumentTypeError(f"Invalid processing: {name}")


def create_transform(model_preprocessor, processing):

    if processing == processing_type("MASK_95"):
        compose_transforms = model_preprocessor.transforms
        compose_transforms = transforms.Compose(
            compose_transforms[:2]
            + [processing.get_processing()]
            + compose_transforms[2:]
        )
        compose_transforms = [
            processing_type("IDENTITY").get_processing(),
            compose_transforms,
        ]
    else:
        compose_transforms = [processing.get_processing(), model_preprocessor]

    compose_transforms = transforms.Compose(compose_transforms)
    
    return compose_transforms        


def main(args):
    models_and_variants = list(
        chain.from_iterable(
            [
                (model, variant)
                for variant in variants
                if (variant == args.variant if (args.variant and args.model) else True)
            ]
            for model, variants in (
                custom_variant_registry if args.use_custom_models else variant_registry
            ).items()
            if (model == args.model if args.model else True)
        )
    )

    output_dir = os.path.join(args.output_dir, args.dataset)

    if args.processing != Processor.IDENTITY:
        output_dir = os.path.join(output_dir, args.processing.name)

    for model, variant in models_and_variants:
        print(f"extracting embeddings for {model}, {variant}")
        split_ext = ""
        if args.split:
            split_ext = f"_split={args.split}"
        save_path = os.path.join(
            output_dir,
            f"model={model}_variant={variant.replace('/', '-')}{split_ext}.safetensors",
        )

        if os.path.exists(save_path):
            print(f"{save_path} already exists, skipping")
            continue

        device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
        print(f"using device {device}")
        if args.use_custom_models:
            encoder = load_custom_encoder(
                model,
                variant,
                weights_dir=args.custom_model_weights_dir,
                device=device,
            ).eval()
        else:
            encoder = load_encoder(model, variant, device=device).eval()

        dataloader = get_dataloader(
            args.dataset,
            create_transform(encoder.preprocess, args.processing),
            data_path=args.data_path,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        to_save = {"image_embeddings": []}

        with torch.no_grad():
            for i, (batch_images, metadata) in tqdm(
                enumerate(dataloader), total=len(dataloader)
            ):
                batch_images = batch_images.to(device)
                batch_embeddings = encoder.embed_images(batch_images).cpu()
                to_save["image_embeddings"].append(batch_embeddings)
                
                for key, value in metadata.items():
                    if key not in to_save:
                        to_save[key] = []
                    to_save[key].append(value.clone())

        for k, v in to_save.items():
            to_save[k] = torch.cat(v)

        os.makedirs(output_dir, exist_ok=True)

        save_file(
            to_save,
            save_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code for feature extraction"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="If specified, will only train on variants of specified model",
    )
    parser.add_argument(
        "--variant",
        type=str,
        help="If specified, will only train on specified variant of specified model",
    )
    parser.add_argument(
        "--output_dir", type=str, help="Directory to save extracted embeddings to"
    )
    parser.add_argument("--dataset", type=str, default="FlickrExif", help="Dataset to extract embeddings from", choices=["FlickrExif", "PairCams", "ImageNet", "iNaturalist2018"])
    parser.add_argument("--data_path", type=str, help="Path to dataset")
    parser.add_argument("--split", type=str, help="Split to use for datasets that require it (ImageNet, iNaturalist2018)", choices=["train", "val"])
    parser.add_argument(
        "--processing",
        type=processing_type,
        choices=list(Processor),  # for auto-generated help
        help="Choose a processing method",
        default=Processor.IDENTITY,
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU id to use for extracting embeddings",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size to use for extracting embeddings",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers to use for data loading",
    )
    parser.add_argument(
        "--use_custom_models",
        action="store_true",
        help="Flag indicating if custom models will be used to extract embeddings",
    )
    parser.add_argument(
        "--custom_model_weights_dir",
        type=str,
        help="Path to custom model weights, if custom models are being used",
    )
    args = parser.parse_args()

    main(args)
