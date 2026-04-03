import glob
import os
import re

import numpy as np
import json
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .utils import tiny_imagenet_wordnet_ids, paircams_model_mapping
import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import torchvision.datasets as tv_datasets
from pillow_heif import register_heif_opener

register_heif_opener()


def _transforms(examples, preprocess=None):
    examples["pixel_values"] = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return examples


def _flickrexif_collate_fn(examples):
    images = []
    ids = []
    for example in examples:
        images.append((example["pixel_values"]))
        ids.append(example["Flickr ID"])

    pixel_values = torch.stack(images)
    ids = torch.tensor(ids, dtype=torch.int64)
    return pixel_values, {"Flickr ID": ids}


def _paircams_collate_fn(examples):
    images = []
    subjects = []
    models = []
    for example in examples:
        images.append((example["pixel_values"]))
        subjects.append(int(example["subject_idx"]))
        models.append(paircams_model_mapping[example["model"].strip().lower()])

    pixel_values = torch.stack(images)
    subjects = torch.tensor(subjects)
    models = torch.tensor(models)
    return pixel_values, {"subject": subjects, "model": models}


def _image_label_collate_fn(examples):
    images = []
    labels = []
    for example in examples:
        images.append((example[0]))
        labels.append(int(example[1]))

    pixel_values = torch.stack(images)
    labels = torch.tensor(labels)
    return pixel_values, {"label": labels}


def get_dataloader(dataset, preprocess, data_path=None, split=None, batch_size=128, num_workers=4):
    if dataset == "FlickrExif":
        dataset = datasets.Dataset.from_list(
            [
                {
                    "image": image_path,
                    "Flickr ID": int(os.path.splitext(os.path.basename(image_path))[0]),
                }
                for image_path in sorted(glob.glob(os.path.join(data_path, "*.*")))
            ]
        ).cast_column("image", datasets.Image())
        dataset.set_transform(
            lambda examples: _transforms(examples, preprocess=preprocess)
        )
        collate_fn = _flickrexif_collate_fn
    elif dataset == "PairCams":
        dataset = load_dataset("CTU-OU/PairCams", split="train")
        dataset.set_transform(lambda examples: _transforms(examples, preprocess=preprocess))
        collate_fn = _paircams_collate_fn
    elif dataset == "ImageNet":
        assert split in ("train", "val"), "split must be 'train' or 'val'"
        dataset = tv_datasets.ImageNet(
            root=data_path,
            split=split,
            transform=preprocess,
        )
        collate_fn = _image_label_collate_fn
    elif dataset == "iNaturalist2018":
        assert split in ("train", "val"), "split must be 'train' or 'val'"
        dataset = tv_datasets.INaturalist(
            root=data_path,
            version="2018",
            download=True,
            transform=preprocess,
            loader=lambda path: Image.open(path).convert("RGB"),
        )
        with open(os.path.join("data/splits/iNaturalist2018/", f"{split}2018.json"), "r") as f:
            split_data = json.load(f)
        split_images = set(i["file_name"].split("/")[-1] for i in split_data["images"])
        indices = [idx for idx, i in enumerate(dataset.index) if i[1] in split_images]
        dataset = torch.utils.data.Subset(dataset, indices)
        collate_fn = _image_label_collate_fn
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented yet")
    
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn
    )
    
    return dataloader


class CustomImageNetES(Dataset):
    """
    Considers only the manual parameter images of the val set. Items are images with two labels: parameter configuration and ImageNet class.
    """

    def __init__(
        self, data_path, split, train_size=0.75, random_state=42, preprocess=None
    ):
        paths = sorted(
            glob.glob(
                os.path.join(
                    data_path, "es-val", "param_control", "l*", "param_*", "n*", "*.*"
                )
            )
        )
        train_paths, test_paths = train_test_split(
            paths, train_size=train_size, random_state=random_state
        )

        assert split in ("train", "test", "val"), (
            f"Unrecognized split name {split} does not match `train`, `test`, or `val`"
        )
        self.paths = train_paths if split == "train" else test_paths
        (
            self.imagenet_labels,
            self.light_labels,
            self.iso_labels,
            self.shutter_speed_labels,
            self.aperture_labels,
        ) = [], [], [], [], []
        for i, fpath in enumerate(self.paths):
            self.imagenet_labels.append(
                tiny_imagenet_wordnet_ids.index(
                    re.search(r"(?<=/)n[^/]+", fpath).group(0)
                )
            )

            light_setup = re.search(r"(?<=/)l(\d+)", fpath).group(1)
            assert light_setup in (
                "1",
                "5",
            ), f"unidentified light setup {light_setup + 1}, expected `1` or `5`"
            self.light_labels.append(0 if light_setup == "1" else 1)

            parameter_setup = int(re.search(r"param_(\d+)", fpath).group(1)) - 1
            assert parameter_setup in range(64), (
                f"unidentified parameter setup {parameter_setup + 1}, expected int from 1 to 64 inclusive"
            )
            aperture_label, shutter_speed_label, iso_label = map(
                int, list(np.base_repr(parameter_setup, base=4).zfill(3))
            )
            self.iso_labels.append(iso_label)
            self.shutter_speed_labels.append(shutter_speed_label)
            self.aperture_labels.append(aperture_label)

        self.preprocess = preprocess or (lambda x: x)

    def __getitem__(self, idx):
        return (
            self.preprocess(Image.open(self.paths[idx]).convert("RGB")),
            {
                "imagenet_label": self.imagenet_labels[idx],
                "light_label": self.light_labels[idx],
                "iso_label": self.iso_labels[idx],
                "shutter_speed_label": self.shutter_speed_labels[idx],
                "aperture_label": self.aperture_labels[idx],
            }
        )

    def __len__(self):
        return len(self.paths)
