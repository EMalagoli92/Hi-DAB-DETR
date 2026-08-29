import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image

import datasets.transforms as T  # noqa: N812
from datasets.coco import CocoDetection


class DetectionDataset(Dataset):
    """Object Detection dataset."""

    def __init__(
            self,
            annotations_file: str | Path,
            img_dir: str | Path,
            transform: nn.Module | None = None,
            target_transform: nn.Module | None = None,
            dtype: torch.dtype = torch.float32,
        ) -> None:
        """
        Initialize.

        This dataset loads images and their corresponding annotations
        from a COCO-style JSON file. The images are loaded from the
        specified directory, and optional transformations can be applied.

        Parameters
        ----------
        annotations_file : str | Path
            Path to the JSON file containing COCO-style annotations.
        img_dir : str | Path
            Path to the directory containing the image files.
        transform : nn.Module | None, optional
            Transformation to apply to images.
            The default is `None`.
        target_transform : nn.Module | None, optional
            Transformation to apply to annotation targets.
            The default is `None`.
        dtype : torch.dtype, optional
            Data type for bounding boxes.
            The default is `torch.float32`.
        """
        self.annotations = self.preprocess_annotations(annotations_file)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.dtype = dtype

    def preprocess_annotations(self, annotations_file: str | Path) -> dict:
        """
        Pre-process annotations.

        Parameters
        ----------
        annotations_file : str | Path
            Path to the JSON file containing COCO-style annotations.

        Returns
        -------
        dict
            Processed annotations.
        """
        # Read
        with Path(annotations_file).open(mode="r", encoding="utf-8") as handle:
            annotations = json.load(handle)
        annotations_processed = {}

        # Annotations
        annotations_processed["annotations"] = defaultdict(list)
        for ann in annotations["annotations"]:
            annotations_processed["annotations"][ann["image_id"]].append(ann)

        # Images
        # Filter out images with no annotations
        annotations_processed["images"] = [
            img_info for img_info in annotations["images"]
            if img_info["id"] in annotations_processed["annotations"]
        ]

        return annotations_processed


    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.annotations["images"])

    def __getitem__(
            self,
            idx: int
            ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Retrieve an image and its corresponding annotations.

        Parameters
        ----------
        idx : int
            Index of the image in the dataset.

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            - image (torch.Tensor): The image tensor with shape [3, H, W].
            - target (dict[str, torch.Tensor]): A dictionary with:
                - "boxes" (torch.Tensor)
                    Bounding boxes in xyxy format.
                - "labels" (torch.Tensor)
                    Class labels corresponding to objects.
        """
        image_info = self.annotations["images"][idx]
        image = read_image(
            path=self.img_dir.joinpath(image_info["file_name"]),
            mode=ImageReadMode.RGB,
            )
        if self.transform is not None:
            image = self.transform(image)
        image = image.to(self.dtype)

        image_annotations = self.annotations["annotations"][image_info["id"]]
        # xywh --> xyxy
        boxes = torch.tensor(
            [[x, y, x + w, y + h] for x, y, w, h in [
                ann["bbox"] for ann in image_annotations]],
            dtype=self.dtype
            )
        labels = torch.tensor(
            [ann["category_id"]
             for ann in image_annotations],
            dtype=torch.int64
            )
        target = {
            "boxes": boxes,
            "labels": labels,
        }

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target


def make_custom_transforms(
        image_set: Literal["train", "val", "test"],
        fix_size: bool,  # noqa: FBT001
        ):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # config the params for data aug
    scales = [200, 240, 280, 320, 360, 400, 448]
    max_size = 1024
    scales2_resize = [200, 240, 280]
    scales2_crop = [192, 320]

    if image_set == 'train':
        if fix_size:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResize([(max_size, max(scales))]),
                normalize,
            ])

        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(scales2_resize),
                    T.RandomSizeCrop(*scales2_crop),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])

    if image_set in ['val', 'test']:

        return T.Compose([
            T.RandomResize([max(scales)], max_size=max_size),
            normalize,
        ])

    msg = f'unknown {image_set}'
    raise ValueError(msg)


def build_custom(image_set, args):
    root = Path(args.custom_path)

    paths = {
        "train": (root / "images", root / "annotations" / 'trainvalno8k.json'),
        "val": (root / "images", root / "annotations" / '8k.json'),
        }

    img_folder, ann_file = paths[image_set]

    dataset = CocoDetection(
        img_folder=img_folder,
        ann_file=ann_file,
        transforms=make_custom_transforms(
            image_set=image_set,
            fix_size=args.fix_size
            ),
            return_masks=args.masks,
            aux_target_hacks=None,
        )

    return dataset
