# Feature extraction

## Processing parameters

### ImageNet-1k

1. download ImageNet-1k

2. run the following snippet from `camera-bias-private/publishing` (not `camera-bias-private/publishing/feature_extraction`):
   ```
   python -m feature_extraction.extract_embeddings --output_dir /path/to/save/features --dataset ImageNet --data_path /path/to/ImageNet --split train --processing JPEG75_420
   ```

3. repet for val split

4. repet for other processing parameters (JPEG75_444, JPEG85_420, JPEG85_444, JPEG95_420, JPEG95_444, SHARPEN_2, SHARPEN_4, RESIZE_HALF, RESIZE_DOUBLE, INTERPOLATION_BILINEAR, INTERPOLATION_BICUBIC, INTERPOLATION_LANCZOS, INTERPOLATION_BOX)

### iNaturalist2018

1. iNaturalist2018 will be automatically downloaded if not present in the specified folder. The data sould be compatible with [torchvision version of iNaturalist dataset](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.INaturalist.html).

2. run the following snippet from `camera-bias-private/publishing` (not `camera-bias-private/publishing/feature_extraction`):
   ```
   python -m feature_extraction.extract_embeddings --output_dir /path/to/save/features --dataset iNaturalist2018 --data_path /path/to/iNaturalist2018 --split train --processing JPEG75_420
   ```

3. repet for val split

4. repet for other processing parameters (JPEG75_444, JPEG85_420, JPEG85_444, JPEG95_420, JPEG95_444, SHARPEN_2, SHARPEN_4, RESIZE_HALF, RESIZE_DOUBLE, INTERPOLATION_BILINEAR, INTERPOLATION_BICUBIC, INTERPOLATION_LANCZOS, INTERPOLATION_BOX)

## Acquisition parameters

### FlickrExif

1. download images using the URLs from HuggingFace located [here](https://huggingface.co/datasets/CTU-OU/FlickrExif)

2. run the following snippet from `camera-bias-private/publishing` (not `camera-bias-private/publishing/feature_extraction`):
   ```
   python -m feature_extraction.extract_embeddings --output_dir /path/to/save/features --dataset FlickrExif --data_path /path/to/FlickrExif
   ```


### PairCams

1. run the following snippet from `camera-bias-private/publishing` (not `camera-bias-private/publishing/feature_extraction`):
   ```
   python -m feature_extraction.extract_embeddings --output_dir /path/to/save/features --dataset PairCams --batch_size 32 --num_workers 0
   ```