# Linear prediction

## FlickrExif (acquisition prediction)

1. run the following snippet

   ```
   . scripts/acquisition_prediction.sh /path/to/split/data/jsons /path/to/embeddings device /path/to/save/results
   ```

   The above snippet can be appended with `model variant target_attribute`. Empty strings can be used to skip unused arguments. The JSON files for the splits are located at `./split_data`.


## ImageNet/iNaturalist (processing prediction)

1. run the following snippet

   ```
   . scripts/processing_prediction.sh /path/to/embeddings device /path/to/save/results types_of_processing_for_prediction
   ```

   The above snippet can be appended with `model variant`. Empty strings can be used to skip unused arguments.

   To reproduce different experiments from the paper use the following values for `types_of_processing_for_prediction`:

       - JPEG: "JPEG75_420 JPEG75_444 JPEG85_420 JPEG85_444 JPEG95_420 JPEG95_444"
       - Sharpening: "JPEG95_444 SHARPEN_2 SHARPEN_4"
       - Resizing: "JPEG95_444 RESIZE_HALF RESIZE_DOUBLE"
       - Interpolation: "INTERPOLATION_BILINEAR INTERPOLATION_BICUBIC INTERPOLATION_LANCZOS INTERPOLATION_BOX"
