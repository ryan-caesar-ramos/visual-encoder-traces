# Processing influence

1. run the following snippet

   ```
   scripts/run_processing_influence.sh /path/to/embeddings device /path/to/save/results types_of_processing_for_prediction k
   ```

   The above snippet can be appended with `model variant` to run only a specific model.

   To reproduce different experiments from the paper use the following values for `types_of_processing_for_prediction`:

       - JPEG: "JPEG75_420 JPEG75_444 JPEG85_420 JPEG85_444 JPEG95_420 JPEG95_444"
       - Sharpening: "JPEG95_444 SHARPEN_2 SHARPEN_4"
       - Resizing: "JPEG95_444 RESIZE_HALF RESIZE_DOUBLE"
       - Interpolation: "INTERPOLATION_BILINEAR INTERPOLATION_BICUBIC INTERPOLATION_LANCZOS INTERPOLATION_BOX"

    For ImageNet use $k=10$ and for iNaturalist2018 $k=1$.
