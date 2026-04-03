# Near duplicate retrieval

Be sure to first extract the features for PairCams using the provided code.


### Running retrieval

run the following snippet from `camera-bias-private/publishing/near_duplicate_retrieval`:
   ```
   python retrieval.py --model MODEL --variant VARIANT --embeddings_dir /path/to/saved/features --output_dir /path/to/save/results
   ```