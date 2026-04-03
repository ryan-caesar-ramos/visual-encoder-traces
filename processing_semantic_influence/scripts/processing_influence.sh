EMBEDDINGS_DIR=$1
DEVICE=$2
OUTPUT_DIR=$3
PROCESSING=$4
K=$5
MODEL=$6
VARIANT=$7

# every combination of processing types for test, positive and negative sets
# generates results for baseline, pos-same, neg-same, all-diff
for test_processing in $PROCESSING
do
    for positive_processing in $PROCESSING
    do
        for negative_processing in $PROCESSING
        do
            CUDA_VISIBLE_DEVICES=$DEVICE python knn_classifier.py \
                --model $MODEL \
                --variant $VARIANT \
                --test_processing_type $test_processing \
                --positives_processing_type $positive_processing \
                --negatives_processing_type $negative_processing \
                --embeddings_dir $EMBEDDINGS_DIR \
                --nb_knn $K \
                --output_name "$OUTPUT_DIR/model=${MODEL}_variant=${VARIANT}/${PROCESSING}/${test_processing}_${positive_processing}_${negative_processing}.txt"
        done
    done
done

# generation of results for uniform
for test_processing in $PROCESSING
do
    for seed in {0..9}
    do
        CUDA_VISIBLE_DEVICES=$DEVICE python knn_classifier.py \
            --model $MODEL \
            --variant $VARIANT \
            --test_processing_type $test_processing \
            --positives_processing_type ${PROCESSING} \
            --negatives_processing_type ${PROCESSING} \
            --embeddings_dir $EMBEDDINGS_DIR \
            --nb_knn $K \
            --seed $seed \
            --output_name "$OUTPUT_DIR/model=${MODEL}_variant=${VARIANT}/${PROCESSING}/${test_processing}_uniform_seed=${seed}.txt"
    done
done

CUDA_VISIBLE_DEVICES=$DEVICE python generate_results.py --model $MODEL --variant $VARIANT --processing_type $PROCESSING --output_dir $OUTPUT_DIR