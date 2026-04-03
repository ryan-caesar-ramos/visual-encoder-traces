import logging
import argparse

import torch
import numpy as np
import os

from tqdm import tqdm

from safetensors import safe_open

log = logging.getLogger(__name__)


@torch.no_grad()
def knn_classifier(
    positives_features,
    negatives_features,
    train_labels,
    test_features,
    test_labels,
    k,
    num_chunks=100,
    num_classes=1000,
):
    top1, total = 0.0, 0
    num_test_images = test_labels.shape[0]
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(negatives_features.device)

    for idx in tqdm(range(0, num_test_images, imgs_per_chunk)):
        # get the features for test images
        features = test_features[idx : min((idx + imgs_per_chunk), num_test_images), :]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = features @ negatives_features.T
        if positives_features is not negatives_features:
            pos_sim = features @ positives_features.T
            for i in range(pos_sim.shape[0]):
                similarity[i, torch.where(targets[i] == train_labels)[0]] = pos_sim[i, torch.where(targets[i] == train_labels)[0]]
        _, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        probs = torch.sum(retrieval_one_hot.view(batch_size, -1, num_classes), 1)
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    return top1


def _normalize_embeddings(embeddings):
    normalized_embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return normalized_embeddings


def _load_feature_file(embeddings_path):
    with safe_open(embeddings_path, framework="pt", device="cpu") as f:
        embeddings = _normalize_embeddings(f.get_tensor("image_embeddings").float())
        labels = f.get_tensor("label")

    return embeddings, labels


def main(args):
    if os.path.exists(args.output_name):
        print(f"Output file {args.output_name} already exists, skipping...")
        return
    
    test_features, test_semantic_labels = _load_feature_file(f"{args.embeddings_dir}/{args.test_processing_type}/model={args.model}_variant={args.variant}_split=val.safetensors")
    test_semantic_labels = test_semantic_labels.long()

    if len(args.positives_processing_type) == 1:
        positives_features, positives_semantic_labels = _load_feature_file(f"{args.embeddings_dir}/{args.positives_processing_type[0]}/model={args.model}_variant={args.variant}_split=train.safetensors")
        positives_semantic_labels = positives_semantic_labels.long()
    else:
        rng = np.random.default_rng(args.seed)
        tmp_features, positives_semantic_labels = _load_feature_file(f"{args.embeddings_dir}/{args.positives_processing_type[0]}/model={args.model}_variant={args.variant}_split=train.safetensors")
        positives_semantic_labels = positives_semantic_labels.long()
        positives_features = torch.zeros_like(tmp_features)
        idxs = rng.choice(len(args.positives_processing_type), positives_features.shape[0], replace=True)
        for i in range(len(args.positives_processing_type)):
            if i != 0:
                tmp_features, _ = _load_feature_file(f"{args.embeddings_dir}/{args.positives_processing_type[i]}/model={args.model}_variant={args.variant}_split=train.safetensors")
            mask = idxs == i
            positives_features[mask] = tmp_features[mask]

    if len(args.positives_processing_type) > 1:
        assert len(args.positives_processing_type) == len(args.negatives_processing_type)
        assert all(x==y for x, y in zip(args.positives_processing_type, args.negatives_processing_type))
        negatives_features = positives_features
    else:
        if args.positives_processing_type[0] != args.negatives_processing_type[0]:
            negatives_features, _ = _load_feature_file(f"{args.embeddings_dir}/{args.negatives_processing_type[0]}/model={args.model}_variant={args.variant}_split=train.safetensors")
        else:
            negatives_features = positives_features

    print("test feature shape:", test_features.shape)
    print("positives feature shape:", positives_features.shape)
    print("negatives feature shape:", negatives_features.shape)

    test_features = test_features.cuda()
    test_semantic_labels = test_semantic_labels.cuda()
    if positives_features is negatives_features:
        positives_features = positives_features.cuda()
        negatives_features = positives_features
    else:
        positives_features = positives_features.cuda()
        negatives_features = negatives_features.cuda()
    positives_semantic_labels = positives_semantic_labels.cuda()

    top1 = knn_classifier(
        positives_features,
        negatives_features,
        positives_semantic_labels,
        test_features,
        test_semantic_labels,
        args.nb_knn,
        args.num_chunks,
        num_classes=positives_semantic_labels.max().item() + 1,
    )
    print(f"{args.nb_knn}-NN classifier result: Top1: {top1}")

    os.makedirs(os.path.dirname(args.output_name), exist_ok=True)
    with open(args.output_name, "w") as f:
        f.write(str(top1))


if __name__ == "__main__":
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(
        prog, max_help_position=80
    )
    parser = argparse.ArgumentParser(
        description="Code for knn classifier evaluation of the processing influence.",
        formatter_class=formatter,
    
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model to use for classification; variant specified separately",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="",
        help="Variant of model to use for classification",
    )
    parser.add_argument(
        "--test_processing_type",
        type=str,
        help="Type of processing to use for the test images",
    )
    parser.add_argument(
        "--positives_processing_type",
        nargs='+',
        help="Type of processing to use for the semantic positive samples"
    )
    parser.add_argument(
        "--negatives_processing_type",
        nargs='+',
        help="Type of processing to use for the semantic negative samples"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for generation of random dataset split"
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default="",
        help="Path to embedding safetensors",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="./processing_influence",
        help="Name of the output file",
    )
    parser.add_argument(
        "--nb_knn",
        type=int,
        default=10,
        help="Number of NN to use.",
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=100,
    )
    args = parser.parse_args()

    main(args)
