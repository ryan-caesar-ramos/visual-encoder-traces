from safetensors import safe_open
import torch
import argparse
import os
import json


MODEL_MAP = {
    0: "dslr/compact",
    1: "phone",
}


def normalize_embeddings(embeddings):
    normalized_embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return normalized_embeddings


@torch.no_grad()
def calculate_recall(normalized_embeddings, models, subjects, query_class):
    query_embeddings = normalized_embeddings[models == query_class]
    cos_sims = query_embeddings @ normalized_embeddings.T
    cos_sims[torch.arange(query_embeddings.size(0)), torch.where(models == query_class)[0]] = float("-inf")
    target_idxs = torch.where(models != query_class)[0]
    assert torch.all(
            subjects[models == query_class] == subjects[target_idxs]
    )
    cos_sims_negatives_same_class = []
    cos_sims_negatives_diff_class = []

    for i in range(query_embeddings.size(0)):
        cos_sims_negatives_same_class_row = cos_sims[i].clone()
        cos_sims_negatives_same_class_row[torch.where(models != query_class)[0]] = (
            float("-inf")
        )
        cos_sims_negatives_same_class_row[target_idxs[i]] = cos_sims[i, target_idxs[i]]
        cos_sims_negatives_same_class.append(cos_sims_negatives_same_class_row)

        cos_sims_negatives_diff_class_row = cos_sims[i].clone()
        cos_sims_negatives_diff_class_row[torch.where(models == query_class)[0]] = (
            float("-inf")
        )
        cos_sims_negatives_diff_class_row[target_idxs[i]] = cos_sims[i, target_idxs[i]]
        cos_sims_negatives_diff_class.append(cos_sims_negatives_diff_class_row)

    cos_sims_negatives_same_class = torch.stack(cos_sims_negatives_same_class)
    cos_sims_negatives_diff_class = torch.stack(cos_sims_negatives_diff_class)

    top_k_idxs_negatives_same_class = cos_sims_negatives_same_class.topk(
        k=1, dim=-1
    ).indices.flatten()
    score_negatives_same_class = (
        (top_k_idxs_negatives_same_class == target_idxs)
        .float()
        .mean()
        .item()
    )

    top_k_idxs_negatives_diff_class = cos_sims_negatives_diff_class.topk(
        k=1, dim=-1
    ).indices.flatten()
    score_negatives_diff_class = (
        (top_k_idxs_negatives_diff_class == target_idxs)
        .float()
        .mean()
        .item()
    )

    return score_negatives_same_class, score_negatives_diff_class


def main(args):
    with safe_open(f"{args.embeddings_dir}/model={args.model}_variant={args.variant}.safetensors", framework="pt", device="cpu") as f:
        embeddings = f.get_tensor("image_embeddings")
        subjects = f.get_tensor("subject")
        models = f.get_tensor("model")

    embeddings = embeddings.to("cuda")
    models = models.to("cuda")
    subjects = subjects.to("cuda")
    normalized_embeddings = normalize_embeddings(embeddings)

    output = {}

    for query_class in torch.unique(models):
        score_negatives_same_class, score_negatives_diff_class = calculate_recall(normalized_embeddings, models, subjects, query_class)
        print(f"Query class: {MODEL_MAP[query_class.item()]}, Score negatives same class: {score_negatives_same_class}, Score negatives diff class: {score_negatives_diff_class}")
        output[MODEL_MAP[query_class.item()]] = {
            "score_negatives_same_class": score_negatives_same_class,
            "score_negatives_diff_class": score_negatives_diff_class,
        }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code for near duplicate retrieval"
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
        "--embeddings_dir",
        type=str,
        default="",
        help="Path to embedding safetensors",
    )
    parser.add_argument(
        "--output_dir", type=str, help="Directory to save the results to"
    )
    args = parser.parse_args()
    main(args)