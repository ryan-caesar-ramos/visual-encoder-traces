# trex
# Copyright (C) 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

import argparse
import json
import os
import random
import sys

import numpy as np
import optuna
import torch
import utils
from logreg_trainer import LogregSklearnTrainer, LogregTorchTrainer
from safetensors import safe_open


def _prepare_features(args):
    data_dict = {}

    with open(
        os.path.join(
            args.split_data_json_dir,
            f"split_data_target_attribute={args.target_attribute}.json",
        ),
        "r",
    ) as f:
        split_data = json.load(f)

    assert split_data["target_attribute"] == args.target_attribute, (
        "accessed json file may not be correct"
    )

    embeddings_path = os.path.join(
        args.embeddings_dir, f"model={args.model}_variant={args.variant}.safetensors"
    )

    with safe_open(embeddings_path, framework="pt", device="cpu") as f:
        embeddings = f.get_tensor("image_embeddings")
        flickr_ids = f.get_tensor("Flickr ID").tolist()

    flickr_id_to_idx = {flickr_id: i for i, flickr_id in enumerate(flickr_ids)}

    for split in ("train", "val", "test"):
        if f"{split}_ids" not in split_data:
            continue
        split_flickr_ids = split_data[f"{split}_ids"]
        split_idxs = [flickr_id_to_idx[flickr_id] for flickr_id in split_flickr_ids]
        X = embeddings[split_idxs]
        Y = torch.tensor(
            split_data[f"encoded_{split}_labels"],
            dtype=torch.long,
        )
        data_dict[split] = [X, Y]

    data_dict["trainval"] = [
        torch.cat((data_dict["train"][0], data_dict["val"][0])),
        torch.cat((data_dict["train"][1], data_dict["val"][1])),
    ]

    return data_dict


def main(args):
    if os.path.exists(os.path.join(args.output_dir, "predictions.pkl")):
        print("predictions.pkl already exists")
        sys.exit()
    data_dict = _prepare_features(args)

    trainer_class = (
        LogregSklearnTrainer
        if args.clf_type == "logreg_sklearn"
        else LogregTorchTrainer
    )

    # tune hyper-parameters with optuna
    print("==> Starting hyper-parameter tuning")
    clf_trainer = trainer_class(
        data_dict["train"][0],
        data_dict["train"][1],
        data_dict["val"][0],
        data_dict["val"][1],
        args,
    )
    hps_sampler = optuna.samplers.TPESampler(
        multivariate=args.clf_type == "logreg_torch",
        group=args.clf_type == "logreg_torch",
        seed=args.seed,
    )
    study = optuna.create_study(sampler=hps_sampler, direction="maximize")
    study.optimize(
        clf_trainer,
        n_trials=args.n_optuna_trials,
        n_jobs=args.n_optuna_workers,
        show_progress_bar=False,
    )
    utils.save_pickle(study, os.path.join(args.output_dir, "study.pkl"))
    fig = optuna.visualization.plot_contour(study, params=clf_trainer.hps_list)
    fig.write_html(os.path.join(args.output_dir, "study_contour.html"))

    print("*" * 50)
    print("Hyper-parameter search ended")
    print("best_trial:")
    print(str(study.best_trial))
    print("best_params:")
    print(str(study.best_params))
    print("*" * 50, flush=True)

    # train the final classifier with the tuned hyper-parameters
    print("==> Training the final classifier")
    del clf_trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    clf_trainer = trainer_class(
        data_dict["trainval"][0],
        data_dict["trainval"][1],
        data_dict["test"][0],
        data_dict["test"][1],
        args,
    )
    clf_trainer.set_hps(study.best_params)
    clf_trainer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        "--target_attribute",
        type=str,
        default="",
        help="Target attribute to classify",
    )
    parser.add_argument(
        "--split_data_json_dir",
        type=str,
        default="",
        help="Directory containing json files of split details",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default="",
        help="Path to embedding safetensors",
    )
    parser.add_argument(
        "--features_norm",
        type=str,
        default="none",
        choices=["standard", "l2", "none"],
        help="Normalization applied to features before the classifier",
    )
    parser.add_argument(
        "--clf_type",
        type=str,
        default="logreg_sklearn",
        choices=["logreg_sklearn", "logreg_torch"],
        help="Type of linear classifier to train on top of features",
    )
    parser.add_argument(
        "--dataset_per_val",
        type=float,
        default=0.2,
        help="Percentage of the val set, sampled from the trainval set for hyper-parameter tuning",
    )
    # For the L-BFGS-based logistic regression trainer implemented in scikit-learn
    parser.add_argument(
        "--clf_C",
        type=float,
        help="""Inverse regularization strength for sklearn.linear_model.LogisticRegression.
        Note that this variable is determined by Optuna, so do not set it manually""",
    )
    parser.add_argument(
        "--clf_C_min",
        type=float,
        default=1e-5,
        help="Power of the minimum C parameter sampled by Optuna",
    )
    parser.add_argument(
        "--clf_C_max",
        type=float,
        default=1e6,
        help="Power of the maximum C parameter sampled by Optuna",
    )
    parser.add_argument(
        "--clf_max_iter",
        type=int,
        default=2000,
        help="Maximum number of iterations to run the classifier for sklearn.linear_model.LogisticRegression during the hyper-parameter tuning stage.",
    )
    # For the SGD-based logistic regression trainer implemented in PyTorch
    parser.add_argument(
        "--clf_lr",
        type=float,
        help="""Learning rate.
        Note that this variable is determined by Optuna, so do not set it manually""",
    )
    parser.add_argument(
        "--clf_lr_min",
        type=float,
        default=1e-1,
        help="Power of the minimum lr parameter sampled by Optuna",
    )
    parser.add_argument(
        "--clf_lr_max",
        type=float,
        default=1e2,
        help="Power of the maximum lr parameter sampled by Optuna",
    )
    parser.add_argument(
        "--clf_wd",
        type=float,
        help="""Weight decay.
        Note that this variable is determined by Optuna, do not set it manually""",
    )
    parser.add_argument(
        "--clf_wd_min",
        type=float,
        default=1e-12,
        help="Power of the minimum weight decay parameter sampled by Optuna",
    )
    parser.add_argument(
        "--clf_wd_max",
        type=float,
        default=1e-4,
        help="Power of the maximum weight decay parameter sampled by Optuna",
    )
    parser.add_argument(
        "--clf_mom",
        type=float,
        default=0.9,
        help="SGD momentum. We do not tune this variable.",
    )
    parser.add_argument(
        "--clf_epochs",
        type=int,
        default=100,
        help="""Number of epochs to train the linear classifier.
        We do not tune this variable""",
    )
    parser.add_argument(
        "--clf_batch_size",
        type=int,
        default=1024,
        help="""Batch size for SGD.
        We do not tune this variable""",
    )
    # Common for all trainers
    parser.add_argument(
        "--n_sklearn_workers",
        type=int,
        default=-1,
        help="Number of CPU cores to use in Scikit-learn jobs. -1 means to use all available cores.",
    )
    parser.add_argument(
        "--n_optuna_workers",
        type=int,
        default=1,
        help="Number of concurrent Optuna jobs",
    )
    parser.add_argument(
        "--n_optuna_trials",
        type=int,
        default=30,
        help="Number of trials run by Optuna",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=22,
        help="Seed for the random number generators",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Whether to use CUDA during feature extraction and classifier training",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./linear-classifier-output",
        help="Whether to save the logs",
    )

    args = parser.parse_args()
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        args.device = "cuda"
    else:
        args.device = "cpu"
    utils.print_program_info(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
