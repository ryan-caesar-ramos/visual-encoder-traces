# trex
# Copyright (C) 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

import argparse
import os
import random
import sys

import numpy as np
import optuna
import torch
import utils
from logreg_trainer import LogregSklearnTrainer, LogregTorchTrainer
from safetensors import safe_open
import pickle


def _load_feature_file(embeddings_path):
    with safe_open(embeddings_path, framework="pt", device="cpu") as f:
        embeddings = f.get_tensor("image_embeddings").float()

    return embeddings


def _split_trainval(X, Y, per_val=0.2):
    train_inds, val_inds = [], []

    labels = np.unique(Y)
    for c in labels:
        inds = np.where(Y == c)[0]
        random.shuffle(inds)
        n_val = int(len(inds) * per_val)
        if n_val == 0:
            assert len(inds) >= 2, (
                "We need at least 2 samples for class {}, "
                "to use one of them for validation set."
            )
            n_val = 1
            per_val = n_val / len(inds)
            print(
                "Validation set percentage for class {} is not enough, "
                "number of training images for this class: {}. "
                "Taking one sample for validation set by overriding per_val as {}".format(
                    c, len(inds), per_val
                )
            )
        assert n_val > 0
        train_inds.extend(inds[:-n_val].tolist())
        val_inds.extend(inds[-n_val:].tolist())

    train_inds = np.array(train_inds)
    val_inds = np.array(val_inds)
    assert (
        train_inds.shape[0] + val_inds.shape[0] == X.shape[0]
    ), "Error: Size mismatch for train ({}), val ({}) and trainval ({}) sets".format(
        train_inds.shape[0], val_inds.shape[0], X.shape[0]
    )
    assert (
        len(np.intersect1d(train_inds, val_inds)) == 0
    ), "Error: train and val sets overlap!"

    train = [X[train_inds], Y[train_inds]]
    val = [X[val_inds], Y[val_inds]]

    return [train, val]


def _get_features(args, tmp_seed):
    test_features = None
    test_processing_labels = None
    for i in range(len(args.processing)):
        tmp_features = _load_feature_file(f"{args.embeddings_dir}/{args.processing[i]}/model={args.model}_variant={args.variant}_split=val.safetensors")
        if i != 0:
            test_features = torch.cat((test_features, tmp_features), dim=0)
            test_processing_labels = torch.cat((test_processing_labels, torch.ones(tmp_features.shape[0], dtype=torch.long) * i), dim=0)
        else:
            test_features = tmp_features
            test_processing_labels = torch.ones(tmp_features.shape[0], dtype=torch.long) * i


    rng = np.random.default_rng(tmp_seed)
    tmp_features = _load_feature_file(f"{args.embeddings_dir}/{args.processing[0]}/model={args.model}_variant={args.variant}_split=train.safetensors")
    trainval_features = torch.zeros_like(tmp_features)
    idxs = rng.choice(len(args.processing), tmp_features.shape[0], replace=True)
    trainval_processing_labels = torch.tensor(idxs).long()
    for i in range(len(args.processing)):
        if i != 0:
            tmp_features = _load_feature_file(f"{args.embeddings_dir}/{args.processing[i]}/model={args.model}_variant={args.variant}_split=train.safetensors")
        mask = idxs == i
        trainval_features[mask] = tmp_features[mask]

    return trainval_features, trainval_processing_labels, test_features, test_processing_labels


def _prepare_features(args, tmp_seed=None):
    trainval_features, trainval_labels, test_features, test_labels = _get_features(args, args.seed if tmp_seed is None else tmp_seed)

    data_dict = {
        "test": (test_features, test_labels),
    }

    if args.dataset_per_val > 0:
        data_dict["train"], data_dict["val"] = _split_trainval(
            trainval_features, trainval_labels, args.dataset_per_val
        )
    else:
        data_dict["train"] = [trainval_features, trainval_labels]

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
    print("==> Training the final classifiers")
    accs = []
    for tmp_seed in range(10):
        args.dataset_per_val = 0
        data_dict = _prepare_features(args, tmp_seed)
        del clf_trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        clf_trainer = trainer_class(
            data_dict["train"][0],
            data_dict["train"][1],
            data_dict["test"][0],
            data_dict["test"][1],
            args,
        )
        clf_trainer.set_hps(study.best_params)
        acc1 = clf_trainer()
        accs.append(acc1)

    with open(os.path.join(args.output_dir, "accs_per_seed.pkl"), "wb") as f:
        pickle.dump(accs, f)
    print(f"Mean accuracy: {np.mean(accs):.2f} +/- {np.std(accs):.2f}")


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
        "--processing",
        nargs='+',
        help="Type of processing for evaluation."
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
