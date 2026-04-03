import argparse
from collections import defaultdict
import json
import os


def main(args):
    report = defaultdict(list)

    processing_str = " ".join(args.processing_type)

    for test_processing in args.processing_type:
        for other_processing in args.processing_type:
            if other_processing == test_processing:
                with open(f"{args.output_dir}/model={args.model}_variant={args.variant}/{processing_str}/{test_processing}_{other_processing}_{other_processing}.txt", "r") as f:
                    acc = float(f.read().strip())
                    report["baseline"].append(acc)
            else:
                with open(f"{args.output_dir}/model={args.model}_variant={args.variant}/{processing_str}/{test_processing}_{other_processing}_{other_processing}.txt", "r") as f:
                    acc = float(f.read().strip())
                    report[f"all_diff"].append(acc)
                with open(f"{args.output_dir}/model={args.model}_variant={args.variant}/{processing_str}/{test_processing}_{test_processing}_{other_processing}.txt", "r") as f:
                    acc = float(f.read().strip())
                    report[f"pos_same"].append(acc)
                with open(f"{args.output_dir}/model={args.model}_variant={args.variant}/{processing_str}/{test_processing}_{other_processing}_{test_processing}.txt", "r") as f:
                    acc = float(f.read().strip())
                    report[f"neg_same"].append(acc)

    for test_processing in args.processing_type:
        for seed in range(10):
            with open(f"{args.output_dir}/model={args.model}_variant={args.variant}/{processing_str}/{test_processing}_uniform_seed={seed}.txt", "r") as f:
                acc = float(f.read().strip())
                report[f"uniform"].append(acc)

    output = {
        "detailed": report,
        "baseline": sum(report["baseline"]) / len(report["baseline"]),
        "all_diff": sum(report["all_diff"]) / len(report["all_diff"]),
        "pos_same": sum(report["pos_same"]) / len(report["pos_same"]),
        "neg_same": sum(report["neg_same"]) / len(report["neg_same"]),
        "uniform": sum(report["uniform"]) / len(report["uniform"]),
    }

    print(json.dumps(output, indent=4))

    os.makedirs(f"{args.output_dir}/model={args.model}_variant={args.variant}/{processing_str}", exist_ok=True)

    with open(f"{args.output_dir}/model={args.model}_variant={args.variant}/{processing_str}/summary.json", "w") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(
        prog, max_help_position=80
    )
    parser = argparse.ArgumentParser(
        description="Code for generating results for processing influence.",
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
        "--processing_type",
        nargs='+',
        help="Type of processing to generate results for."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./processing_influence",
        help="Directory to save the output files",
    )
    args = parser.parse_args()
    main(args)