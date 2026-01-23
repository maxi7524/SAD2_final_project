import argparse
from sad2_final_project.bnfinder.bnfinder import manager_bnfinder


def parse_args():
    parser = argparse.ArgumentParser(description="Task 3 – BNFinder pipeline")
    parser.add_argument("--data", required=True)
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument("--metrics-csv", default="results/metrics.csv")
    parser.add_argument("--model-dir", default="results/models")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    manager_bnfinder(
        data=args.data,
        ground_truth=args.ground_truth,
        output_template="results/{dataset}_{score}.sif",
        metrics_csv=args.metrics_csv,
        model_dir=args.model_dir,
    )
