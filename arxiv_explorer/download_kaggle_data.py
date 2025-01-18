import kagglehub
import argparse
import subprocess


def main(path: str):
    k_path = kagglehub.dataset_download("Cornell-University/arxiv")

    if path is not None:
        subprocess.run(["mv", k_path, path])

    print(f"Saved kaggle dataset to: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the arxiv dataset from kaggle."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help=f"Directory to download the dataset to. Default is set by kagglehub.",
    )
    args = parser.parse_args()
    main(path=args.output_dir)
