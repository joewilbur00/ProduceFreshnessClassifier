import argparse
import json

from .model import FreshnessClassifier


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run freshness inference on a single image."
    )
    parser.add_argument(
        "--image-path",
        required=True,
        help="Path to the image to classify.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold for Rotten class.",
    )

    args = parser.parse_args()

    classifier = FreshnessClassifier(threshold=args.threshold)
    result = classifier.predict_image(args.image_path)

    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
