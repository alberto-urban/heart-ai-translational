import argparse
import json
import logging
from pathlib import Path

import joblib

from heart_ai.visualization import plot_calibration_curve, plot_feature_importance, plot_roc_curve

from heart_ai.config import ALL_FEATURES, DATA_PATH
from heart_ai.models import MODEL_NAMES
from heart_ai.pipeline import train_and_evaluate
from heart_ai.data import load_dataset
from heart_ai.pipeline import cross_validate_model


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    """Configure application-wide logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a heart disease classification model."
    )

    parser.add_argument(
        "--model",
        default="logistic_regression",
        choices=MODEL_NAMES,
        help="Model to train.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all available models.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model_name: str, dataset_path: Path) -> None:
    logger.info("Starting training for model: %s", model_name)

    dataset = load_dataset(dataset_path)

    pipeline, result = train_and_evaluate(
        df=dataset,
        model_name=model_name,
        features=ALL_FEATURES,
    )

    cv_results = cross_validate_model(
        df=dataset,
        model_name=model_name,
        features=ALL_FEATURES,
        n_splits=5,
    )

    print("\n=== 5-Fold Cross-Validation ===")

    for metric_name, values in cv_results.items():
        print(
            f"{metric_name:20s}: "
            f"{values['mean']:.4f} ± {values['std']:.4f}"
        )

    logger.info("Training completed for model: %s", model_name)

    # -----------------------------------------------------------------------
    # Save trained pipeline
    # -----------------------------------------------------------------------

    model_path = MODELS_DIR / f"{model_name}.joblib"
    joblib.dump(pipeline, model_path)

    logger.info("Model saved to: %s", model_path)

    # -----------------------------------------------------------------------
    # Save evaluation report
    # -----------------------------------------------------------------------

    report = {
        "model": result.model_name,
        "features": result.features,
        "metrics": result.metrics,
        "cross_validation": cv_results,
        "feature_importance": result.feature_importance,
    }

    report_path = REPORTS_DIR / f"{model_name}.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    logger.info("Report saved to: %s", report_path)

    # -----------------------------------------------------------------------
    # Plot curves and Print concise results to terminal
    # -----------------------------------------------------------------------

    plot_roc_curve(
        roc_curve_data=result.metrics["roc_curve"],
        output_path=REPORTS_DIR / f"{model_name}_roc.png",
        title=f"ROC Curve — {model_name}",
    )

    plot_calibration_curve(
        calibration_data=result.metrics["calibration_curve"],
        output_path=REPORTS_DIR / f"{model_name}_calibration.png",
        title=f"Calibration Curve — {model_name}",
    )

    plot_feature_importance(
        feature_importance=result.feature_importance,
        output_path=REPORTS_DIR / f"{model_name}_feature_importance.png",
        title=f"Feature Importance — {model_name}",
    )

    print(f"\n=== {result.model_name} ===")

    for metric_name, metric_value in result.metrics.items():
        if isinstance(metric_value, float):
            print(f"{metric_name:20s}: {metric_value:.4f}")
        else:
            print(f"{metric_name:20s}: {metric_value}")

    print("\nGenerated artifacts:")
    print(f"  Model: {model_path}")
    print(f"  Report: {report_path}")
    print(
        f"  ROC curve: "
        f"{REPORTS_DIR / f'{model_name}_roc.png'}"
    )
    print(
        f"  Calibration curve: "
        f"{REPORTS_DIR / f'{model_name}_calibration.png'}"
    )
    print(
        f"  Feature importance: "
        f"{REPORTS_DIR / f'{model_name}_feature_importance.png'}"
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the training workflow."""
    configure_logging()

    args = parse_arguments()

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    if args.all:
        models_to_train = MODEL_NAMES
    else:
        models_to_train = [args.model]

    logger.info("Models selected: %s", models_to_train)

    for model_name in models_to_train:
        train_model(
            model_name=model_name,
            dataset_path=DATA_PATH,
        )


if __name__ == "__main__":
    main()

