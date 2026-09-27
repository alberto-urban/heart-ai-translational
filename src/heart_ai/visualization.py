from pathlib import Path

import matplotlib.pyplot as plt


def plot_roc_curve(
    roc_curve_data: dict[str, list[float]],
    output_path: Path,
    title: str = "ROC Curve",
) -> None:
    """Create and save a ROC curve plot."""

    false_positive_rate = roc_curve_data["fpr"]
    true_positive_rate = roc_curve_data["tpr"]

    figure, axis = plt.subplots(figsize=(7, 6))

    axis.plot(
        false_positive_rate,
        true_positive_rate,
        label="Model",
    )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title(title)
    axis.legend()
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_calibration_curve(
    calibration_data: dict[str, list[float]],
    output_path: Path,
    title: str = "Calibration Curve",
) -> None:
    """Create and save a calibration curve plot."""

    predicted_probability = calibration_data["probabilities"]
    observed_frequency = calibration_data["observed_frequencies"]

    figure, axis = plt.subplots(figsize=(7, 6))

    axis.plot(
        predicted_probability,
        observed_frequency,
        marker="o",
        label="Model",
    )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed frequency")
    axis.set_title(title)
    axis.legend()
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_feature_importance(
    feature_importance: dict[str, float],
    output_path: Path,
    title: str = "Feature Importance",
    top_n: int = 15,
) -> None:
    """Create and save a feature-importance plot."""

    if not feature_importance:
        return

    sorted_features = sorted(
        feature_importance.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:top_n]

    feature_names = [item[0] for item in sorted_features]
    importance_values = [item[1] for item in sorted_features]

    figure, axis = plt.subplots(figsize=(9, 6))

    axis.barh(
        feature_names[::-1],
        importance_values[::-1],
    )

    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")
    axis.set_title(title)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)