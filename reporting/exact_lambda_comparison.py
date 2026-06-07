from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def plot_exact_lambda_route_comparison(
    *,
    panels: Iterable[dict],
    output_path: Path,
    title: str,
) -> Path:
    """
    Combine already-generated exact route maps into one comparison figure.

    Each panel dictionary must contain:
      - lambda_value
      - image_path
      - cost
      - emission
      - status

    This is reporting-only. It does not alter any solution or objective.
    """
    panel_list = list(panels)

    if not panel_list:
        raise ValueError(
            "At least one lambda panel is required."
        )

    figure_width = max(
        6.2 * len(panel_list),
        10.0,
    )

    figure, axes = plt.subplots(
        1,
        len(panel_list),
        figsize=(figure_width, 6.8),
        squeeze=False,
    )

    for axis, panel in zip(
        axes[0],
        panel_list,
    ):
        image_path = Path(
            panel["image_path"]
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing route-map panel: {image_path}"
            )

        image = mpimg.imread(
            image_path
        )

        axis.imshow(image)
        axis.axis("off")

        lambda_value = float(
            panel["lambda_value"]
        )
        cost = float(panel["cost"])
        emission = float(
            panel["emission"]
        )
        status = str(
            panel.get(
                "status",
                "",
            )
        )

        axis.set_title(
            (
                f"λ = {lambda_value:g}\n"
                f"Cost = {cost:.4f} | "
                f"Emission = {emission:.4f}\n"
                f"{status}"
            ),
            fontsize=11,
            pad=10,
        )

    figure.suptitle(
        title,
        fontsize=16,
        y=0.995,
    )

    figure.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.965)
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path
