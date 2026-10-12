"""Fixed unmatched references derived without CPR evaluation labels."""
from __future__ import annotations

import math


def basic_unmatched_bound(image_mean_norm, text_mean_norm, sim_image_min, sim_text_min, harris_lambda):
    """A lower bound of the unchanged official BASIC polynomial score.

    Unit encoder vectors have centered norms <= 1+||official mean||.
    Orthogonal projection and convex query expansion cannot increase this
    bound. Official normalization and clamping bound x/y to [0,X]/[0,Y].
    xy-lambda(x+y)^2 is separately concave, so its minimum on this rectangle
    occurs at a corner. All constants come from the released BASIC resources.
    """
    if (not all(math.isfinite(x) for x in (image_mean_norm, text_mean_norm, sim_image_min,
                                          sim_text_min, harris_lambda))
            or image_mean_norm < 0 or text_mean_norm < 0 or harris_lambda < 0
            or abs(sim_image_min) <= 1e-12):
        raise ValueError("Invalid official BASIC score-bound constants")
    image_bound, text_bound = 1 + image_mean_norm, 1 + text_mean_norm
    x = max(0.0, (image_bound ** 2 - sim_image_min) / abs(sim_image_min))
    y = max(0.0, (text_bound * image_bound - sim_text_min) / abs(sim_image_min))
    # A small fixed arithmetic guard covers float32 projection/normalization.
    bound = min(0.0, -harris_lambda * x*x, -harris_lambda * y*y,
                x*y - harris_lambda * (x+y)**2)
    return float(bound - max(1.0, abs(bound)) * 1e-5)
