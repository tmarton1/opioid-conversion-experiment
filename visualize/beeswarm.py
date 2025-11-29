import numpy as np
from dataclasses import dataclass
from typing import Sequence


@dataclass
class BeeswarmPlacer:
    """
    Compute beeswarm-style x positions for 1D data.

    - Points are placed so they don't overlap in (x, y) space.
    - By default we work in log10(y) so it looks natural on a log y-axis.
    - All x positions are kept within [-x_max, x_max].

    Parameters
    ----------
    x_max : float
        Maximum horizontal extent of the swarm.
    r : float
        “Radius” used to decide overlap in (x, y_scaled) space.
        Smaller r → tighter packing; larger r → more spread.
    use_log_y : bool
        If True, use log10(y) internally for spacing.
    """
    x_max: float = 0.35
    r: float = 0.06
    use_log_y: bool = True

    def _transform_y(self, y: Sequence[float]) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if self.use_log_y:
            # Guard against non-positive values on log scale
            # (you shouldn't have these for coefficients, but just in case)
            y = np.where(y > 0, np.log10(y), -np.inf)
        return y

    def compute(self, y_values: Sequence[float]) -> np.ndarray:
        """
        Compute x positions for a beeswarm layout.

        Parameters
        ----------
        y_values : sequence of float
            The raw y-values for each point (e.g., coefficients).

        Returns
        -------
        np.ndarray
            Array of x positions, same length as y_values.
        """
        y_scaled = self._transform_y(y_values)
        n = len(y_scaled)
        xs = np.zeros(n, dtype=float)

        order = np.argsort(y_scaled)
        placed: list[tuple[float, float]] = []

        # How close in y (in scaled units) counts as "same band"
        y_band = 0.05  # tweak: ~0.05 in log10 ≈ 12% difference in value

        for idx in order:
            y = y_scaled[idx]
            x = 0.0
            k = 0

            while True:
                overlap = False
                for px, py in placed:
                    dy = abs(y - py)
                    dx = abs(x - px)

                    if dy < y_band:
                        # Treat as same horizontal band: enforce separation in x only
                        if dx < 2 * self.r:
                            overlap = True
                            break
                    else:
                        # Use full 2D distance when y is appreciably different
                        if dx * dx + dy * dy < (2 * self.r) ** 2:
                            overlap = True
                            break

                if (not overlap) or (abs(x) >= self.x_max):
                    break

                k += 1
                direction = -1 if (k % 2) else 1
                steps = (k + 1) // 2
                x = direction * steps * self.r

            xs[idx] = x
            placed.append((x, y))

        return xs
