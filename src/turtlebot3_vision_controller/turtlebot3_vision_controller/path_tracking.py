"""Continuous lookahead on a closed path, independent of ROS."""

import numpy as np


class ClosedPath:
    """Project onto nearby segments and advance a target by arc length."""

    def __init__(self, points):
        points = np.asarray(points, dtype=float)
        if np.allclose(points[0], points[-1]):
            points = points[:-1]
        vectors = np.roll(points, -1, axis=0) - points
        lengths = np.linalg.norm(vectors, axis=1)
        if len(points) < 3 or np.any(lengths < 1e-6):
            raise ValueError('A closed path needs at least three distinct vertices')
        self.points = points
        self.vectors = vectors
        self.lengths = lengths
        self.starts = np.concatenate(([0.0], np.cumsum(lengths)))
        self.length = self.starts[-1]
        self.progress = None

    def target(self, position, lookahead):
        """Return a moving target without jumping to a neighboring lane."""
        position = np.asarray(position, dtype=float)
        fractions = np.clip(np.sum((position - self.points) * self.vectors,
                                   axis=1) / self.lengths**2, 0.0, 1.0)
        projections = self.points + fractions[:, None] * self.vectors
        distances = np.linalg.norm(projections - position, axis=1)
        progress = self.starts[:-1] + fractions * self.lengths
        if self.progress is not None:
            # Unwrap around the last position to make crossing the lap seam
            # continuous. Restrict search to the local piece of the route.
            progress += self.length * np.round(
                (self.progress - progress) / self.length)
            nearby = ((progress >= self.progress - 0.25)
                      & (progress <= self.progress + 0.75))
            distances = np.where(nearby, distances, np.inf)
        nearest = int(np.argmin(distances))
        projected = float(progress[nearest])
        self.progress = (projected if self.progress is None
                         else max(self.progress, projected))
        target_s = (self.progress + lookahead) % self.length
        index = min(len(self.points) - 1,
                    int(np.searchsorted(self.starts, target_s, side='right') - 1))
        fraction = (target_s - self.starts[index]) / self.lengths[index]
        return self.points[index] + fraction * self.vectors[index]


def slew(previous, desired, rate, delta_time):
    """Limit a command's change per second, independent of camera FPS."""
    step = rate * max(0.0, delta_time)
    return float(np.clip(desired, previous - step, previous + step))
