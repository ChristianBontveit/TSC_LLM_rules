import math
import numpy as np
import os
from typing import Optional, Tuple

def generate_time_series(num_samples: int, num_timesteps: int = 60, class0_mean: float = 5.0, class1_mean: float = 10.0, class_std: float = 3.0, end_value: float = 0.0, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Generate 2-class time series where each sample is a straight line from a
    sampled start value to a shared end value. Labels are 0 and 1.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    num_class0 = num_samples // 2
    num_class1 = num_samples - num_class0
    steps = np.linspace(0.0, 1.0, num_timesteps, dtype=float)

    data = []
    for label, mean, count in (
        (0, class0_mean, num_class0),
        (1, class1_mean, num_class1),
    ):
        starts = rng.normal(loc=mean, scale=class_std, size=count)
        series = starts[:, None] + (end_value - starts)[:, None] * steps
        labels = np.full((count, 1), label, dtype=float)
        data.append(np.hstack((labels, series)))

    dataset = np.vstack(data)
    dataset = dataset[rng.permutation(dataset.shape[0])]
    return dataset


def normalize_data(dataset: np.ndarray, min_value: Optional[float] = None, max_value: Optional[float] = None) -> Tuple[np.ndarray, float, float]:
    labels = dataset[:,0]
    dataset = dataset[:,1:]
    if min_value is None:
        min_value = np.min(dataset)
    if max_value is None:
        max_value = np.max(dataset)

    dataset = (dataset - min_value) / (max_value - min_value + 1e-8)

    dataset = np.hstack((labels.reshape(-1, 1), dataset))
    assert not np.isnan(dataset).any(), "NaN values in the dataset."
    return dataset, min_value, max_value


if __name__ == "__main__":
    output_dir = "data/Synthetic"
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(42)

    # Train
    train_dataset = generate_time_series(num_samples=200, rng=rng)
    np.save(os.path.join(output_dir, "Synthetic_TRAIN.npy"), train_dataset)
    train_norm, min_value, max_value = normalize_data(train_dataset)
    np.save(os.path.join(output_dir, "Synthetic_TRAIN_normalized.npy"), train_norm)

    # Validation
    validation_dataset = generate_time_series(num_samples=100, rng=rng)
    np.save(os.path.join(output_dir, "Synthetic_VALIDATION.npy"), validation_dataset)
    validation_norm, _, _ = normalize_data(validation_dataset, min_value=min_value, max_value=max_value)
    np.save(os.path.join(output_dir, "Synthetic_VALIDATION_normalized.npy"), validation_norm)

    # Test
    test_dataset = generate_time_series(num_samples=100, rng=rng)
    np.save(os.path.join(output_dir, "Synthetic_TEST.npy"), test_dataset)
    test_norm, _, _ = normalize_data(test_dataset, min_value=min_value, max_value=max_value)
    np.save(os.path.join(output_dir, "Synthetic_TEST_normalized.npy"), test_norm)

