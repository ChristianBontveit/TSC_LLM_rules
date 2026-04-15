from sklearn_extra.cluster import KMedoids
from Utils.load_data import load_dataset, load_dataset_labels
import numpy as np
from tslearn.metrics import dtw


def select_prototypes(dataset_name: str, num_instances: int, data_type: str="TEST_normalized", metric=dtw, return_metadata: bool=False):
    # 1. Load dataset and labels
    X_test = load_dataset(dataset_name=dataset_name, data_type=data_type)
    labels_test = load_dataset_labels(dataset_name=dataset_name, data_type=data_type)
    labels = np.array(labels_test)

    unique_labels = np.unique(labels)
    prototypes_list = []
    support_examples = []

    # 2. For each label, fit KMedoids and collect prototypes
    for label in unique_labels:
        mask = labels == label
        dataset_indices = np.where(mask)[0]
        X_label = X_test[mask]
        if len(X_label) < num_instances:
            continue  # Skip if not enough samples for clustering
        km = KMedoids(n_clusters=num_instances, metric=metric, init="random", random_state=42)  # type: ignore
        km.fit(X_label)
        medoid_indices = km.medoid_indices_
        prototypes = X_label[medoid_indices]
        prototypes_list.append(prototypes)
        support_examples.append({
            "class_label": int(label),
            "selection_type": "prototype",
            "indices": dataset_indices[medoid_indices].astype(int).tolist(),
        })

    # 3. Concatenate all prototypes
    if prototypes_list:
        prototypes = np.concatenate(prototypes_list)
    else:
        prototypes = np.array([])  # Return empty array if no prototypes found

    if return_metadata:
        return prototypes, support_examples

    return prototypes

if __name__ == "__main__":
    select_prototypes("ItalyPowerDemand", num_instances=3)
