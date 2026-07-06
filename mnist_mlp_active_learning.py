# %% MNIST pool-based active learning with an MLP downstream model.
#
# Builds experiments compatible with adult_prototype_fitted_q_learning episode
# runners (same splits, uncertainty acquisition, state features).

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from adult_prototype_fitted_q_learning import (
    batch_size_for_fraction,
    resolve_row_count_config,
)

MNIST_CLASS_COUNT = 10


def load_mnist_arrays():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    features = np.asarray(mnist.data, dtype=np.float32) / 255.0
    targets = np.asarray(mnist.target, dtype=np.int64)
    train_features = features[:60_000]
    train_targets = targets[:60_000]
    test_features = features[60_000:]
    test_targets = targets[60_000:]
    return train_features, train_targets, test_features, test_targets


def select_initial_train_row_ids(train_targets, config):
    labels_per_class = config.get("labels_per_class")
    if labels_per_class is not None:
        labels_per_class = int(labels_per_class)
        if labels_per_class < 1:
            raise ValueError("labels_per_class must be >= 1")
        selected = []
        for class_label in range(MNIST_CLASS_COUNT):
            class_positions = np.flatnonzero(train_targets == class_label)
            if len(class_positions) < labels_per_class:
                raise ValueError(
                    f"Need {labels_per_class} examples for class {class_label}; "
                    f"only {len(class_positions)} available in the subsample"
                )
            selected.extend(class_positions[:labels_per_class].tolist())
        return np.asarray(selected, dtype=np.int64)
    return None


def split_mnist_rows(train_features, train_targets, test_features, test_targets, config):
    config = resolve_row_count_config(config)
    total_rows = int(config["total_rows"])
    if total_rows > len(train_features):
        raise ValueError(
            f"total_rows={total_rows:,} exceeds MNIST train subsample "
            f"({len(train_features):,} rows)"
        )

    all_row_ids = np.arange(len(train_features))
    subsample_row_ids, _ = train_test_split(
        all_row_ids,
        train_size=total_rows,
        stratify=train_targets,
        random_state=config["random_seed"],
    )
    subsample_features = train_features[subsample_row_ids]
    subsample_targets = train_targets[subsample_row_ids]

    preset_initial_row_ids = select_initial_train_row_ids(subsample_targets, config)
    if preset_initial_row_ids is not None:
        initial_train_rows = len(preset_initial_row_ids)
        if initial_train_rows != config["initial_train_rows"]:
            config["initial_train_rows"] = initial_train_rows
        remaining_row_ids = np.setdiff1d(
            np.arange(total_rows, dtype=np.int64),
            preset_initial_row_ids,
            assume_unique=True,
        )
        initial_train_row_ids = preset_initial_row_ids
    else:
        remaining_row_ids = np.arange(total_rows, dtype=np.int64)
        remaining_row_ids, initial_train_row_ids = train_test_split(
            remaining_row_ids,
            train_size=config["initial_train_rows"],
            stratify=subsample_targets[remaining_row_ids],
            random_state=config["random_seed"] + 4,
        )

    remaining_row_ids, hidden_validation_row_ids = train_test_split(
        remaining_row_ids,
        test_size=config["hidden_validation_rows"],
        stratify=subsample_targets[remaining_row_ids],
        random_state=config["random_seed"] + 2,
    )
    remaining_row_ids, evaluation_row_ids = train_test_split(
        remaining_row_ids,
        test_size=config["evaluation_rows"],
        stratify=subsample_targets[remaining_row_ids],
        random_state=config["random_seed"] + 3,
    )
    acquisition_pool_row_ids = remaining_row_ids

    def partition(row_ids):
        return (
            subsample_features[row_ids],
            pd.Series(subsample_targets[row_ids], name="target"),
        )

    initial_train_features, initial_train_targets = partition(initial_train_row_ids)
    acquisition_pool_features, acquisition_pool_targets = partition(
        acquisition_pool_row_ids
    )
    evaluation_features, evaluation_targets = partition(evaluation_row_ids)
    hidden_validation_features, hidden_validation_targets = partition(
        hidden_validation_row_ids
    )

    if config.get("use_full_mnist_test", True):
        test_partition_features = test_features
        test_partition_targets = pd.Series(test_targets, name="target")
    else:
        test_partition_features, test_partition_targets = partition(
            train_test_split(
                np.arange(total_rows),
                train_size=config["test_rows"],
                stratify=subsample_targets,
                random_state=config["random_seed"] + 1,
            )[1]
        )

    config["initial_train_rows"] = len(initial_train_features)
    config["acquisition_pool_rows"] = len(acquisition_pool_features)
    config["evaluation_rows"] = len(evaluation_features)
    config["hidden_validation_rows"] = len(hidden_validation_features)
    if not config.get("use_full_mnist_test", True):
        config["test_rows"] = len(test_partition_features)

    print(
        f"MNIST subsample: {total_rows:,} train rows; "
        f"test rows: {len(test_partition_features):,}; "
        f"initial labels: {len(initial_train_features):,}"
    )

    return {
        "initial_train_features": initial_train_features,
        "initial_train_targets": initial_train_targets,
        "acquisition_pool_features": acquisition_pool_features,
        "acquisition_pool_targets": acquisition_pool_targets,
        "evaluation_features": evaluation_features,
        "evaluation_targets": evaluation_targets,
        "hidden_validation_features": hidden_validation_features,
        "hidden_validation_targets": hidden_validation_targets,
        "test_features": test_partition_features,
        "test_targets": test_partition_targets,
        "config": config,
    }


def make_mlp_model(config):
    model_config = dict(config.get("downstream_model", {}))
    hidden_layer_sizes = tuple(
        int(value) for value in model_config.get("hidden_layer_sizes", [256, 128])
    )
    return Pipeline(
        [
            ("standard_scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    max_iter=int(model_config.get("max_iter", 200)),
                    alpha=float(model_config.get("alpha", 1e-4)),
                    early_stopping=bool(model_config.get("early_stopping", True)),
                    validation_fraction=float(
                        model_config.get("validation_fraction", 0.1)
                    ),
                    learning_rate_init=float(
                        model_config.get("learning_rate_init", 1e-3)
                    ),
                    random_state=int(
                        model_config.get("random_state", config["random_seed"])
                    ),
                ),
            ),
        ]
    )


def build_mnist_experiment(config):
    config = dict(config)
    train_features, train_targets, test_features, test_targets = load_mnist_arrays()
    partitions = split_mnist_rows(
        train_features, train_targets, test_features, test_targets, config
    )
    config = partitions.pop("config")

    scaler_fit_features = np.vstack(
        [
            partitions["initial_train_features"],
            partitions["acquisition_pool_features"],
        ]
    )
    scaler = StandardScaler()
    scaler.fit(scaler_fit_features)

    def encode_features(features):
        return np.asarray(scaler.transform(features), dtype=np.float32)

    experiment = {"config": config, "make_model": lambda: make_mlp_model(config)}
    for key, value in partitions.items():
        if key.endswith("_features"):
            experiment[key] = encode_features(value)
        else:
            experiment[key] = value
    return experiment


def default_mnist_deploy_config(random_seed=42):
    return {
        "random_seed": random_seed,
        "total_rows": 10_000,
        "use_full_mnist_test": True,
        "labels_per_class": 20,
        "initial_train_fraction": 0.02,
        "acquisition_pool_fraction": 0.84,
        "evaluation_fraction": 0.05,
        "hidden_validation_fraction": 0.05,
        "test_fraction": 0.04,
        "batch_fraction": 0.01,
        "acquisition_budget": 8,
        "retrain_budget": 4,
        "evaluation_budget": 6,
        "acquisition_percentile_low": 0,
        "acquisition_percentile_high": 95,
        "final_retrain_pending_rows": True,
        "allow_stop_action": True,
        "step_penalty": 0.0,
        "target_metric": "accuracy",
        "policy_metric": "accuracy",
        "introspection_metric": "accuracy",
        "introspection_eval_set": "hidden_validation",
        "reporting_metrics": ["accuracy"],
        "downstream_model": {
            "hidden_layer_sizes": [256, 128],
            "max_iter": 200,
            "alpha": 1e-4,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "random_state": random_seed,
        },
    }


def merge_q_policy_rollout_config(mnist_config, training_config):
    rollout_budgets = training_config.get(
        "default_rollout_budgets",
        {
            "acquisition_budget": training_config["acquisition_budget"],
            "retrain_budget": training_config["retrain_budget"],
            "evaluation_budget": training_config["evaluation_budget"],
        },
    )
    batch_fraction = training_config.get(
        "default_rollout_batch_fraction",
        training_config.get("batch_fraction", mnist_config["batch_fraction"]),
    )
    merged = {
        **mnist_config,
        **rollout_budgets,
        "batch_fraction": batch_fraction,
        "batch_size": batch_size_for_fraction(
            batch_fraction, int(mnist_config["total_rows"])
        ),
        "evaluation_return_windows": training_config.get(
            "evaluation_return_windows",
            [training_config.get("evaluation_return_window", 5)],
        ),
        "training_budget_maxima": training_config.get(
            "training_budget_maxima",
            {
                "acquisition_budget": training_config["training_budget_ranges"][
                    "acquisition_budget"
                ][1],
                "retrain_budget": training_config["training_budget_ranges"][
                    "retrain_budget"
                ][1],
                "evaluation_budget": training_config["training_budget_ranges"][
                    "evaluation_budget"
                ][1],
            },
        ),
        "training_batch_fraction_maxima": training_config.get(
            "training_batch_fraction_maxima",
            max(training_config.get("training_batch_fraction_range", [batch_fraction])),
        ),
        "discount_factor": training_config.get("discount_factor", 0.95),
        "lagrangian_q_learning": training_config.get("lagrangian_q_learning", {}),
    }
    return merged
