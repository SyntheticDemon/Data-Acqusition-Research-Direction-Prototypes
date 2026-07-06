# Continuous-state fitted-Q prototype for Adult batch acquisition.
# Replaces the tabular q_values[(state, action)] dict with one
# linear regressor per action over a compact state vector.
# LaTeX formulation: docs/formulation/adult_fitted_q_learning.tex

# %% Imports
import numpy as np
import pandas as pd
from joblib import dump, load
from pathlib import Path
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, PolynomialFeatures, StandardScaler
from tqdm.auto import tqdm


Q_ACTIONS = ["Acquire", "Retrain", "Evaluate", "Stop"]

SUPPORTED_METRICS = (
    "f1",
    "roc_auc",
    "accuracy",
    "balanced_accuracy",
    "pr_auc",
)

METRIC_LABELS = {
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced accuracy",
    "pr_auc": "PR-AUC",
}

BASE_STATE_FEATURE_NAMES = [
    "initial_acq_budget_frac_of_max",
    "initial_retrain_budget_frac_of_max",
    "initial_eval_budget_frac_of_max",
    "remaining_acq_budget_frac_of_max",
    "remaining_retrain_budget_frac_of_max",
    "remaining_eval_budget_frac_of_max",
    "pending_rows_frac",
    "batch_fraction_frac_of_max",
    "last_eval_score",
    "last_eval_score_change",
]


def evaluation_return_windows_from_config(config):
    if "evaluation_return_windows" in config:
        windows = [int(window) for window in config["evaluation_return_windows"]]
    else:
        windows = [int(config.get("evaluation_return_window", 1))]
    windows = sorted(set(windows))
    if not windows or any(window < 1 for window in windows):
        raise ValueError(
            "evaluation_return_windows entries must be a non-empty list of "
            "integers >= 1"
        )
    return windows


def rolling_eval_score_change_feature_names(config):
    return [
        f"rolling_eval_score_change_sum_w{window}"
        for window in evaluation_return_windows_from_config(config)
    ]


def state_feature_names(config):
    return BASE_STATE_FEATURE_NAMES + rolling_eval_score_change_feature_names(config)


# Default feature list when no config is available (legacy single window).
STATE_FEATURE_NAMES = state_feature_names({"evaluation_return_window": 5})


def eval_score_change_from_start(latest_evaluation_score, initial_evaluation_score):
    if latest_evaluation_score is None:
        return 0.0
    return float(latest_evaluation_score) - float(initial_evaluation_score)


def compute_rolling_eval_score_change_features(
    evaluation_incremental_change_history,
    latest_evaluation_score,
    initial_evaluation_score,
    windows,
):
    change_from_start = eval_score_change_from_start(
        latest_evaluation_score, initial_evaluation_score
    )
    features = []
    for window in windows:
        if len(evaluation_incremental_change_history) >= window:
            features.append(
                float(sum(evaluation_incremental_change_history[-window:]))
            )
        else:
            features.append(change_from_start)
    return features


Q_POLICY_BUNDLE_VERSION = 13
DEFAULT_Q_POLICY_ARTIFACT_PATH = Path("artifacts/fitted_q_policy.joblib")


# %% Lagrangian Q-learning
def lagrangian_settings(config):
    settings = config.get("lagrangian_q_learning", {})
    return {
        "enabled": bool(settings.get("enabled", False)),
        "lambda_acquisition": float(settings.get("lambda_acquisition", 0.0)),
        "lambda_retrain": float(settings.get("lambda_retrain", 0.0)),
        "lambda_evaluation": float(settings.get("lambda_evaluation", 0.0)),
        "lambda_learning_rate_acquisition": float(
            settings.get("lambda_learning_rate_acquisition", 0.0)
        ),
        "lambda_learning_rate_retrain": float(
            settings.get("lambda_learning_rate_retrain", 0.0)
        ),
        "lambda_learning_rate_evaluation": float(
            settings.get("lambda_learning_rate_evaluation", 0.0)
        ),
    }


def initial_lagrangian_lambdas(config):
    settings = lagrangian_settings(config)
    return {
        "lambda_acquisition": settings["lambda_acquisition"],
        "lambda_retrain": settings["lambda_retrain"],
        "lambda_evaluation": settings["lambda_evaluation"],
    }


def lagrangian_action_penalty(action, lagrangian_lambdas, config):
    settings = lagrangian_settings(config)
    if not settings["enabled"]:
        return 0.0
    if action == "Acquire":
        return lagrangian_lambdas["lambda_acquisition"]
    if action == "Retrain":
        return lagrangian_lambdas["lambda_retrain"]
    if action == "Evaluate":
        return lagrangian_lambdas["lambda_evaluation"]
    return 0.0


def lagrangian_step_reward(base_utility, action, lagrangian_lambdas, config):
    return base_utility - lagrangian_action_penalty(
        action, lagrangian_lambdas, config
    )


def relative_terminal_reward_settings(config):
    settings = config.get("relative_terminal_reward", {})
    baseline_name = settings.get("baseline", "acquire_retrain")
    if baseline_name not in ("acquire_only", "acquire_retrain"):
        raise ValueError(
            "relative_terminal_reward.baseline must be 'acquire_only' or "
            f"'acquire_retrain'; got {baseline_name!r}"
        )
    return {
        "enabled": bool(settings.get("enabled", False)),
        "baseline": baseline_name,
    }


def baseline_terminal_condition_key(config):
    return (
        (
            int(config["acquisition_budget"]),
            int(config["retrain_budget"]),
            int(config["evaluation_budget"]),
        ),
        round(float(config["batch_fraction"]), 12),
    )


def baseline_terminal_introspection_score(experiment, config, baseline_name):
    baseline_curves = baseline_policy_curves(experiment, config)
    if baseline_name == "acquire_only":
        baseline_curve = baseline_curves["acquire_only"]
    else:
        baseline_curve = baseline_curves["acquire_retrain"]
    return float(baseline_curve["introspection_score"].iloc[-1])


def ensure_baseline_terminal_score(experiment, config, baseline_terminal_scores):
    condition_key = baseline_terminal_condition_key(config)
    if condition_key in baseline_terminal_scores:
        return baseline_terminal_scores[condition_key]
    baseline_name = relative_terminal_reward_settings(config)["baseline"]
    baseline_score = baseline_terminal_introspection_score(
        experiment, config, baseline_name
    )
    baseline_terminal_scores[condition_key] = baseline_score
    return baseline_score


def terminal_reward_components(
    terminal_score, experiment, config, baseline_terminal_scores
):
    terminal_score = float(terminal_score)
    if not relative_terminal_reward_settings(config)["enabled"]:
        return {
            "reward_utility": terminal_score,
            "baseline_terminal_score": None,
            "excess_over_baseline": None,
        }
    baseline_score = ensure_baseline_terminal_score(
        experiment, config, baseline_terminal_scores
    )
    excess_over_baseline = terminal_score - baseline_score
    return {
        "reward_utility": excess_over_baseline,
        "baseline_terminal_score": baseline_score,
        "excess_over_baseline": excess_over_baseline,
    }


def compute_terminal_step_reward(
    terminal_score,
    action,
    experiment,
    config,
    lagrangian_lambdas,
    baseline_terminal_scores,
):
    components = terminal_reward_components(
        terminal_score, experiment, config, baseline_terminal_scores
    )
    reward = lagrangian_step_reward(
        components["reward_utility"], action, lagrangian_lambdas, config
    )
    return reward, components


def update_lagrangian_lambdas(lagrangian_lambdas, config, episode_costs):
    settings = lagrangian_settings(config)
    if not settings["enabled"]:
        return
    lagrangian_lambdas["lambda_acquisition"] = max(
        0.0,
        lagrangian_lambdas["lambda_acquisition"]
        + settings["lambda_learning_rate_acquisition"]
        * (
            episode_costs["acquisition_cost"]
            - config["acquisition_budget"]
        ),
    )
    lagrangian_lambdas["lambda_retrain"] = max(
        0.0,
        lagrangian_lambdas["lambda_retrain"]
        + settings["lambda_learning_rate_retrain"]
        * (episode_costs["retrain_cost"] - config["retrain_budget"]),
    )
    lagrangian_lambdas["lambda_evaluation"] = max(
        0.0,
        lagrangian_lambdas["lambda_evaluation"]
        + settings["lambda_learning_rate_evaluation"]
        * (
            episode_costs["evaluation_cost"]
            - config["evaluation_budget"]
        ),
    )


# %% Metrics
def validate_metric_name(metric_name):
    if metric_name not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unknown metric {metric_name!r}; supported: {list(SUPPORTED_METRICS)}"
        )


def metric_from_config(config, role):
    if role == "target":
        metric_name = config.get("target_metric", "f1")
    elif role == "policy":
        metric_name = config.get(
            "policy_metric", config.get("target_metric", "f1")
        )
    elif role == "introspection":
        metric_name = config.get(
            "introspection_metric", config.get("target_metric", "f1")
        )
    else:
        raise ValueError(f"Unknown metric role {role!r}")
    validate_metric_name(metric_name)
    return metric_name


def metric_label(metric_name):
    return METRIC_LABELS.get(metric_name, metric_name)


def metric_value(targets, probabilities, metric_name, threshold=0.5):
    validate_metric_name(metric_name)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets)
    predicted_labels = probabilities >= threshold
    if metric_name == "f1":
        return float(f1_score(targets, predicted_labels, zero_division=0.0))
    if metric_name == "roc_auc":
        return float(roc_auc_score(targets, probabilities))
    if metric_name == "accuracy":
        return float(accuracy_score(targets, predicted_labels))
    if metric_name == "balanced_accuracy":
        return float(balanced_accuracy_score(targets, predicted_labels))
    if metric_name == "pr_auc":
        return float(average_precision_score(targets, probabilities))
    raise ValueError(f"Unhandled metric {metric_name!r}")


def model_metric_on_split(model, features, targets, metric_name, threshold=0.5):
    probabilities = model.predict_proba(features)[:, 1]
    return metric_value(targets, probabilities, metric_name, threshold)


def split_features_targets(experiment, split_name):
    if split_name == "hidden_validation":
        return (
            experiment["hidden_validation_features"],
            experiment["hidden_validation_targets"],
        )
    if split_name == "evaluation":
        return experiment["evaluation_features"], experiment["evaluation_targets"]
    if split_name == "test":
        return experiment["test_features"], experiment["test_targets"]
    raise ValueError(
        f"Unknown split {split_name!r}; "
        "use 'hidden_validation', 'evaluation', or 'test'."
    )


def terminal_hidden_validation_score(model, experiment, config):
    metric_name = metric_from_config(config, "target")
    features, targets = split_features_targets(experiment, "hidden_validation")
    return model_metric_on_split(model, features, targets, metric_name)


def policy_evaluation_score(model, experiment, config):
    metric_name = metric_from_config(config, "policy")
    features, targets = split_features_targets(experiment, "evaluation")
    return model_metric_on_split(model, features, targets, metric_name)


def introspection_score(model, experiment, config):
    eval_set = config.get("introspection_eval_set", "hidden_validation")
    metric_name = metric_from_config(config, "introspection")
    features, targets = split_features_targets(experiment, eval_set)
    return model_metric_on_split(model, features, targets, metric_name)


# %% Data loading and partitioning
def resolve_row_count_config(config):
    config = dict(config)
    split_names = [
        "initial_train",
        "acquisition_pool",
        "evaluation",
        "hidden_validation",
        "test",
    ]
    fraction_keys = [f"{split_name}_fraction" for split_name in split_names]
    supplied_fraction_keys = [key for key in fraction_keys if key in config]

    if supplied_fraction_keys:
        if len(supplied_fraction_keys) != len(fraction_keys):
            missing_keys = sorted(set(fraction_keys) - set(supplied_fraction_keys))
            raise ValueError(f"Missing dataset split fractions: {missing_keys}")

        fractions = np.asarray(
            [float(config[key]) for key in fraction_keys], dtype=np.float64
        )
        if np.any(fractions <= 0.0):
            raise ValueError("All dataset split fractions must be greater than 0")
        if not np.isclose(fractions.sum(), 1.0):
            raise ValueError(
                "Dataset split fractions must sum to 1.0; "
                f"got {fractions.sum():.8f}"
            )

        exact_counts = fractions * int(config["total_rows"])
        row_counts = np.floor(exact_counts).astype(int)
        rows_left = int(config["total_rows"]) - int(row_counts.sum())
        largest_remainders = np.argsort(-(exact_counts - row_counts))
        row_counts[largest_remainders[:rows_left]] += 1
        for split_name, row_count in zip(split_names, row_counts):
            config[f"{split_name}_rows"] = int(row_count)

    if "batch_fraction" in config:
        batch_fraction = float(config["batch_fraction"])
        if not 0.0 < batch_fraction <= 1.0:
            raise ValueError("batch_fraction must satisfy 0 < batch_fraction <= 1")
        config["batch_size"] = batch_size_for_fraction(
            batch_fraction, int(config["total_rows"])
        )

    return config


def batch_size_for_fraction(batch_fraction, total_rows):
    return max(1, int(round(float(batch_fraction) * int(total_rows))))


def resolve_batch_generalization_config(config):
    config = dict(config)
    total_rows = int(config["total_rows"])

    default_rollout_batch_fraction = config.get("default_rollout_batch_fraction")
    if default_rollout_batch_fraction is None:
        default_rollout_batch_fraction = config.get("batch_fraction")
    if default_rollout_batch_fraction is None:
        raise ValueError(
            "Provide batch_fraction or default_rollout_batch_fraction"
        )
    default_rollout_batch_fraction = float(default_rollout_batch_fraction)
    if not 0.0 < default_rollout_batch_fraction <= 1.0:
        raise ValueError(
            "default_rollout_batch_fraction must satisfy 0 < fraction <= 1"
        )

    explicit_training = config.get("training_batch_fractions")
    range_values = config.get("training_batch_fraction_range")
    if explicit_training is not None:
        training_values = sorted({float(value) for value in explicit_training})
    elif range_values is not None:
        if len(range_values) != 2:
            raise ValueError(
                "training_batch_fraction_range must contain [minimum, maximum]"
            )
        minimum, maximum = [float(value) for value in range_values]
        if not 0.0 < minimum <= maximum <= 1.0:
            raise ValueError(
                "training_batch_fraction_range values must satisfy "
                "0 < minimum <= maximum <= 1"
            )
        step = config.get("training_batch_fraction_step")
        if step is None:
            raise ValueError(
                "training_batch_fraction_step is required when "
                "training_batch_fraction_range is set"
            )
        step = float(step)
        if step <= 0.0:
            raise ValueError("training_batch_fraction_step must be > 0")
        training_values = []
        value = minimum
        while value <= maximum + (step * 1e-9):
            training_values.append(round(value, 12))
            value += step
        training_values = sorted(set(training_values))
    else:
        training_values = [default_rollout_batch_fraction]

    for value in training_values:
        if not 0.0 < value <= 1.0:
            raise ValueError(
                "Each training batch fraction must satisfy 0 < fraction <= 1"
            )

    held_out = sorted(
        {
            round(float(value), 12)
            for value in config.get("held_out_batch_fractions", [])
        }
    )
    training_values = [
        value for value in training_values if round(value, 12) not in set(held_out)
    ]
    if not training_values:
        raise ValueError(
            "No training batch fractions remain after held-out exclusions"
        )

    config["training_batch_fractions"] = training_values
    config["held_out_batch_fractions"] = held_out
    config["default_rollout_batch_fraction"] = default_rollout_batch_fraction
    config["batch_fraction"] = default_rollout_batch_fraction
    config["batch_size"] = batch_size_for_fraction(
        default_rollout_batch_fraction, total_rows
    )
    config["training_batch_fraction_maxima"] = max(
        max(training_values),
        default_rollout_batch_fraction,
    )
    return config


def resolve_budget_generalization_config(config):
    config = dict(config)
    budget_names = [
        "acquisition_budget",
        "retrain_budget",
        "evaluation_budget",
    ]
    held_out_config = config.get("held_out_budget_combinations", [])
    default_rollout_budgets = config.get("default_rollout_budgets")
    if default_rollout_budgets is None and all(name in config for name in budget_names):
        default_rollout_budgets = {name: config[name] for name in budget_names}
    if default_rollout_budgets is None and held_out_config:
        default_rollout_budgets = dict(zip(budget_names, held_out_config[0]))
    if default_rollout_budgets is None:
        raise ValueError(
            "Provide default_rollout_budgets, a held-out budget combination, "
            "or the legacy top-level budget keys."
        )
    if any(name not in default_rollout_budgets for name in budget_names):
        raise ValueError(
            "default_rollout_budgets must define acquisition_budget, "
            "retrain_budget, and evaluation_budget"
        )
    default_rollout_budgets = {
        name: int(default_rollout_budgets[name]) for name in budget_names
    }
    if any(value < 1 for value in default_rollout_budgets.values()):
        raise ValueError("Default rollout budgets must be >= 1")

    configured_ranges = config.get("training_budget_ranges")
    if configured_ranges is None:
        configured_ranges = {
            name: [value, value]
            for name, value in default_rollout_budgets.items()
        }

    resolved_ranges = {}
    for name in budget_names:
        if name not in configured_ranges or len(configured_ranges[name]) != 2:
            raise ValueError(
                f"training_budget_ranges[{name!r}] must contain [minimum, maximum]"
            )
        minimum, maximum = [int(value) for value in configured_ranges[name]]
        if minimum < 1 or maximum < minimum:
            raise ValueError(
                f"Invalid range for {name}: [{minimum}, {maximum}]"
            )
        resolved_ranges[name] = [minimum, maximum]

    held_out = {
        tuple(int(value) for value in combination)
        for combination in held_out_config
    }
    if any(len(combination) != 3 for combination in held_out):
        raise ValueError(
            "Each held-out budget combination must contain "
            "[acquisition, retrain, evaluation]"
        )
    combination_count = int(np.prod(
        [maximum - minimum + 1 for minimum, maximum in resolved_ranges.values()]
    ))
    held_out_inside_training_ranges = sum(
        all(
            resolved_ranges[name][0] <= value <= resolved_ranges[name][1]
            for name, value in zip(budget_names, combination)
        )
        for combination in held_out
    )
    if held_out_inside_training_ranges >= combination_count:
        raise ValueError("Held-out combinations leave no budgets for training")

    config["training_budget_ranges"] = resolved_ranges
    config["held_out_budget_combinations"] = [list(values) for values in held_out]
    config["default_rollout_budgets"] = default_rollout_budgets
    config.update(default_rollout_budgets)
    config["training_budget_maxima"] = {
        name: max(resolved_ranges[name][1], default_rollout_budgets[name])
        for name in budget_names
    }
    return config


def load_adult_income():
    adult = fetch_openml(data_id=1590, as_frame=True, parser="auto")
    features = adult.data.copy()
    targets = (
        adult.target.astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
        .eq(">50K")
        .astype("int8")
        .rename("income_above_50k")
    )
    return features, targets


def split_experiment_rows(all_features, all_targets, config):
    total_rows = config["total_rows"]
    initial_train_rows = config["initial_train_rows"]
    acquisition_pool_rows = config["acquisition_pool_rows"]
    evaluation_rows = config["evaluation_rows"]
    hidden_validation_rows = config["hidden_validation_rows"]
    test_rows = config["test_rows"]
    batch_size = config["batch_size"]
    random_seed = config["random_seed"]

    expected_total_rows = (
        initial_train_rows
        + acquisition_pool_rows
        + evaluation_rows
        + hidden_validation_rows
        + test_rows
    )
    assert total_rows == expected_total_rows
    assert batch_size > 0

    if total_rows > len(all_features):
        raise ValueError(
            f"total_rows={total_rows:,} exceeds Adult's "
            f"{len(all_features):,} available rows."
        )

    all_row_ids = np.arange(len(all_features))
    selected_row_ids, _ = train_test_split(
        all_row_ids,
        train_size=total_rows,
        stratify=all_targets,
        random_state=random_seed,
    )
    remaining_row_ids, test_row_ids = train_test_split(
        selected_row_ids,
        test_size=test_rows,
        stratify=all_targets.iloc[selected_row_ids],
        random_state=random_seed + 1,
    )
    remaining_row_ids, hidden_validation_row_ids = train_test_split(
        remaining_row_ids,
        test_size=hidden_validation_rows,
        stratify=all_targets.iloc[remaining_row_ids],
        random_state=random_seed + 2,
    )
    remaining_row_ids, evaluation_row_ids = train_test_split(
        remaining_row_ids,
        test_size=evaluation_rows,
        stratify=all_targets.iloc[remaining_row_ids],
        random_state=random_seed + 3,
    )
    initial_train_row_ids, acquisition_pool_row_ids = train_test_split(
        remaining_row_ids,
        test_size=acquisition_pool_rows,
        stratify=all_targets.iloc[remaining_row_ids],
        random_state=random_seed + 4,
    )

    split_row_ids = [
        initial_train_row_ids,
        acquisition_pool_row_ids,
        evaluation_row_ids,
        hidden_validation_row_ids,
        test_row_ids,
    ]
    assert [len(row_ids) for row_ids in split_row_ids] == [
        initial_train_rows,
        acquisition_pool_rows,
        evaluation_rows,
        hidden_validation_rows,
        test_rows,
    ]
    assert len(np.unique(np.concatenate(split_row_ids))) == total_rows

    def partition(row_ids):
        return (
            all_features.iloc[row_ids].reset_index(drop=True),
            all_targets.iloc[row_ids].reset_index(drop=True),
        )

    (
        initial_train_features_raw,
        initial_train_targets,
    ) = partition(initial_train_row_ids)
    (
        acquisition_pool_features_raw,
        acquisition_pool_targets,
    ) = partition(acquisition_pool_row_ids)
    evaluation_features_raw, evaluation_targets = partition(evaluation_row_ids)
    (
        hidden_validation_features_raw,
        hidden_validation_targets,
    ) = partition(hidden_validation_row_ids)
    test_features_raw, test_targets = partition(test_row_ids)

    positive_fraction = all_targets.iloc[selected_row_ids].mean()
    print(
        f"Adult rows used: {total_rows:,}; "
        f"features: {all_features.shape[1]}; "
        f"positive fraction: {positive_fraction:.3f}"
    )

    return {
        "initial_train_features_raw": initial_train_features_raw,
        "initial_train_targets": initial_train_targets,
        "acquisition_pool_features_raw": acquisition_pool_features_raw,
        "acquisition_pool_targets": acquisition_pool_targets,
        "evaluation_features_raw": evaluation_features_raw,
        "evaluation_targets": evaluation_targets,
        "hidden_validation_features_raw": hidden_validation_features_raw,
        "hidden_validation_targets": hidden_validation_targets,
        "test_features_raw": test_features_raw,
        "test_targets": test_targets,
    }


# %% Feature encoding and model factory
def downstream_model_settings(config):
    if isinstance(config.get("downstream_model"), dict):
        settings = dict(config["downstream_model"])
        model_type = settings.pop("type", "hist_gradient_boosting")
        return model_type, settings
    return (
        config.get("downstream_model_type", "hist_gradient_boosting"),
        dict(config.get("model_parameters", {})),
    )


def _require_no_unused_parameters(parameters, allowed_names, model_type):
    unused = sorted(set(parameters) - set(allowed_names))
    if unused:
        raise ValueError(
            f"Unused {model_type} downstream_model parameters: {unused}"
        )


def make_downstream_model(config, categorical_feature_mask):
    model_type, parameters = downstream_model_settings(config)
    parameters = dict(parameters)

    if model_type == "hist_gradient_boosting":
        allowed_names = {
            "learning_rate",
            "max_iter",
            "max_leaf_nodes",
            "min_samples_leaf",
            "max_depth",
            "random_state",
            "l2_regularization",
            "max_bins",
            "early_stopping",
            "validation_fraction",
            "n_iter_no_change",
        }
        _require_no_unused_parameters(parameters, allowed_names, model_type)
        if "random_state" not in parameters:
            parameters["random_state"] = config["random_seed"]
        return HistGradientBoostingClassifier(
            **parameters,
            categorical_features=categorical_feature_mask,
        )

    if model_type == "logistic_regression":
        allowed_names = {"max_iter", "C", "random_state"}
        _require_no_unused_parameters(parameters, allowed_names, model_type)
        logistic_parameters = {
            "max_iter": int(parameters.pop("max_iter", 1000)),
            "C": float(parameters.pop("C", 1.0)),
            "random_state": int(
                parameters.pop("random_state", config["random_seed"])
            ),
        }
        return Pipeline(
            [
                ("standard_scaler", StandardScaler()),
                (
                    "logistic_regression",
                    LogisticRegression(**logistic_parameters),
                ),
            ]
        )

    if model_type == "extra_trees_classifier":
        allowed_names = {
            "n_estimators",
            "min_samples_leaf",
            "max_depth",
            "random_state",
            "n_jobs",
        }
        _require_no_unused_parameters(parameters, allowed_names, model_type)
        max_depth = parameters.pop("max_depth", None)
        if max_depth is not None:
            max_depth = int(max_depth)
        return ExtraTreesClassifier(
            n_estimators=int(parameters.pop("n_estimators", 200)),
            min_samples_leaf=int(parameters.pop("min_samples_leaf", 10)),
            max_depth=max_depth,
            random_state=int(
                parameters.pop("random_state", config["random_seed"])
            ),
            n_jobs=int(parameters.pop("n_jobs", -1)),
        )

    raise ValueError(
        "Unknown downstream_model type "
        f"{model_type!r}; use 'hist_gradient_boosting', "
        "'logistic_regression', or 'extra_trees_classifier'."
    )


def build_preprocessor(initial_train_features_raw, acquisition_pool_features_raw, config):
    all_features_raw = pd.concat(
        [initial_train_features_raw, acquisition_pool_features_raw],
        ignore_index=True,
    )
    numerical_columns = all_features_raw.select_dtypes(
        include=["number"]
    ).columns.tolist()
    categorical_columns = all_features_raw.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                SimpleImputer(strategy="median"),
                numerical_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                dtype=np.float32,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.fit(all_features_raw)

    number_of_numerical_features = len(numerical_columns)
    number_of_categorical_features = len(categorical_columns)
    categorical_feature_mask = np.array(
        [False] * number_of_numerical_features
        + [True] * number_of_categorical_features,
        dtype=bool,
    )

    def encode_features(features):
        encoded = preprocessor.transform(features)
        return np.asarray(encoded, dtype=np.float32)

    def make_model():
        return make_downstream_model(config, categorical_feature_mask)

    return encode_features, make_model


def encode_experiment_features(raw_partitions, encode_features):
    encoded = {}
    for partition_name, raw_frame in raw_partitions.items():
        encoded[partition_name.replace("_raw", "")] = encode_features(raw_frame)
    return encoded


def build_experiment(config):
    config = resolve_row_count_config(config)
    config = resolve_budget_generalization_config(config)
    config = resolve_batch_generalization_config(config)
    raw_partitions = split_experiment_rows(*load_adult_income(), config)
    encode_features, make_model = build_preprocessor(
        raw_partitions["initial_train_features_raw"],
        raw_partitions["acquisition_pool_features_raw"],
        config,
    )
    encoded_partitions = encode_experiment_features(
        {key: value for key, value in raw_partitions.items() if key.endswith("_raw")},
        encode_features,
    )
    return {
        **encoded_partitions,
        **{
            key: value
            for key, value in raw_partitions.items()
            if not key.endswith("_raw")
        },
        "make_model": make_model,
        "config": config,
        **(
            {"baseline_terminal_scores": {}}
            if relative_terminal_reward_settings(config)["enabled"]
            else {}
        ),
    }


# %% Fitted-Q helpers
def make_q_models(config):
    q_model_config = config.get("q_model", {"type": "linear_regression"})
    if isinstance(q_model_config, dict):
        q_model_type = q_model_config.get("type", "linear_regression")
        if q_model_type == "linear_regression":
            q_model = LinearRegression()
        elif q_model_type == "ridge_regression":
            alpha = float(q_model_config.get("alpha", 1.0))
            if alpha <= 0.0:
                raise ValueError("q_model Ridge alpha must be > 0")
            q_model = Pipeline(
                [
                    ("standard_scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=alpha)),
                ]
            )
        elif q_model_type == "polynomial_regression":
            degree = int(q_model_config.get("degree", 2))
            if degree < 1:
                raise ValueError("q_model polynomial degree must be >= 1")
            q_model = Pipeline(
                [
                    (
                        "polynomial_features",
                        PolynomialFeatures(degree=degree, include_bias=False),
                    ),
                    ("linear_regression", LinearRegression()),
                ]
            )
        elif q_model_type == "polynomial_ridge":
            degree = int(q_model_config.get("degree", 2))
            alpha = float(q_model_config.get("alpha", 1.0))
            if degree < 1:
                raise ValueError("q_model polynomial degree must be >= 1")
            if alpha <= 0.0:
                raise ValueError("q_model Ridge alpha must be > 0")
            q_model = Pipeline(
                [
                    ("standard_scaler", StandardScaler()),
                    (
                        "polynomial_features",
                        PolynomialFeatures(degree=degree, include_bias=False),
                    ),
                    ("ridge", Ridge(alpha=alpha)),
                ]
            )
        elif q_model_type == "extra_trees_regressor":
            n_estimators = int(q_model_config.get("n_estimators", 200))
            min_samples_leaf = int(q_model_config.get("min_samples_leaf", 10))
            max_depth = q_model_config.get("max_depth")
            if max_depth is not None:
                max_depth = int(max_depth)
            if n_estimators < 1:
                raise ValueError("q_model n_estimators must be >= 1")
            if min_samples_leaf < 1:
                raise ValueError("q_model min_samples_leaf must be >= 1")
            if max_depth is not None and max_depth < 1:
                raise ValueError("q_model max_depth must be >= 1 or None")
            q_model = ExtraTreesRegressor(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                max_depth=max_depth,
                random_state=int(
                    q_model_config.get("random_state", config["random_seed"])
                ),
                n_jobs=int(q_model_config.get("n_jobs", -1)),
            )
        else:
            raise ValueError(
                "Unknown q_model type "
                f"{q_model_type!r}; use 'linear_regression', "
                "'ridge_regression', 'polynomial_regression', or "
                "'polynomial_ridge', or 'extra_trees_regressor'."
            )
    else:
        q_model = q_model_config
    return {action: clone(q_model) for action in Q_ACTIONS}


def make_q_model_is_fitted():
    return {action: False for action in Q_ACTIONS}


def row_uncertainty_scores(model, pool_features, row_indices):
    if len(row_indices) == 0:
        return np.array([], dtype=np.float64)
    probabilities = model.predict_proba(pool_features[row_indices])[:, 1]
    probabilities = np.clip(
        np.asarray(probabilities, dtype=np.float64), 1e-9, 1.0 - 1e-9
    )
    one_minus_probabilities = 1.0 - probabilities
    return (
        -probabilities * np.log(probabilities)
        - one_minus_probabilities * np.log(one_minus_probabilities)
    )


def ensure_pool_uncertainty_cache(
    model,
    pool_features,
    pool_row_count,
    cache_key,
    pool_uncertainty_cache,
    cached_key,
):
    if pool_uncertainty_cache is not None and cached_key == cache_key:
        return pool_uncertainty_cache, cached_key
    pool_uncertainty_cache = row_uncertainty_scores(
        model,
        pool_features,
        np.arange(pool_row_count, dtype=np.int64),
    )
    return pool_uncertainty_cache, cache_key


def select_top_k_pool_rows(
    model,
    pool_features,
    available_row_indices,
    batch_size,
    config,
    pool_uncertainty_cache=None,
):
    """Pick highest-uncertainty rows inside a pool percentile band."""
    if len(available_row_indices) == 0:
        raise ValueError("No pool rows available to acquire.")

    percentile_low = float(config.get("acquisition_percentile_low", 0.0))
    percentile_high = float(config.get("acquisition_percentile_high", 100.0))
    if not (0.0 <= percentile_low < percentile_high <= 100.0):
        raise ValueError(
            "acquisition_percentile_low and acquisition_percentile_high must satisfy "
            "0 <= low < high <= 100; "
            f"got low={percentile_low}, high={percentile_high}"
        )

    if pool_uncertainty_cache is not None:
        if len(pool_uncertainty_cache) != pool_features.shape[0]:
            raise ValueError(
                "pool_uncertainty_cache length must match acquisition pool rows"
            )
        uncertainties = pool_uncertainty_cache[available_row_indices]
    else:
        uncertainties = row_uncertainty_scores(
            model, pool_features, available_row_indices
        )
    row_count_to_acquire = min(batch_size, len(available_row_indices))

    low_cut = np.percentile(uncertainties, percentile_low)
    high_cut = np.percentile(uncertainties, percentile_high)
    in_band = (uncertainties >= low_cut) & (uncertainties <= high_cut)
    eligible_positions = np.where(in_band)[0]
    if len(eligible_positions) == 0:
        eligible_positions = np.arange(len(available_row_indices))

    eligible_uncertainties = uncertainties[eligible_positions]
    pick_count = min(row_count_to_acquire, len(eligible_positions))
    top_within_eligible = np.argsort(eligible_uncertainties)[-pick_count:]
    top_positions = eligible_positions[top_within_eligible]

    selected_row_indices = available_row_indices[top_positions]
    mean_row_utility = float(np.mean(uncertainties[top_positions]))
    remaining_row_indices = np.delete(available_row_indices, top_positions)
    return selected_row_indices, mean_row_utility, remaining_row_indices


def assert_acquired_rows_not_in_pool(available_row_indices, pending_row_indices):
    if len(available_row_indices) == 0 or len(pending_row_indices) == 0:
        return
    overlap = np.intersect1d(available_row_indices, pending_row_indices)
    if len(overlap) > 0:
        raise RuntimeError(
            "Acquired pool rows must not be selectable again; overlap: "
            f"{overlap.tolist()}"
        )


def make_state(
    pending_row_count,
    remaining_acquisition_budget,
    remaining_retrain_budget,
    remaining_evaluation_budget,
    latest_evaluation_score,
    last_eval_score_change,
    rolling_eval_score_change_features,
    config,
):
    last_eval_score_or_zero = (
        0.0 if latest_evaluation_score is None else float(latest_evaluation_score)
    )
    training_budget_maxima = config.get(
        "training_budget_maxima",
        {
            "acquisition_budget": config["acquisition_budget"],
            "retrain_budget": config["retrain_budget"],
            "evaluation_budget": config["evaluation_budget"],
        },
    )
    training_batch_fraction_maxima = float(
        config.get("training_batch_fraction_maxima", config["batch_fraction"])
    )
    windows = evaluation_return_windows_from_config(config)
    rolling_features = [float(value) for value in rolling_eval_score_change_features]
    if len(rolling_features) != len(windows):
        raise ValueError(
            "rolling_eval_score_change_features length "
            f"{len(rolling_features)} does not match "
            f"evaluation_return_windows length {len(windows)}"
        )
    return np.asarray(
        [
            float(config["acquisition_budget"])
            / float(training_budget_maxima["acquisition_budget"]),
            float(config["retrain_budget"])
            / float(training_budget_maxima["retrain_budget"]),
            float(config["evaluation_budget"])
            / float(training_budget_maxima["evaluation_budget"]),
            float(remaining_acquisition_budget)
            / float(training_budget_maxima["acquisition_budget"]),
            float(remaining_retrain_budget)
            / float(training_budget_maxima["retrain_budget"]),
            float(remaining_evaluation_budget)
            / float(training_budget_maxima["evaluation_budget"]),
            float(pending_row_count) / float(config["total_rows"]),
            float(config["batch_fraction"]) / training_batch_fraction_maxima,
            last_eval_score_or_zero,
            float(last_eval_score_change),
            *rolling_features,
        ],
        dtype=np.float64,
    )


def feasible_actions(
    available_row_count,
    pending_row_count,
    remaining_acquisition_budget,
    remaining_retrain_budget,
    remaining_evaluation_budget,
    model_needs_evaluation,
    config,
):
    actions = []
    can_acquire = (
        available_row_count > 0
        and remaining_acquisition_budget > 0
    )
    can_retrain = pending_row_count > 0 and remaining_retrain_budget > 0
    if can_acquire:
        actions.append("Acquire")
    if can_retrain:
        actions.append("Retrain")
    if (
        can_acquire
        and model_needs_evaluation
        and remaining_evaluation_budget > 0
    ):
        actions.append("Evaluate")
    if config.get("allow_stop_action", True):
        actions.append("Stop")
    return actions


def predicted_q_by_action(state, q_models, q_model_is_fitted):
    return {
        f"q_{action.lower()}": predict_q(state, action, q_models, q_model_is_fitted)
        for action in Q_ACTIONS
    }


EPISODE_ACTION_FILL_COLORS = {
    "Initial": "rgba(158, 158, 158, 0.18)",
    "Acquire": "rgba(31, 119, 180, 0.18)",
    "Retrain": "rgba(44, 160, 44, 0.18)",
    "Evaluate": "rgba(255, 127, 14, 0.18)",
    "Stop": "rgba(214, 39, 40, 0.18)",
    "FinalRetrain": "rgba(148, 103, 189, 0.22)",
}

EPISODE_ACTION_LINE_COLORS = {
    "Acquire": "rgb(31, 119, 180)",
    "Retrain": "rgb(44, 160, 44)",
    "Evaluate": "rgb(255, 127, 14)",
    "Stop": "rgb(214, 39, 40)",
}

EPISODE_ACTION_BAND_LEGEND = [
    ("Initial", "rgb(158, 158, 158)"),
    ("Acquire", EPISODE_ACTION_LINE_COLORS["Acquire"]),
    ("Retrain", EPISODE_ACTION_LINE_COLORS["Retrain"]),
    ("Evaluate", EPISODE_ACTION_LINE_COLORS["Evaluate"]),
    ("Stop", EPISODE_ACTION_LINE_COLORS["Stop"]),
    ("FinalRetrain", "rgb(148, 103, 189)"),
]


def finalize_episode_model(
    config,
    make_model,
    model,
    model_version,
    training_features,
    training_targets,
    pending_row_indices,
    pool_features,
    pool_targets,
):
    final_pending_retrain_rows = 0
    if config.get("final_retrain_pending_rows", False) and len(pending_row_indices) > 0:
        final_pending_retrain_rows = len(pending_row_indices)
        training_features = np.vstack(
            [training_features, pool_features[pending_row_indices]]
        )
        training_targets = pd.concat(
            [training_targets, pool_targets.iloc[pending_row_indices]],
            ignore_index=True,
        )
        pending_row_indices = np.array([], dtype=np.int64)
        model = make_model()
        model.fit(training_features, training_targets)
        model_version += 1
    return {
        "model": model,
        "model_version": model_version,
        "training_features": training_features,
        "training_targets": training_targets,
        "pending_row_indices": pending_row_indices,
        "final_pending_retrain_rows": final_pending_retrain_rows,
    }


def append_final_retrain_diagnostic(
    step_diagnostics,
    step_number,
    model,
    experiment,
    config,
    q_models,
    q_model_is_fitted,
    model_version,
    training_rows,
    remaining_acquisition_budget,
    remaining_retrain_budget,
    remaining_evaluation_budget,
    latest_evaluation_score,
    last_eval_score_change,
    rolling_eval_score_change_features,
):
    state = make_state(
        pending_row_count=0,
        remaining_acquisition_budget=remaining_acquisition_budget,
        remaining_retrain_budget=remaining_retrain_budget,
        remaining_evaluation_budget=remaining_evaluation_budget,
        latest_evaluation_score=latest_evaluation_score,
        last_eval_score_change=last_eval_score_change,
        rolling_eval_score_change_features=rolling_eval_score_change_features,
        config=config,
    )
    step_diagnostics.append(
        {
            "step": step_number,
            "action": "FinalRetrain",
            **predicted_q_by_action(state, q_models, q_model_is_fitted),
            "introspection_score": introspection_score(model, experiment, config),
            "model_version": model_version,
            "pending_rows": 0,
            "training_rows": training_rows,
            "acquired_row_indices": [],
        }
    )


def baseline_policy_curves(experiment, config):
    """Run batch-acquisition and alternating acquire/retrain baselines."""
    make_model = experiment["make_model"]
    pool_features = experiment["acquisition_pool_features"]
    pool_targets = experiment["acquisition_pool_targets"]
    batch_size = config["batch_size"]

    # Baseline 1: acquire every batch with the unchanged initial model, then
    # perform one final retrain on all acquired rows.
    acquire_only_model = make_model()
    acquire_only_model.fit(
        experiment["initial_train_features"],
        experiment["initial_train_targets"],
    )
    acquire_only_available_rows = np.arange(
        config["acquisition_pool_rows"], dtype=np.int64
    )
    acquire_only_records = [
        {
            "step": 0,
            "action": "Initial",
            "introspection_score": introspection_score(
                acquire_only_model, experiment, config
            ),
            "acquisition_cost": 0,
            "retrain_cost": 0,
        }
    ]
    acquire_only_cost = 0
    acquire_only_selected_rows = []
    acquire_only_uncertainty_cache, _ = ensure_pool_uncertainty_cache(
        acquire_only_model,
        pool_features,
        config["acquisition_pool_rows"],
        0,
        None,
        None,
    )
    while (
        acquire_only_cost < config["acquisition_budget"]
        and len(acquire_only_available_rows) > 0
    ):
        selected_row_indices, _, acquire_only_available_rows = select_top_k_pool_rows(
            acquire_only_model,
            pool_features,
            acquire_only_available_rows,
            batch_size,
            config,
            pool_uncertainty_cache=acquire_only_uncertainty_cache,
        )
        acquire_only_selected_rows.extend(selected_row_indices.tolist())
        acquire_only_cost += 1
        acquire_only_records.append(
            {
                "step": len(acquire_only_records),
                "action": "Acquire",
                "introspection_score": introspection_score(
                    acquire_only_model, experiment, config
                ),
                "acquisition_cost": acquire_only_cost,
                "retrain_cost": 0,
            }
        )
    acquire_only_retrain_cost = 0
    if acquire_only_selected_rows and config["retrain_budget"] > 0:
        acquired_row_indices = np.asarray(
            acquire_only_selected_rows, dtype=np.int64
        )
        acquire_only_training_features = np.vstack(
            [
                experiment["initial_train_features"],
                pool_features[acquired_row_indices],
            ]
        )
        acquire_only_training_targets = pd.concat(
            [
                experiment["initial_train_targets"],
                pool_targets.iloc[acquired_row_indices],
            ],
            ignore_index=True,
        )
        acquire_only_model = make_model()
        acquire_only_model.fit(
            acquire_only_training_features,
            acquire_only_training_targets,
        )
        acquire_only_retrain_cost = 1
        acquire_only_records.append(
            {
                "step": len(acquire_only_records),
                "action": "FinalRetrain",
                "introspection_score": introspection_score(
                    acquire_only_model, experiment, config
                ),
                "acquisition_cost": acquire_only_cost,
                "retrain_cost": acquire_only_retrain_cost,
            }
        )
    acquire_only_records.append(
        {
            "step": len(acquire_only_records),
            "action": "Stop",
            "introspection_score": introspection_score(
                acquire_only_model, experiment, config
            ),
            "acquisition_cost": acquire_only_cost,
            "retrain_cost": acquire_only_retrain_cost,
        }
    )

    # Baseline 2: alternate Acquire -> Retrain while both budgets allow a pair.
    # After the retrain budget is exhausted, keep acquiring with the last fitted
    # model. If rows were acquired since the last retrain, apply one final
    # retrain before Stop (same spirit as the acquire-only baseline).
    training_features = experiment["initial_train_features"].copy()
    training_targets = experiment["initial_train_targets"].copy()
    available_row_indices = np.arange(config["acquisition_pool_rows"], dtype=np.int64)
    remaining_acquisition_budget = config["acquisition_budget"]
    remaining_retrain_budget = config["retrain_budget"]
    model = make_model()
    model.fit(training_features, training_targets)
    model_is_current = True
    acquire_retrain_model_key = 0
    pool_uncertainty_cache = None
    cached_uncertainty_key = None
    acquire_retrain_records = [
        {
            "step": 0,
            "action": "Initial",
            "introspection_score": introspection_score(model, experiment, config),
            "acquisition_cost": 0,
            "retrain_cost": 0,
        }
    ]
    while (
        remaining_acquisition_budget > 0
        and remaining_retrain_budget > 0
        and len(available_row_indices) > 0
    ):
        pool_uncertainty_cache, cached_uncertainty_key = ensure_pool_uncertainty_cache(
            model,
            pool_features,
            config["acquisition_pool_rows"],
            acquire_retrain_model_key,
            pool_uncertainty_cache,
            cached_uncertainty_key,
        )
        (
            selected_row_indices,
            _mean_row_utility,
            available_row_indices,
        ) = select_top_k_pool_rows(
            model,
            pool_features,
            available_row_indices,
            batch_size,
            config,
            pool_uncertainty_cache=pool_uncertainty_cache,
        )
        remaining_acquisition_budget -= 1
        acquire_retrain_records.append(
            {
                "step": len(acquire_retrain_records),
                "action": "Acquire",
                "introspection_score": introspection_score(
                    model, experiment, config
                ),
                "acquisition_cost": config["acquisition_budget"]
                - remaining_acquisition_budget,
                "retrain_cost": config["retrain_budget"]
                - remaining_retrain_budget,
            }
        )
        training_features = np.vstack(
            [training_features, pool_features[selected_row_indices]]
        )
        training_targets = pd.concat(
            [training_targets, pool_targets.iloc[selected_row_indices]],
            ignore_index=True,
        )
        model_is_current = False
        model = make_model()
        model.fit(training_features, training_targets)
        remaining_retrain_budget -= 1
        model_is_current = True
        acquire_retrain_model_key += 1
        acquire_retrain_records.append(
            {
                "step": len(acquire_retrain_records),
                "action": "Retrain",
                "introspection_score": introspection_score(
                    model, experiment, config
                ),
                "acquisition_cost": config["acquisition_budget"]
                - remaining_acquisition_budget,
                "retrain_cost": config["retrain_budget"]
                - remaining_retrain_budget,
            }
        )
    while remaining_acquisition_budget > 0 and len(available_row_indices) > 0:
        pool_uncertainty_cache, cached_uncertainty_key = ensure_pool_uncertainty_cache(
            model,
            pool_features,
            config["acquisition_pool_rows"],
            acquire_retrain_model_key,
            pool_uncertainty_cache,
            cached_uncertainty_key,
        )
        (
            selected_row_indices,
            _mean_row_utility,
            available_row_indices,
        ) = select_top_k_pool_rows(
            model,
            pool_features,
            available_row_indices,
            batch_size,
            config,
            pool_uncertainty_cache=pool_uncertainty_cache,
        )
        remaining_acquisition_budget -= 1
        training_features = np.vstack(
            [training_features, pool_features[selected_row_indices]]
        )
        training_targets = pd.concat(
            [training_targets, pool_targets.iloc[selected_row_indices]],
            ignore_index=True,
        )
        model_is_current = False
        acquire_retrain_records.append(
            {
                "step": len(acquire_retrain_records),
                "action": "Acquire",
                "introspection_score": introspection_score(
                    model, experiment, config
                ),
                "acquisition_cost": config["acquisition_budget"]
                - remaining_acquisition_budget,
                "retrain_cost": config["retrain_budget"]
                - remaining_retrain_budget,
            }
        )
    acquisition_cost_so_far = (
        config["acquisition_budget"] - remaining_acquisition_budget
    )
    retrain_cost_so_far = config["retrain_budget"] - remaining_retrain_budget
    if not model_is_current and config["retrain_budget"] > 0:
        model = make_model()
        model.fit(training_features, training_targets)
        model_is_current = True
        retrain_cost_so_far += 1
        acquire_retrain_records.append(
            {
                "step": len(acquire_retrain_records),
                "action": "FinalRetrain",
                "introspection_score": introspection_score(
                    model, experiment, config
                ),
                "acquisition_cost": acquisition_cost_so_far,
                "retrain_cost": retrain_cost_so_far,
            }
        )
    acquire_retrain_records.append(
        {
            "step": len(acquire_retrain_records),
            "action": "Stop",
            "introspection_score": introspection_score(model, experiment, config),
            "acquisition_cost": acquisition_cost_so_far,
            "retrain_cost": retrain_cost_so_far,
        }
    )

    return {
        "acquire_only": pd.DataFrame(acquire_only_records),
        "acquire_retrain": pd.DataFrame(acquire_retrain_records),
    }


def plot_episode_step_diagnostics(step_diagnostics, config, experiment=None):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    diagnostic_frame = pd.DataFrame(step_diagnostics).sort_values("step")
    if "model_version" not in diagnostic_frame.columns:
        diagnostic_frame["model_version"] = 0
    if "pending_rows" not in diagnostic_frame.columns:
        diagnostic_frame["pending_rows"] = 0
    if "acquisitions_so_far" not in diagnostic_frame.columns:
        diagnostic_frame = diagnostic_frame.copy()
        diagnostic_frame["acquisitions_so_far"] = 0

    eval_set = config.get("introspection_eval_set", "hidden_validation")
    introspection_metric = metric_from_config(config, "introspection")
    introspection_label = (
        f"{eval_set.replace('_', ' ')} {metric_label(introspection_metric)} (after step)"
    )
    target_metric = metric_from_config(config, "target")
    policy_metric = metric_from_config(config, "policy")

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            (
                f"Top: model quality after each step — "
                f"{metric_label(introspection_metric)} on {eval_set.replace('_', ' ')} "
                "(agent and two fixed baseline policies)"
            ),
            "Bottom: predicted Q at decision time, before the banded action runs",
        ),
    )
    figure.add_trace(
        go.Scatter(
            x=diagnostic_frame["step"],
            y=diagnostic_frame["introspection_score"],
            mode="lines+markers",
            name=f"fitted Q policy: {introspection_label}",
            line={"width": 3, "color": "rgb(214, 39, 40)"},
            marker={"size": 8, "color": "rgb(214, 39, 40)"},
            customdata=diagnostic_frame[
                ["model_version", "pending_rows", "action"]
            ].to_numpy(),
            hovertemplate=(
                "Step %{x}<br>"
                f"{introspection_label}=%{{y:.4f}}<br>"
                "model_version=%{customdata[0]}<br>"
                "pending_rows=%{customdata[1]}<br>"
                "action=%{customdata[2]}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    if experiment is not None:
        benchmark_config = {**experiment["config"], **config}
        benchmark_config = resolve_row_count_config(benchmark_config)
        baseline_curves = baseline_policy_curves(experiment, benchmark_config)
        acquire_only_curve = baseline_curves["acquire_only"]
        acquire_retrain_curve = baseline_curves["acquire_retrain"]
        eval_set_label = eval_set.replace("_", " ")
        figure.add_trace(
            go.Scatter(
                x=acquire_only_curve["step"],
                y=acquire_only_curve["introspection_score"],
                mode="lines+markers",
                name=(
                    f"initial-model acquisition → final retrain: {eval_set_label} "
                    f"{metric_label(introspection_metric)}"
                ),
                line={"width": 2, "dash": "dot", "color": "rgb(120, 120, 120)"},
                marker={"size": 7, "symbol": "circle-open"},
                customdata=acquire_only_curve[
                    ["action", "acquisition_cost", "retrain_cost"]
                ].to_numpy(),
                hovertemplate=(
                    "Step %{x}<br>"
                    f"batch acquisition {metric_label(introspection_metric)}=%{{y:.4f}}<br>"
                    "action=%{customdata[0]}<br>"
                    "acquisition cost=%{customdata[1]}<br>"
                    "retrain cost=%{customdata[2]}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=acquire_retrain_curve["step"],
                y=acquire_retrain_curve["introspection_score"],
                mode="lines+markers",
                name=(
                    f"acquire → retrain baseline: {eval_set_label} "
                    f"{metric_label(introspection_metric)}"
                ),
                line={"width": 2, "dash": "dash", "color": "rgb(31, 119, 180)"},
                marker={"size": 7, "symbol": "triangle-up"},
                customdata=acquire_retrain_curve[
                    ["action", "acquisition_cost", "retrain_cost"]
                ].to_numpy(),
                hovertemplate=(
                    "Step %{x}<br>"
                    f"acquire → retrain {metric_label(introspection_metric)}=%{{y:.4f}}<br>"
                    "action=%{customdata[0]}<br>"
                    "acquisition cost=%{customdata[1]}<br>"
                    "retrain cost=%{customdata[2]}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
    for action in Q_ACTIONS:
        column_name = f"q_{action.lower()}"
        figure.add_trace(
            go.Scatter(
                x=diagnostic_frame["step"],
                y=diagnostic_frame[column_name],
                mode="lines+markers",
                name=f"Q({action})",
                line={"color": EPISODE_ACTION_LINE_COLORS[action]},
                marker={"color": EPISODE_ACTION_LINE_COLORS[action]},
                customdata=diagnostic_frame["action"].to_numpy(),
                hovertemplate=(
                    "Step %{x}<br>"
                    f"Q({action})=%{{y:.4f}}<br>"
                    "banded action at this step=%{customdata}<extra></extra>"
                ),
                legendgroup="q_lines",
            ),
            row=2,
            col=1,
        )

    for action_name, band_color in EPISODE_ACTION_BAND_LEGEND:
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={
                    "size": 12,
                    "color": band_color,
                    "symbol": "square",
                },
                name=f"band: {action_name}",
                legendgroup="action_bands",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    for _, row in diagnostic_frame.iterrows():
        fill_color = EPISODE_ACTION_FILL_COLORS.get(
            row["action"], "rgba(200, 200, 200, 0.15)"
        )
        for subplot_row in (1, 2):
            figure.add_vrect(
                x0=row["step"] - 0.5,
                x1=row["step"] + 0.5,
                fillcolor=fill_color,
                layer="below",
                line_width=0,
                row=subplot_row,
                col=1,
            )

    figure.update_layout(
        title=(
            "Final episode diagnostics "
            f"(target={metric_label(target_metric)}, policy={metric_label(policy_metric)}). "
            "Vertical bands = action at that step (see legend). "
            f"Top = policy-model {metric_label(introspection_metric)} after the action "
            "plus budget-respecting baselines; bottom = Q before the action."
        ),
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.08, "x": 0},
        height=720,
    )
    figure.update_xaxes(title_text="Step", row=2, col=1)
    figure.update_yaxes(
        title_text=metric_label(introspection_metric), row=1, col=1
    )
    q_scale_column_names = [f"q_{action.lower()}" for action in ("Acquire", "Retrain")]
    q_scale_values = diagnostic_frame[q_scale_column_names].to_numpy(dtype=float).ravel()
    finite_q_values = q_scale_values[np.isfinite(q_scale_values)]
    if len(finite_q_values) > 0:
        q_low, q_high = np.percentile(finite_q_values, [5, 95])
        if q_low == q_high:
            q_low -= 1.0
            q_high += 1.0
        q_padding = 0.08 * (q_high - q_low)
        figure.update_yaxes(
            range=[q_low - q_padding, q_high + q_padding],
            title_text="Predicted Q (Acquire/Retrain scale)",
            row=2,
            col=1,
        )
    else:
        figure.update_yaxes(title_text="Predicted Q", row=2, col=1)
    return figure


def epsilon_for_episode(episode_number, config):
    number_of_training_episodes = config["number_of_training_episodes"]
    if number_of_training_episodes == 1:
        return config["final_epsilon"]
    return config["initial_epsilon"] + (
        (config["final_epsilon"] - config["initial_epsilon"])
        * episode_number
        / (number_of_training_episodes - 1)
    )


def predict_q(state, action, q_models, q_model_is_fitted):
    if not q_model_is_fitted[action]:
        return 0.0
    return float(q_models[action].predict(state.reshape(1, -1))[0])


def max_predicted_q(next_state, next_feasible_actions, q_models, q_model_is_fitted):
    if not next_feasible_actions:
        return 0.0

    q_values = [
        predict_q(next_state, next_action, q_models, q_model_is_fitted)
        for next_action in next_feasible_actions
    ]
    finite_q_values = [q_value for q_value in q_values if np.isfinite(q_value)]
    if not finite_q_values:
        return 0.0
    return max(finite_q_values)


def choose_action(
    q_models,
    q_model_is_fitted,
    state,
    feasible_action_list,
    episode_random,
    epsilon,
    use_exploration,
):
    if use_exploration and episode_random.random() < epsilon:
        return episode_random.choice(feasible_action_list)

    q_by_action = {
        action: predict_q(state, action, q_models, q_model_is_fitted)
        for action in feasible_action_list
    }
    finite_q_by_action = {
        action: q_value
        for action, q_value in q_by_action.items()
        if np.isfinite(q_value)
    }
    if not finite_q_by_action:
        return episode_random.choice(feasible_action_list)

    best_q_value = max(finite_q_by_action.values())
    best_actions = [
        action
        for action, q_value in finite_q_by_action.items()
        if q_value == best_q_value
    ]
    return episode_random.choice(best_actions)


def compute_transition_target(transition, q_models, q_model_is_fitted, config):
    if transition["done"]:
        return transition["reward"]
    return transition["reward"] + config["discount_factor"] * max_predicted_q(
        transition["next_state"],
        transition["next_feasible_actions"],
        q_models,
        q_model_is_fitted,
    )


def fit_q_models(replay_buffer, q_models, q_model_is_fitted, config):
    if not replay_buffer:
        return

    transitions_by_action_and_condition = {action: {} for action in Q_ACTIONS}

    for transition in replay_buffer:
        action = transition["action"]
        condition_key = replay_condition_key(transition)
        transitions_by_action_and_condition[action].setdefault(
            condition_key, []
        ).append(transition)

    for action in Q_ACTIONS:
        transitions_by_condition = transitions_by_action_and_condition[action]
        if not transitions_by_condition:
            continue

        samples_per_condition = min(
            len(transitions) for transitions in transitions_by_condition.values()
        )
        balanced_transitions = []
        for transitions in transitions_by_condition.values():
            balanced_transitions.extend(transitions[-samples_per_condition:])

        if len(balanced_transitions) < config["min_samples_per_action"]:
            continue

        states = np.vstack(
            [transition["state"] for transition in balanced_transitions]
        )
        targets = np.asarray(
            [
                compute_transition_target(
                    transition, q_models, q_model_is_fitted, config
                )
                for transition in balanced_transitions
            ],
            dtype=np.float64,
        )
        finite_rows = np.isfinite(states).all(axis=1) & np.isfinite(targets)
        states = states[finite_rows]
        targets = targets[finite_rows]
        if len(targets) < config["min_samples_per_action"]:
            continue
        q_models[action].fit(states, targets)
        q_model_is_fitted[action] = True


def count_replay_samples_by_action(replay_buffer):
    counts = {action: 0 for action in Q_ACTIONS}
    for transition in replay_buffer:
        counts[transition["action"]] += 1
    return counts


def states_from_replay_buffer(replay_buffer, action=None, config=None):
    if action is None:
        states = [transition["state"] for transition in replay_buffer]
    else:
        states = [
            transition["state"]
            for transition in replay_buffer
            if transition["action"] == action
        ]
    if not states:
        if config is None:
            return np.empty((0, len(STATE_FEATURE_NAMES)), dtype=np.float64)
        return np.empty((0, len(state_feature_names(config))), dtype=np.float64)
    return np.vstack(states)


def build_q_policy_bundle(
    q_models,
    q_model_is_fitted,
    replay_buffer,
    config,
    episode_results=None,
    action_results=None,
    lagrangian_lambdas_by_budget=None,
):
    if lagrangian_lambdas_by_budget is None:
        default_budget = (
            config["acquisition_budget"],
            config["retrain_budget"],
            config["evaluation_budget"],
        )
        lagrangian_lambdas_by_budget = {
            default_budget: initial_lagrangian_lambdas(config)
        }
    return {
        "bundle_version": Q_POLICY_BUNDLE_VERSION,
        "q_models": q_models,
        "q_model_is_fitted": q_model_is_fitted,
        "replay_buffer": replay_buffer,
        "episode_results": episode_results,
        "action_results": action_results,
        "state_feature_names": state_feature_names(config),
        "q_actions": Q_ACTIONS,
        "config": config,
        "lagrangian_lambdas_by_budget": lagrangian_lambdas_by_budget,
    }


def save_q_policy(
    q_models,
    q_model_is_fitted,
    replay_buffer,
    config,
    episode_results=None,
    action_results=None,
    lagrangian_lambdas_by_budget=None,
):
    artifact_path = Path(config["artifact_path"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = build_q_policy_bundle(
        q_models=q_models,
        q_model_is_fitted=q_model_is_fitted,
        replay_buffer=replay_buffer,
        config=config,
        episode_results=episode_results,
        action_results=action_results,
        lagrangian_lambdas_by_budget=lagrangian_lambdas_by_budget,
    )
    dump(bundle, artifact_path)
    print(f"Saved fitted-Q policy to {artifact_path.resolve()}")
    return artifact_path


def load_q_policy(artifact_path):
    artifact_path = Path(artifact_path)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"No fitted-Q artifact at {artifact_path.resolve()}."
        )
    bundle = load(artifact_path)
    if bundle.get("bundle_version") != Q_POLICY_BUNDLE_VERSION:
        raise ValueError(
            f"Unsupported bundle version {bundle.get('bundle_version')}; "
            f"expected {Q_POLICY_BUNDLE_VERSION}."
        )
    return bundle


def finalize_and_score_terminal(
    config,
    make_model,
    model,
    model_version,
    training_features,
    training_targets,
    pending_row_indices,
    pool_features,
    pool_targets,
    experiment,
):
    finalized = finalize_episode_model(
        config,
        make_model,
        model,
        model_version,
        training_features,
        training_targets,
        pending_row_indices,
        pool_features,
        pool_targets,
    )
    terminal_score = terminal_hidden_validation_score(
        finalized["model"], experiment, config
    )
    return {
        **finalized,
        "terminal_score": terminal_score,
    }


# %% Continuous-Q episode
def run_continuous_q_episode(
    q_models,
    q_model_is_fitted,
    experiment,
    episode_seed,
    epsilon,
    collect_transitions,
    record_step_diagnostics=None,
    lagrangian_lambdas=None,
):
    config = experiment["config"]
    budget_combination = (
        config["acquisition_budget"],
        config["retrain_budget"],
        config["evaluation_budget"],
    )
    batch_fraction = float(config["batch_fraction"])
    baseline_terminal_scores = experiment.get("baseline_terminal_scores")
    if baseline_terminal_scores is None:
        baseline_terminal_scores = {}
    episode_baseline_terminal_score = None
    episode_excess_over_baseline = None
    if lagrangian_lambdas is None:
        lagrangian_lambdas = initial_lagrangian_lambdas(config)
    if record_step_diagnostics is None:
        record_step_diagnostics = config.get(
            "record_step_diagnostics", not collect_transitions
        )
    episode_random = np.random.default_rng(episode_seed)
    batch_size = config["batch_size"]

    training_features = experiment["initial_train_features"].copy()
    training_targets = experiment["initial_train_targets"].copy()
    available_row_indices = np.arange(
        config["acquisition_pool_rows"], dtype=np.int64
    )
    pending_row_indices = np.array([], dtype=np.int64)
    action_history = []
    episode_transitions = []

    remaining_acquisition_budget = config["acquisition_budget"]
    remaining_retrain_budget = config["retrain_budget"]
    remaining_evaluation_budget = config["evaluation_budget"]

    make_model = experiment["make_model"]
    model = make_model()
    model.fit(training_features, training_targets)
    model_version = 0
    initial_evaluation_score = policy_evaluation_score(model, experiment, config)
    evaluation_score_by_model_version = {model_version: initial_evaluation_score}
    latest_evaluation_score = initial_evaluation_score
    last_eval_score_change = 0.0
    evaluation_incremental_change_history = []
    previous_evaluation_score_for_incremental = initial_evaluation_score
    evaluation_return_windows = evaluation_return_windows_from_config(config)
    rolling_eval_score_change_features = compute_rolling_eval_score_change_features(
        evaluation_incremental_change_history,
        latest_evaluation_score,
        initial_evaluation_score,
        evaluation_return_windows,
    )
    cumulative_actual_reward = 0.0
    final_pending_retrain_rows = 0

    pool_features = experiment["acquisition_pool_features"]
    pool_targets = experiment["acquisition_pool_targets"]
    pool_uncertainty_cache = None
    pool_uncertainty_cache_model_version = None

    initial_state = make_state(
        pending_row_count=0,
        remaining_acquisition_budget=remaining_acquisition_budget,
        remaining_retrain_budget=remaining_retrain_budget,
        remaining_evaluation_budget=remaining_evaluation_budget,
        latest_evaluation_score=latest_evaluation_score,
        last_eval_score_change=last_eval_score_change,
        rolling_eval_score_change_features=rolling_eval_score_change_features,
        config=config,
    )
    step_diagnostics = []
    if record_step_diagnostics:
        step_diagnostics = [
            {
                "step": 0,
                "action": "Initial",
                **predicted_q_by_action(initial_state, q_models, q_model_is_fitted),
                "introspection_score": introspection_score(model, experiment, config),
                "model_version": model_version,
                "pending_rows": len(pending_row_indices),
                "training_rows": len(training_features),
                "acquisitions_so_far": 0,
                "acquired_row_indices": [],
            }
        ]

    while True:
        state = make_state(
            pending_row_count=len(pending_row_indices),
            remaining_acquisition_budget=remaining_acquisition_budget,
            remaining_retrain_budget=remaining_retrain_budget,
            remaining_evaluation_budget=remaining_evaluation_budget,
            latest_evaluation_score=latest_evaluation_score,
            last_eval_score_change=last_eval_score_change,
            rolling_eval_score_change_features=rolling_eval_score_change_features,
            config=config,
        )
        feasible_action_list = feasible_actions(
            available_row_count=len(available_row_indices),
            pending_row_count=len(pending_row_indices),
            remaining_acquisition_budget=remaining_acquisition_budget,
            remaining_retrain_budget=remaining_retrain_budget,
            remaining_evaluation_budget=remaining_evaluation_budget,
            model_needs_evaluation=(
                model_version not in evaluation_score_by_model_version
            ),
            config=config,
        )
        if not feasible_action_list:
            terminal_result = finalize_and_score_terminal(
                config,
                make_model,
                model,
                model_version,
                training_features,
                training_targets,
                pending_row_indices,
                pool_features,
                pool_targets,
                experiment,
            )
            model = terminal_result["model"]
            model_version = terminal_result["model_version"]
            training_features = terminal_result["training_features"]
            training_targets = terminal_result["training_targets"]
            pending_row_indices = terminal_result["pending_row_indices"]
            final_pending_retrain_rows = terminal_result["final_pending_retrain_rows"]
            terminal_score = terminal_result["terminal_score"]
            break

        selected_action = choose_action(
            q_models,
            q_model_is_fitted,
            state,
            feasible_action_list,
            episode_random,
            epsilon,
            use_exploration=collect_transitions,
        )
        q_at_decision = predicted_q_by_action(state, q_models, q_model_is_fitted)
        q_value_at_action = q_at_decision[f"q_{selected_action.lower()}"]

        acquired_row_count = None
        selected_row_indices = np.array([], dtype=np.int64)
        mean_row_utility = None
        evaluation_score = None
        reward = lagrangian_step_reward(
            config["step_penalty"], selected_action, lagrangian_lambdas, config
        )

        if selected_action == "Acquire":
            pool_uncertainty_cache, pool_uncertainty_cache_model_version = (
                ensure_pool_uncertainty_cache(
                    model,
                    pool_features,
                    config["acquisition_pool_rows"],
                    model_version,
                    pool_uncertainty_cache,
                    pool_uncertainty_cache_model_version,
                )
            )
            (
                selected_row_indices,
                mean_row_utility,
                available_row_indices,
            ) = select_top_k_pool_rows(
                model,
                pool_features,
                available_row_indices,
                batch_size,
                config,
                pool_uncertainty_cache=pool_uncertainty_cache,
            )
            acquired_row_count = len(selected_row_indices)
            pending_row_indices = np.concatenate(
                [pending_row_indices, selected_row_indices]
            )
            assert_acquired_rows_not_in_pool(
                available_row_indices, pending_row_indices
            )
            remaining_acquisition_budget -= 1

        elif selected_action == "Retrain":
            training_features = np.vstack(
                [training_features, pool_features[pending_row_indices]]
            )
            training_targets = pd.concat(
                [training_targets, pool_targets.iloc[pending_row_indices]],
                ignore_index=True,
            )
            pending_row_indices = np.array([], dtype=np.int64)
            remaining_retrain_budget -= 1
            model = make_model()
            model.fit(training_features, training_targets)
            model_version += 1
            pool_uncertainty_cache = None
            pool_uncertainty_cache_model_version = None

        elif selected_action == "Evaluate":
            if model_version not in evaluation_score_by_model_version:
                evaluation_score_by_model_version[model_version] = (
                    policy_evaluation_score(model, experiment, config)
                )
            evaluation_score = evaluation_score_by_model_version[model_version]
            incremental_change = (
                evaluation_score - previous_evaluation_score_for_incremental
            )
            evaluation_incremental_change_history.append(incremental_change)
            previous_evaluation_score_for_incremental = evaluation_score
            last_eval_score_change = eval_score_change_from_start(
                evaluation_score, initial_evaluation_score
            )
            rolling_eval_score_change_features = (
                compute_rolling_eval_score_change_features(
                    evaluation_incremental_change_history,
                    evaluation_score,
                    initial_evaluation_score,
                    evaluation_return_windows,
                )
            )
            latest_evaluation_score = evaluation_score
            remaining_evaluation_budget -= 1

        elif selected_action == "Stop":
            terminal_result = finalize_and_score_terminal(
                config,
                make_model,
                model,
                model_version,
                training_features,
                training_targets,
                pending_row_indices,
                pool_features,
                pool_targets,
                experiment,
            )
            model = terminal_result["model"]
            model_version = terminal_result["model_version"]
            training_features = terminal_result["training_features"]
            training_targets = terminal_result["training_targets"]
            pending_row_indices = terminal_result["pending_row_indices"]
            final_pending_retrain_rows = terminal_result["final_pending_retrain_rows"]
            terminal_score = terminal_result["terminal_score"]
            reward, terminal_components = compute_terminal_step_reward(
                terminal_score,
                "Stop",
                experiment,
                config,
                lagrangian_lambdas,
                baseline_terminal_scores,
            )
            episode_baseline_terminal_score = terminal_components[
                "baseline_terminal_score"
            ]
            episode_excess_over_baseline = terminal_components[
                "excess_over_baseline"
            ]
            cumulative_actual_reward += reward
            step_number = len(action_history) + 1
            step_record = {
                "action_number": step_number,
                "action": selected_action,
                "reward": reward,
                "q_value_at_action": q_value_at_action,
                **q_at_decision,
                "cumulative_actual_reward": cumulative_actual_reward,
                "acquisition_cost_so_far": config["acquisition_budget"]
                - remaining_acquisition_budget,
                "retrain_cost_so_far": config["retrain_budget"]
                - remaining_retrain_budget,
                "evaluation_cost_so_far": config["evaluation_budget"]
                - remaining_evaluation_budget,
                "acquired_row_count": None,
                "mean_row_utility": None,
                "evaluation_score": evaluation_score,
                "evaluation_score_change_from_start": last_eval_score_change,
                "rolling_eval_score_change_features": list(
                    rolling_eval_score_change_features
                ),
                "introspection_score": (
                    introspection_score(model, experiment, config)
                    if record_step_diagnostics
                    else np.nan
                ),
                "training_rows": len(training_features),
                "pending_rows": len(pending_row_indices),
                "remaining_pool_rows": len(available_row_indices),
                "model_version": model_version,
                "remaining_acquisition_budget": remaining_acquisition_budget,
                "remaining_retrain_budget": remaining_retrain_budget,
                "remaining_evaluation_budget": remaining_evaluation_budget,
                "estimated_discounted_return": reward,
            }
            if record_step_diagnostics:
                if final_pending_retrain_rows > 0:
                    append_final_retrain_diagnostic(
                        step_diagnostics,
                        step_number + 1,
                        model,
                        experiment,
                        config,
                        q_models,
                        q_model_is_fitted,
                        model_version,
                        len(training_features),
                        remaining_acquisition_budget,
                        remaining_retrain_budget,
                        remaining_evaluation_budget,
                        latest_evaluation_score,
                        last_eval_score_change,
                        rolling_eval_score_change_features,
                    )
                step_diagnostics.append(
                    {
                        "step": step_number,
                        "action": selected_action,
                        **q_at_decision,
                        "introspection_score": step_record["introspection_score"],
                        "model_version": step_record["model_version"],
                        "pending_rows": step_record["pending_rows"],
                        "training_rows": step_record["training_rows"],
                        "acquisitions_so_far": step_record["acquisition_cost_so_far"],
                        "acquired_row_indices": [],
                    }
                )
            action_history.append(step_record)
            if collect_transitions:
                episode_transitions.append(
                    {
                        "budget_combination": budget_combination,
                        "batch_fraction": batch_fraction,
                        "state": state,
                        "action": selected_action,
                        "reward": reward,
                        "next_state": make_state(
                            pending_row_count=len(pending_row_indices),
                            remaining_acquisition_budget=remaining_acquisition_budget,
                            remaining_retrain_budget=remaining_retrain_budget,
                            remaining_evaluation_budget=remaining_evaluation_budget,
                            latest_evaluation_score=latest_evaluation_score,
                            last_eval_score_change=last_eval_score_change,
                            rolling_eval_score_change_features=rolling_eval_score_change_features,
                            config=config,
                        ),
                        "next_feasible_actions": [],
                        "done": True,
                    }
                )
            break

        cumulative_actual_reward += reward
        step_number = len(action_history) + 1
        step_record = {
            "action_number": step_number,
            "action": selected_action,
            "reward": reward,
            "q_value_at_action": q_value_at_action,
            **q_at_decision,
            "cumulative_actual_reward": cumulative_actual_reward,
            "acquisition_cost_so_far": config["acquisition_budget"] - remaining_acquisition_budget,
            "retrain_cost_so_far": config["retrain_budget"] - remaining_retrain_budget,
            "evaluation_cost_so_far": config["evaluation_budget"] - remaining_evaluation_budget,
            "acquired_row_count": acquired_row_count,
            "mean_row_utility": mean_row_utility,
            "evaluation_score": evaluation_score,
            "evaluation_score_change_from_start": last_eval_score_change,
            "rolling_eval_score_change_features": list(
                rolling_eval_score_change_features
            ),
            "introspection_score": (
                introspection_score(model, experiment, config)
                if record_step_diagnostics
                else np.nan
            ),
            "training_rows": len(training_features),
            "pending_rows": len(pending_row_indices),
            "remaining_pool_rows": len(available_row_indices),
            "model_version": model_version,
            "remaining_acquisition_budget": remaining_acquisition_budget,
            "remaining_retrain_budget": remaining_retrain_budget,
            "remaining_evaluation_budget": remaining_evaluation_budget,
        }
        if record_step_diagnostics:
            step_diagnostics.append(
                {
                    "step": step_number,
                    "action": selected_action,
                    **q_at_decision,
                    "introspection_score": step_record["introspection_score"],
                    "model_version": step_record["model_version"],
                    "pending_rows": step_record["pending_rows"],
                    "training_rows": step_record["training_rows"],
                    "acquisitions_so_far": step_record["acquisition_cost_so_far"],
                    "acquired_row_indices": selected_row_indices.tolist(),
                }
            )

        next_state = make_state(
            pending_row_count=len(pending_row_indices),
            remaining_acquisition_budget=remaining_acquisition_budget,
            remaining_retrain_budget=remaining_retrain_budget,
            remaining_evaluation_budget=remaining_evaluation_budget,
            latest_evaluation_score=latest_evaluation_score,
            last_eval_score_change=last_eval_score_change,
            rolling_eval_score_change_features=rolling_eval_score_change_features,
            config=config,
        )
        next_feasible_action_list = feasible_actions(
            available_row_count=len(available_row_indices),
            pending_row_count=len(pending_row_indices),
            remaining_acquisition_budget=remaining_acquisition_budget,
            remaining_retrain_budget=remaining_retrain_budget,
            remaining_evaluation_budget=remaining_evaluation_budget,
            model_needs_evaluation=(
                model_version not in evaluation_score_by_model_version
            ),
            config=config,
        )

        if not next_feasible_action_list:
            terminal_result = finalize_and_score_terminal(
                config,
                make_model,
                model,
                model_version,
                training_features,
                training_targets,
                pending_row_indices,
                pool_features,
                pool_targets,
                experiment,
            )
            model = terminal_result["model"]
            model_version = terminal_result["model_version"]
            training_features = terminal_result["training_features"]
            training_targets = terminal_result["training_targets"]
            pending_row_indices = terminal_result["pending_row_indices"]
            final_pending_retrain_rows = terminal_result["final_pending_retrain_rows"]
            terminal_score = terminal_result["terminal_score"]
            reward, terminal_components = compute_terminal_step_reward(
                terminal_score,
                selected_action,
                experiment,
                config,
                lagrangian_lambdas,
                baseline_terminal_scores,
            )
            episode_baseline_terminal_score = terminal_components[
                "baseline_terminal_score"
            ]
            episode_excess_over_baseline = terminal_components[
                "excess_over_baseline"
            ]
            step_record["reward"] = reward
            step_record["estimated_discounted_return"] = reward
            cumulative_actual_reward += reward
            step_record["cumulative_actual_reward"] = cumulative_actual_reward
            action_history.append(step_record)
            if collect_transitions:
                episode_transitions.append(
                    {
                        "budget_combination": budget_combination,
                        "batch_fraction": batch_fraction,
                        "state": state,
                        "action": selected_action,
                        "reward": reward,
                        "next_state": next_state,
                        "next_feasible_actions": [],
                        "done": True,
                    }
                )
            break

        estimated_discounted_return = reward + config["discount_factor"] * max_predicted_q(
            next_state,
            next_feasible_action_list,
            q_models,
            q_model_is_fitted,
        )
        step_record["estimated_discounted_return"] = estimated_discounted_return
        action_history.append(step_record)

        if collect_transitions:
            episode_transitions.append(
                {
                    "budget_combination": budget_combination,
                    "batch_fraction": batch_fraction,
                    "state": state,
                    "action": selected_action,
                    "reward": reward,
                    "next_state": next_state,
                    "next_feasible_actions": next_feasible_action_list,
                    "done": False,
                }
            )

    acquisition_cost = config["acquisition_budget"] - remaining_acquisition_budget
    retrain_cost = config["retrain_budget"] - remaining_retrain_budget
    evaluation_cost = config["evaluation_budget"] - remaining_evaluation_budget

    episode_result = {
        "model": model,
        "terminal_score": terminal_score,
        "baseline_terminal_score": episode_baseline_terminal_score,
        "excess_over_baseline": episode_excess_over_baseline,
        "relative_terminal_reward_enabled": relative_terminal_reward_settings(
            config
        )["enabled"],
        "initial_evaluation_score": initial_evaluation_score,
        "latest_evaluation_score": latest_evaluation_score,
        "last_evaluation_score_change_from_start": last_eval_score_change,
        "rolling_eval_score_change_features": list(
            rolling_eval_score_change_features
        ),
        "rolling_evaluation_score_change_sum": (
            rolling_eval_score_change_features[0]
            if rolling_eval_score_change_features
            else 0.0
        ),
        "target_metric": metric_from_config(config, "target"),
        "policy_metric": metric_from_config(config, "policy"),
        "acquisition_cost": acquisition_cost,
        "retrain_cost": retrain_cost,
        "evaluation_cost": evaluation_cost,
        "training_rows_at_end": len(training_features),
        "pending_rows_at_end": len(pending_row_indices),
        "final_pending_retrain_rows": final_pending_retrain_rows,
        "lagrangian_lambdas": dict(lagrangian_lambdas),
        "action_history": action_history,
        "step_diagnostics": step_diagnostics,
    }
    return episode_result, episode_transitions


# %% Training and evaluation
def trim_replay_buffer(replay_buffer, config):
    max_transitions = config.get("replay_buffer_max_transitions")
    if max_transitions is None or len(replay_buffer) <= max_transitions:
        return replay_buffer

    indexed_transitions_by_condition = {}
    for index, transition in enumerate(replay_buffer):
        indexed_transitions_by_condition.setdefault(
            replay_condition_key(transition), []
        ).append((index, transition))

    conditions_by_recency = sorted(
        indexed_transitions_by_condition,
        key=lambda condition: indexed_transitions_by_condition[condition][-1][0],
        reverse=True,
    )
    transitions_per_condition, extra_transition_count = divmod(
        int(max_transitions), len(conditions_by_recency)
    )
    retained_indexed_transitions = []
    for condition_index, condition in enumerate(conditions_by_recency):
        condition_limit = transitions_per_condition + (
            condition_index < extra_transition_count
        )
        if condition_limit > 0:
            retained_indexed_transitions.extend(
                indexed_transitions_by_condition[condition][-condition_limit:]
            )

    retained_indexed_transitions.sort(key=lambda item: item[0])
    return [transition for _, transition in retained_indexed_transitions]


def training_budget_combinations(config):
    ranges = config["training_budget_ranges"]
    held_out = {
        tuple(combination)
        for combination in config.get("held_out_budget_combinations", [])
    }
    combinations = []
    for acquisition_budget in range(
        ranges["acquisition_budget"][0], ranges["acquisition_budget"][1] + 1
    ):
        for retrain_budget in range(
            ranges["retrain_budget"][0], ranges["retrain_budget"][1] + 1
        ):
            for evaluation_budget in range(
                ranges["evaluation_budget"][0],
                ranges["evaluation_budget"][1] + 1,
            ):
                values = (
                    acquisition_budget,
                    retrain_budget,
                    evaluation_budget,
                )
                if values not in held_out:
                    combinations.append(
                        {
                            "acquisition_budget": acquisition_budget,
                            "retrain_budget": retrain_budget,
                            "evaluation_budget": evaluation_budget,
                        }
                    )
    if not combinations:
        raise ValueError("No training budget combinations remain after holdout")
    return combinations


def training_batch_fraction_values(config):
    return [float(value) for value in config["training_batch_fractions"]]


def build_blocked_training_schedule(
    items, episodes_per_block, number_of_episodes, rng
):
    if episodes_per_block < 1:
        raise ValueError("episodes_per_block must be >= 1")
    if not items:
        raise ValueError("build_blocked_training_schedule requires at least one item")
    schedule = []
    while len(schedule) < number_of_episodes:
        for item_index in rng.permutation(len(items)):
            schedule.extend([items[item_index]] * episodes_per_block)
    return schedule[:number_of_episodes]


def replay_condition_key(transition):
    return (
        transition["budget_combination"],
        round(float(transition["batch_fraction"]), 12),
    )


def train_fitted_q_policy(experiment):
    config = experiment["config"]
    q_models = make_q_models(config)
    q_model_is_fitted = make_q_model_is_fitted()
    replay_buffer = []
    episode_records = []
    all_action_records = []
    q_fit_every_n_episodes = int(config.get("q_fit_every_n_episodes", 1))
    if q_fit_every_n_episodes < 1:
        raise ValueError("q_fit_every_n_episodes must be >= 1")
    budget_combinations = training_budget_combinations(config)
    batch_fraction_values = training_batch_fraction_values(config)
    budget_random = np.random.default_rng(
        config["random_seed"] + config.get("training_budget_seed_offset", 30_000)
    )
    batch_random = np.random.default_rng(
        config["random_seed"] + config.get("training_batch_seed_offset", 40_000)
    )
    lagrangian_lambdas_by_budget = {
        (
            budgets["acquisition_budget"],
            budgets["retrain_budget"],
            budgets["evaluation_budget"],
        ): initial_lagrangian_lambdas(config)
        for budgets in budget_combinations
    }
    episodes_per_budget_block = int(config.get("episodes_per_budget_block", 1))
    if episodes_per_budget_block < 1:
        raise ValueError("episodes_per_budget_block must be >= 1")
    episodes_per_batch_block = int(
        config.get("episodes_per_batch_block", episodes_per_budget_block)
    )
    if episodes_per_batch_block < 1:
        raise ValueError("episodes_per_batch_block must be >= 1")

    budget_schedule = build_blocked_training_schedule(
        budget_combinations,
        episodes_per_budget_block,
        config["number_of_training_episodes"],
        budget_random,
    )
    batch_schedule = build_blocked_training_schedule(
        batch_fraction_values,
        episodes_per_batch_block,
        config["number_of_training_episodes"],
        batch_random,
    )

    baseline_terminal_scores = {}
    if relative_terminal_reward_settings(config)["enabled"]:
        experiment["baseline_terminal_scores"] = baseline_terminal_scores

    for episode_number in tqdm(
        range(config["number_of_training_episodes"]),
        desc="Training fitted-Q policy",
    ):
        epsilon = epsilon_for_episode(episode_number, config)
        episode_budgets = budget_schedule[episode_number]
        episode_batch_fraction = batch_schedule[episode_number]
        budget_combination = (
            episode_budgets["acquisition_budget"],
            episode_budgets["retrain_budget"],
            episode_budgets["evaluation_budget"],
        )
        lagrangian_lambdas = lagrangian_lambdas_by_budget[budget_combination]
        episode_config = {
            **config,
            **episode_budgets,
            "batch_fraction": episode_batch_fraction,
            "batch_size": batch_size_for_fraction(
                episode_batch_fraction, config["total_rows"]
            ),
        }
        episode_experiment = {**experiment, "config": episode_config}
        episode_result, episode_transitions = run_continuous_q_episode(
            q_models=q_models,
            q_model_is_fitted=q_model_is_fitted,
            experiment=episode_experiment,
            episode_seed=config["random_seed"]
            + config["training_episode_seed_offset"]
            + episode_number,
            epsilon=epsilon,
            collect_transitions=True,
            record_step_diagnostics=False,
            lagrangian_lambdas=lagrangian_lambdas,
        )
        replay_buffer.extend(episode_transitions)
        replay_buffer = trim_replay_buffer(replay_buffer, config)
        update_lagrangian_lambdas(
            lagrangian_lambdas,
            episode_config,
            {
                "acquisition_cost": episode_result["acquisition_cost"],
                "retrain_cost": episode_result["retrain_cost"],
                "evaluation_cost": episode_result["evaluation_cost"],
            },
        )
        is_last_episode = episode_number == config["number_of_training_episodes"] - 1
        if (
            (episode_number + 1) % q_fit_every_n_episodes == 0
            or is_last_episode
        ):
            fit_q_models(replay_buffer, q_models, q_model_is_fitted, config)

        episode_records.append(
            {
                "episode": episode_number + 1,
                "epsilon": epsilon,
                **episode_budgets,
                "batch_fraction": episode_batch_fraction,
                "batch_size": episode_config["batch_size"],
                "terminal_score": episode_result["terminal_score"],
                "baseline_terminal_score": episode_result[
                    "baseline_terminal_score"
                ],
                "excess_over_baseline": episode_result["excess_over_baseline"],
                "acquisition_cost": episode_result["acquisition_cost"],
                "retrain_cost": episode_result["retrain_cost"],
                "evaluation_cost": episode_result["evaluation_cost"],
                "number_of_actions": len(episode_result["action_history"]),
                "lambda_acquisition": lagrangian_lambdas["lambda_acquisition"],
                "lambda_retrain": lagrangian_lambdas["lambda_retrain"],
                "lambda_evaluation": lagrangian_lambdas["lambda_evaluation"],
            }
        )
        for action in episode_result["action_history"]:
            all_action_records.append(
                {
                    "episode": episode_number + 1,
                    **episode_budgets,
                    "batch_fraction": episode_batch_fraction,
                    "batch_size": episode_config["batch_size"],
                    **action,
                }
            )

    episode_results = pd.DataFrame(episode_records)
    action_results = pd.DataFrame(all_action_records)
    return (
        q_models,
        q_model_is_fitted,
        replay_buffer,
        episode_results,
        action_results,
        lagrangian_lambdas_by_budget,
    )


def run_final_episode(
    q_models,
    q_model_is_fitted,
    experiment,
    lagrangian_lambdas_by_budget=None,
):
    config = experiment["config"]
    budget_combination = (
        config["acquisition_budget"],
        config["retrain_budget"],
        config["evaluation_budget"],
    )
    lagrangian_lambdas = None
    if lagrangian_lambdas_by_budget is not None:
        lagrangian_lambdas = lagrangian_lambdas_by_budget.get(budget_combination)
    episode_result, _ = run_continuous_q_episode(
        q_models=q_models,
        q_model_is_fitted=q_model_is_fitted,
        experiment=experiment,
        episode_seed=config["random_seed"] + config["final_episode_seed_offset"],
        epsilon=0.0,
        collect_transitions=False,
        record_step_diagnostics=True,
        lagrangian_lambdas=lagrangian_lambdas,
    )
    return episode_result


def run_greedy_episode_with_budgets(
    q_models,
    q_model_is_fitted,
    experiment,
    acquisition_budget,
    retrain_budget,
    evaluation_budget,
    episode_seed=None,
    lagrangian_lambdas_by_budget=None,
):
    rollout_config = {
        **experiment["config"],
        "acquisition_budget": acquisition_budget,
        "retrain_budget": retrain_budget,
        "evaluation_budget": evaluation_budget,
    }
    rollout_experiment = {**experiment, "config": rollout_config}
    config = rollout_experiment["config"]
    if episode_seed is None:
        episode_seed = config["random_seed"] + config["final_episode_seed_offset"]
    budget_combination = (
        acquisition_budget,
        retrain_budget,
        evaluation_budget,
    )
    lagrangian_lambdas = None
    if lagrangian_lambdas_by_budget is not None:
        lagrangian_lambdas = lagrangian_lambdas_by_budget.get(budget_combination)
    episode_result, _ = run_continuous_q_episode(
        q_models=q_models,
        q_model_is_fitted=q_model_is_fitted,
        experiment=rollout_experiment,
        episode_seed=episode_seed,
        epsilon=0.0,
        collect_transitions=False,
        lagrangian_lambdas=lagrangian_lambdas,
    )
    return episode_result


def score_on_test(model, experiment, config=None):
    if config is None:
        config = experiment["config"]
    reporting_metrics = config.get("reporting_metrics")
    if reporting_metrics is None:
        reporting_metrics = [metric_from_config(config, "target")]
    scores = {}
    features, targets = split_features_targets(experiment, "test")
    for metric_name in reporting_metrics:
        validate_metric_name(metric_name)
        scores[f"test_{metric_name}"] = model_metric_on_split(
            model, features, targets, metric_name
        )
    return scores


# %% Reporting
def summarize_training_progress(episode_results, replay_buffer, q_model_is_fitted, config):
    first_window = episode_results.head(10)
    last_window = episode_results.tail(10)
    sample_counts = count_replay_samples_by_action(replay_buffer)
    target_metric = metric_from_config(config, "target")
    terminal_column = "terminal_score"
    return pd.DataFrame(
        [
            {
                "replay_transition_count": len(replay_buffer),
                "fitted_action_model_count": sum(q_model_is_fitted.values()),
                "samples_for_acquire": sample_counts["Acquire"],
                "samples_for_retrain": sample_counts["Retrain"],
                "samples_for_evaluate": sample_counts["Evaluate"],
                "samples_for_stop": sample_counts["Stop"],
                f"mean_terminal_{target_metric}_all_episodes": episode_results[
                    terminal_column
                ].mean(),
                f"mean_terminal_{target_metric}_first_10": first_window[
                    terminal_column
                ].mean(),
                f"mean_terminal_{target_metric}_last_10": last_window[
                    terminal_column
                ].mean(),
                f"best_terminal_{target_metric}": episode_results[terminal_column].max(),
                "mean_actions_per_episode": episode_results["number_of_actions"].mean(),
                "mean_acquisition_cost": episode_results["acquisition_cost"].mean(),
                "mean_retrain_cost": episode_results["retrain_cost"].mean(),
                "mean_evaluation_cost": episode_results["evaluation_cost"].mean(),
            }
        ]
    )


def summarize_training_actions(action_results):
    return (
        action_results.groupby("action")
        .size()
        .reset_index(name="count")
        .assign(fraction=lambda frame: frame["count"] / frame["count"].sum())
        .sort_values("count", ascending=False)
    )


def build_final_episode_tables(final_result, test_scores, config):
    final_action_results = pd.DataFrame(final_result["action_history"])
    final_episode_showcase = final_action_results[
        [
            "action_number",
            "action",
            "acquisition_cost_so_far",
            "retrain_cost_so_far",
            "evaluation_cost_so_far",
            "q_value_at_action",
            "estimated_discounted_return",
            "cumulative_actual_reward",
            "reward",
            "training_rows",
            "pending_rows",
            "acquired_row_count",
            "mean_row_utility",
            "evaluation_score",
        ]
    ].copy()

    first_step = final_action_results.iloc[0]
    final_step = final_action_results.iloc[-1]
    target_metric = metric_from_config(config, "target")
    comparison_row = {
        "perceived_return_at_first_action": first_step["q_value_at_action"],
        "perceived_return_at_final_action": final_step["q_value_at_action"],
        "actual_final_reward": final_step["reward"],
        "perceived_minus_actual_at_end": (
            final_step["q_value_at_action"] - final_step["reward"]
        ),
        f"hidden_validation_{target_metric}": final_result["terminal_score"],
        "total_acquisition_cost": final_result["acquisition_cost"],
        "total_retrain_cost": final_result["retrain_cost"],
        "total_evaluation_cost": final_result["evaluation_cost"],
        "training_rows_at_end": final_result["training_rows_at_end"],
        "pending_rows_at_end": final_result["pending_rows_at_end"],
        "final_pending_retrain_rows": final_result["final_pending_retrain_rows"],
    }
    comparison_row.update(test_scores)
    final_reward_comparison = pd.DataFrame([comparison_row])
    return final_action_results, final_episode_showcase, final_reward_comparison


def print_experiment_report(
    episode_results,
    action_results,
    replay_buffer,
    q_model_is_fitted,
    final_action_results,
    final_episode_showcase,
    final_reward_comparison,
    config,
):
    training_summary = summarize_training_progress(
        episode_results, replay_buffer, q_model_is_fitted, config
    )
    action_summary = summarize_training_actions(action_results)

    target_metric = metric_from_config(config, "target")
    policy_metric = metric_from_config(config, "policy")
    print(f"\nFitted-Q training: {config['number_of_training_episodes']} episodes")
    print(
        f"Target metric: {metric_label(target_metric)} | "
        f"Policy metric: {metric_label(policy_metric)}"
    )
    print("\nTraining progress summary")
    print(training_summary.to_string(index=False))
    print("\nAction mix across all training episodes")
    print(action_summary.to_string(index=False))
    print("\nLast 5 training episodes")
    print(episode_results.tail(5).to_string(index=False))

    print("\n" + "=" * 72)
    print("FINAL GREEDY EPISODE (deployment run after fitted-Q training)")
    print("=" * 72)
    print("\nAction sequence:")
    print(" → ".join(final_action_results["action"]))
    print("\nStep-by-step: costs and perceived vs actual reward")
    print(final_episode_showcase.to_string(index=False))
    summary_row = final_reward_comparison.iloc[0]
    training_rows_at_end = int(summary_row["training_rows_at_end"])
    pending_rows_at_end = int(summary_row["pending_rows_at_end"])
    final_pending_retrain_rows = int(summary_row["final_pending_retrain_rows"])
    print(
        f"\nFinal classifier: trained on {training_rows_at_end} rows "
        f"(initial + policy Retrain actions"
        + (
            f" + {final_pending_retrain_rows} rows in end-of-episode flush"
            if final_pending_retrain_rows > 0
            else ""
        )
        + ")."
    )
    if config.get("final_retrain_pending_rows", False):
        if final_pending_retrain_rows > 0:
            print(
                f"Optional end-of-episode flush: merged {final_pending_retrain_rows} "
                "pending acquired rows and retrained before terminal scoring "
                "(does not spend retrain budget)."
            )
        if pending_rows_at_end > 0:
            print(
                f"Warning: {pending_rows_at_end} acquired rows are still pending — "
                "they are NOT in the final model or terminal score."
            )
        else:
            print(
                "Terminal reward and test scoring use the post-episode model "
                "(including any optional flush)."
            )
    elif pending_rows_at_end > 0:
        print(
            f"Warning: {pending_rows_at_end} acquired rows are still pending — "
            "they are NOT in the final model or terminal score "
            "(set final_retrain_pending_rows=True to merge them at episode end)."
        )
    else:
        print(
            "Terminal reward and test scoring use the model as it stood when "
            "the episode ended."
        )
    print("\nPerceived vs actual reward at termination")
    print(final_reward_comparison.to_string(index=False))
    print("=" * 72)
