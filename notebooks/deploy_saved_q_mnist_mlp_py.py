# %% [markdown]
# # Deploy saved Adult Q-policy on MNIST (MLP downstream)
#
# Loads `artifacts/fitted_q_policy.joblib` **without retraining the Q-models** and
# runs a greedy rollout on MNIST with the same state features, budgets, and
# uncertainty acquisition mechanics as the Adult prototype.
#
# **Important:** the Q-models were trained on Adult Income transitions. State
# vectors have the same shape and normalization, but Q-values are not expected
# to transfer meaningfully to MNIST. Treat this as a plumbing / stress test, or
# train a dedicated MNIST Q-policy first for real results.
#
# Edit the **configuration** cell below, then run top to bottom. Restart the
# kernel after editing `adult_prototype_fitted_q_learning.py` or
# `mnist_mlp_active_learning.py`.

# %%
import importlib
import sys
from pathlib import Path

import pandas as pd
from IPython.display import display

PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import adult_prototype_fitted_q_learning
import mnist_mlp_active_learning

importlib.reload(adult_prototype_fitted_q_learning)
importlib.reload(mnist_mlp_active_learning)

from adult_prototype_fitted_q_learning import (
    DEFAULT_Q_POLICY_ARTIFACT_PATH,
    baseline_policy_curves,
    load_q_policy,
    metric_from_config,
    metric_label,
    plot_episode_step_diagnostics,
    run_greedy_episode_with_budgets,
    score_on_test,
    state_feature_names,
)
from mnist_mlp_active_learning import (
    build_mnist_experiment,
    default_mnist_deploy_config,
    merge_q_policy_rollout_config,
)

# %% [markdown]
# ## Configuration
#
# Values below match the saved Adult artifact held-out rollout
# (`acquire=8`, `retrain=4`, `evaluate=6`, `batch=0.01`, windows `[5, 10, 15]`).

# %%
RANDOM_SEED = 42

ARTIFACT_PATH = PROJECT_ROOT / DEFAULT_Q_POLICY_ARTIFACT_PATH

MNIST_DATA = {
    "total_rows": 1_000,
    "use_full_mnist_test": True,
    "labels_per_class": 10,
    "initial_train_fraction": 0.08,
    "acquisition_pool_fraction": 0.78,
    "evaluation_fraction": 0.05,
    "hidden_validation_fraction": 0.05,
    "test_fraction": 0.04,
}

ROLLOUT_BUDGETS = {
    "acquisition_budget": 30,
    "retrain_budget": 30,
    "evaluation_budget": 30,
}

ROLLOUT_BATCH_FRACTION = 0.005

EVALUATION_RETURN_WINDOWS = [5, 10, 15]

ACQUISITION_PERCENTILES = {
    "acquisition_percentile_low": 0,
    "acquisition_percentile_high": 95,
}

MLP_DOWNSTREAM_MODEL = {
    "hidden_layer_sizes": [256, 128],
    "max_iter": 30,
    "alpha": 1e-4,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "learning_rate_init": 1e-3,
}

METRICS = {
    "target_metric": "accuracy",
    "policy_metric": "accuracy",
    "introspection_metric": "accuracy",
    "introspection_eval_set": "hidden_validation",
    "reporting_metrics": ["accuracy"],
}

EPISODE_OPTIONS = {
    "final_retrain_pending_rows": True,
    "allow_stop_action": True,
    "step_penalty": 0.0,
}

EPISODE_SEED_OFFSET = 100_000

DIAGNOSTICS_INTROSPECTION_EVAL_SET = "hidden_validation"

# %%
def config_overview(config, title):
    rows = []
    for key in sorted(config):
        value = config[key]
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                rows.append(
                    {
                        "section": title,
                        "key": f"{key}.{sub_key}",
                        "value": sub_value,
                    }
                )
        else:
            rows.append({"section": title, "key": key, "value": value})
    return rows


bundle = load_q_policy(ARTIFACT_PATH)
q_models = bundle["q_models"]
q_model_is_fitted = bundle["q_model_is_fitted"]
training_config = bundle["config"]
lagrangian_lambdas_by_budget = bundle.get("lagrangian_lambdas_by_budget")
saved_state_features = bundle["state_feature_names"]

mnist_config = {
    **default_mnist_deploy_config(RANDOM_SEED),
    **MNIST_DATA,
    **ACQUISITION_PERCENTILES,
    **METRICS,
    **EPISODE_OPTIONS,
    "downstream_model": {
        **MLP_DOWNSTREAM_MODEL,
        "random_state": RANDOM_SEED,
    },
}
rollout_config = merge_q_policy_rollout_config(mnist_config, training_config)
rollout_config.update(ROLLOUT_BUDGETS)
rollout_config["batch_fraction"] = float(ROLLOUT_BATCH_FRACTION)
rollout_config["batch_size"] = max(
    1, int(round(ROLLOUT_BATCH_FRACTION * rollout_config["total_rows"]))
)
rollout_config["evaluation_return_windows"] = list(EVALUATION_RETURN_WINDOWS)

experiment = build_mnist_experiment(rollout_config)
config = experiment["config"]

expected_state_features = state_feature_names(config)
if expected_state_features != saved_state_features:
    raise ValueError(
        "MNIST rollout config does not match saved Q state features.\n"
        f"Saved: {saved_state_features}\n"
        f"Rollout: {expected_state_features}\n"
        "Set EVALUATION_RETURN_WINDOWS to match the artifact "
        f"(saved: {training_config.get('evaluation_return_windows')})."
    )

overview_rows = []
overview_rows.extend(config_overview(MNIST_DATA, "mnist_data"))
overview_rows.extend(config_overview(ACQUISITION_PERCENTILES, "acquisition"))
overview_rows.extend(config_overview(MLP_DOWNSTREAM_MODEL, "mlp"))
overview_rows.extend(config_overview(METRICS, "metrics"))
overview_rows.extend(
    config_overview(
        {
            "acquisition_budget": config["acquisition_budget"],
            "retrain_budget": config["retrain_budget"],
            "evaluation_budget": config["evaluation_budget"],
            "batch_fraction": config["batch_fraction"],
            "batch_size": config["batch_size"],
            "evaluation_return_windows": config.get("evaluation_return_windows"),
            "episode_seed": RANDOM_SEED + EPISODE_SEED_OFFSET,
        },
        "resolved_rollout",
    )
)
overview_rows.extend(
    config_overview(
        {
            "artifact_path": str(ARTIFACT_PATH),
            "saved_state_feature_count": len(saved_state_features),
            "fitted_q_actions": [
                action for action, fitted in q_model_is_fitted.items() if fitted
            ],
        },
        "artifact",
    )
)
config_overview_frame = pd.DataFrame(overview_rows)
display(
    config_overview_frame.pivot(index="key", columns="section", values="value")
)

print(f"Loaded artifact: {ARTIFACT_PATH}")
print(f"State features ({len(expected_state_features)}): match saved bundle")
print(
    "Resolved rollout: "
    f"acquire={config['acquisition_budget']}, "
    f"retrain={config['retrain_budget']}, "
    f"evaluate={config['evaluation_budget']}, "
    f"batch_size={config['batch_size']}"
)

# %% [markdown]
# ## Greedy rollout (saved Q-policy, MNIST + MLP)

# %%
episode_result = run_greedy_episode_with_budgets(
    q_models=q_models,
    q_model_is_fitted=q_model_is_fitted,
    experiment=experiment,
    acquisition_budget=config["acquisition_budget"],
    retrain_budget=config["retrain_budget"],
    evaluation_budget=config["evaluation_budget"],
    episode_seed=RANDOM_SEED + EPISODE_SEED_OFFSET,
    lagrangian_lambdas_by_budget=lagrangian_lambdas_by_budget,
)

action_history = pd.DataFrame(episode_result["action_history"])
display(action_history)

target_metric = metric_from_config(config, "target")
print(
    f"Terminal {config['introspection_eval_set'].replace('_', ' ')} "
    f"{metric_label(target_metric)}: {episode_result['terminal_score']:.4f}"
)
print(f"Training rows at end: {episode_result['training_rows_at_end']:,}")

test_scores = score_on_test(episode_result["model"], experiment, config)
display(pd.DataFrame([test_scores]))

# %% [markdown]
# ## Step diagnostics vs fixed baselines (same budgets / batch)

# %%
diagnostics_config = {
    "acquisition_budget": config["acquisition_budget"],
    "retrain_budget": config["retrain_budget"],
    "evaluation_budget": config["evaluation_budget"],
    "batch_fraction": config["batch_fraction"],
    "batch_size": config["batch_size"],
    "introspection_eval_set": DIAGNOSTICS_INTROSPECTION_EVAL_SET,
}

figure = plot_episode_step_diagnostics(
    episode_result["step_diagnostics"],
    diagnostics_config,
    experiment=experiment,
)
figure.update_layout(
    title=(
        "MNIST MLP rollout with saved Adult Q-policy "
        f"(acquire={config['acquisition_budget']}, "
        f"retrain={config['retrain_budget']}, "
        f"evaluate={config['evaluation_budget']}, "
        f"batch={config['batch_fraction']})"
    ),
    height=900,
)
figure.show()

# %% [markdown]
# ## Baseline terminal scores (same condition)

# %%
baseline_curves = baseline_policy_curves(experiment, config)
for baseline_name, baseline_curve in baseline_curves.items():
    terminal_score = float(baseline_curve["introspection_score"].iloc[-1])
    print(f"{baseline_name}: terminal {metric_label(target_metric)} = {terminal_score:.4f}")

print(
    f"Saved Q-policy test {metric_label(target_metric)}: "
    f"{test_scores[f'test_{target_metric}']:.4f}"
)

    # %%
