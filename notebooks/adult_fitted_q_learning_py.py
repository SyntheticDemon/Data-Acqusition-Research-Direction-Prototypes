# %% [markdown]
# # Adult fitted-Q learning — full analysis
#
# Train a fitted-Q policy, run greedy rollouts (training and custom budgets), and
# interpret the linear Q-models with SHAP.
#
# Implementation lives in `adult_prototype_fitted_q_learning.py`. Edit the
# **configuration** cell below, then run top to bottom. Restart the kernel if you
# change the Python module.

# %%
import importlib
import sys

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import shap
from IPython.display import display

PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import adult_prototype_fitted_q_learning

importlib.reload(adult_prototype_fitted_q_learning)

from adult_prototype_fitted_q_learning import (
    Q_ACTIONS,
    SUPPORTED_METRICS,
    STATE_FEATURE_NAMES,
    metric_label,
    state_feature_names,
    build_experiment,
    build_final_episode_tables,
    load_q_policy,
    plot_episode_step_diagnostics,
    predict_q,
    print_experiment_report,
    run_final_episode,
    run_greedy_episode_with_budgets,
    save_q_policy,
    score_on_test,
    states_from_replay_buffer,
    train_fitted_q_policy,
)

# %%
RANDOM_SEED = 42

config = {
    "random_seed": RANDOM_SEED,
    "total_rows": 48_000,
    # "total_rows": 6_000,
    # "total_rows": 48_00,
    # "total_rows": 3_000,
    "initial_train_fraction": 0.01,
    "acquisition_pool_fraction": 0.84,
    "evaluation_fraction": 0.05,
    "hidden_validation_fraction": 0.05,
    "test_fraction": 0.05,
    "batch_fraction": 0.01,
    "training_batch_fraction_range": [0.001, 0.01],
    "training_batch_fraction_step": 0.001,
    "held_out_batch_fractions": [0.01],
    "default_rollout_batch_fraction": 0.01,
    "training_batch_seed_offset": 40_000,
    "episodes_per_batch_block": 150,
    "acquisition_percentile_low": 0,
    "acquisition_percentile_high": 95,
    "number_of_training_episodes": 15000,
    "training_budget_ranges": {
        "acquisition_budget": [1, 30],
        "retrain_budget": [1, 30],
        "evaluation_budget": [1, 30],
    },
    # Step through each range (default step is 1 = every integer).
    # "training_budget_step": 2,
    # "training_budget_steps": {
    #     "acquisition_budget": 2,
    #     "retrain_budget": 1,
    #     "evaluation_budget": 5,
    # },
    # Or pass explicit value lists instead of range + step:
    # "training_budget_values": {
    #     "acquisition_budget": [1, 5, 10, 20, 30],
    #     "retrain_budget": [1, 4, 8, 12],
    #     "evaluation_budget": [1, 6, 12, 18, 24, 30],
    # },
    "default_rollout_budgets": {
        "acquisition_budget": 8,
        "retrain_budget": 4,
        "evaluation_budget": 6,
    },
    "q_fit_every_n_episodes": 250,
    # Inclusive ranges visited in randomized, sequential budget blocks.
    # Excluded from training and used by the final rollout.
    "held_out_budget_combinations": [[8, 4, 6]],
    "training_budget_seed_offset": 30_000,
    # Train consecutive episodes at one budget before moving to the next.
    "episodes_per_budget_block": 150,
    # Legacy single window still works: "evaluation_return_window": 5,
    "evaluation_return_windows": [5, 10, 15],
    "discount_factor": 0.95,
    "initial_epsilon": 1.0,
    "final_epsilon": 0.1,
    "step_penalty": 0.0,
    "allow_stop_action": True,
    # Nonlinear Q-model; larger leaves reduce overfitting to bootstrapped targets.
    # "q_model": {
    #     "type": "extra_trees_regressor",
    #     "n_estimators": 300,
    #     "min_samples_leaf": 30,
    #     "max_depth": 5,
    #     "random_state": RANDOM_SEED,
    #     "n_jobs": -1,
    # },
    # "q_model": {
    #     "type": "ridge_regression",
    #     # "degree": 2,
    #     "alpha": 1.0,
    # },
    "q_model": {
        "type": "polynomial_ridge",
        "degree": 2,
        "alpha": 1.0,
    },
    "lagrangian_q_learning": {
        "enabled": True,
        "lambda_acquisition": 1,
        "lambda_retrain": 1,
        "lambda_evaluation": 1,
        "lambda_learning_rate_acquisition": 0.25,
        "lambda_learning_rate_retrain": 0.25,
        "lambda_learning_rate_evaluation": 0.25,
    },
    # Terminal reward = hidden-val target_metric - baseline terminal target_metric
    # (same split/metric as episode terminal_score; baseline final model scored
    # with terminal_hidden_validation_score).
    "relative_terminal_reward": {
        "enabled": True,
        "baseline": "acquire_retrain",  # or "acquire_only"
    },
    # Keep transitions from every processed budget combination.
    "replay_buffer_max_transitions": None,
    "min_samples_per_action": 40,
    "training_episode_seed_offset": 10_000,
    "final_episode_seed_offset": 100_000,
    "target_metric": "f1",
    "policy_metric": "f1",
    "introspection_metric": "f1",
    "reporting_metrics": ["f1", "roc_auc"],
    "introspection_eval_set": "hidden_validation",
    "final_retrain_pending_rows": True,  # optional; default False in code
    "artifact_path": PROJECT_ROOT / "artifacts" / "fitted_q_policy.joblib",
    "downstream_model": {
        "type": "hist_gradient_boosting",
        "learning_rate": 0.1,
        "max_iter": 100,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 5,
        "max_depth": 6,
        "random_state": RANDOM_SEED,
    },
    # "downstream_model": {
    #     "type": "logistic_regression",
    #     "max_iter": 1000,
    #     "C": 1.0,
    #     "random_state": RANDOM_SEED,
    # },
    # "downstream_model": {
    #     "type": "extra_trees_classifier",
    #     "n_estimators": 200,
    #     "min_samples_leaf": 10,
    #     "max_depth": 8,
    #     "random_state": RANDOM_SEED,
    #     "n_jobs": -1,
    # },
}

experiment = build_experiment(config)
config = experiment["config"]  # restore derived row counts used by rollouts and plots
state_feature_names = state_feature_names(config)
config

# %% [markdown]
# ## Training

# %%
(
    q_models,
    q_model_is_fitted,
    replay_buffer,
    episode_results,
    action_results,
    lagrangian_lambdas_by_budget,
) = train_fitted_q_policy(experiment)

artifact_path = save_q_policy(
    q_models,
    q_model_is_fitted,
    replay_buffer,
    config,
    episode_results=episode_results,
    action_results=action_results,
    lagrangian_lambdas_by_budget=lagrangian_lambdas_by_budget,
)
artifact_path

# %% [markdown]
# ### Reload a saved Q learner instead of training
#
# After running the imports cell, run the cell below to restore the saved experiment configuration and all policy variables used by the remaining analysis cells. Skip the training cell above.

# %%|
artifact_path = PROJECT_ROOT / "artifacts" / "fitted_q_policy.joblib"
policy_bundle = load_q_policy(artifact_path)

q_models = policy_bundle["q_models"]
q_model_is_fitted = policy_bundle["q_model_is_fitted"]
replay_buffer = policy_bundle["replay_buffer"]
episode_results = policy_bundle["episode_results"]
action_results = policy_bundle["action_results"]
lagrangian_lambdas_by_budget = policy_bundle["lagrangian_lambdas_by_budget"]
config = policy_bundle["config"]
state_feature_names = policy_bundle["state_feature_names"]
experiment = build_experiment(config)
config = experiment["config"]  # restore derived row counts used by rollouts and plots

print(f"Loaded Q learner from {artifact_path.resolve()}")
print(f"Replay transitions: {len(replay_buffer):,}")
print(f"Training episodes: {len(episode_results):,}")

# %%
budget_columns = ["acquisition_budget", "retrain_budget", "evaluation_budget"]
budget_coverage = episode_results.groupby(budget_columns, as_index=False).agg(
    episodes=("episode", "size"), mean_terminal_score=("terminal_score", "mean")
)
display(budget_coverage.sort_values(budget_columns))
print(f"Observed training combinations: {len(budget_coverage):,}")
print(f"Configured held-out combinations: {config['held_out_budget_combinations']}")

px.line(
    episode_results,
    x="episode",
    y="terminal_score",
    markers=True,
    title=f"Terminal hidden-validation {metric_label(config['target_metric'])} over training episodes",
    hover_data=budget_columns,
    labels={"terminal_score": "terminal score", "episode": "episode"},
)

# %% [markdown]
# ## Final greedy episode (held-out budget combination)
#
# The base `(acquisition, retrain, evaluation)` tuple is excluded from training, so this rollout measures generalization to an unseen budget combination.

# %%
final_result = run_final_episode(
    q_models,
    q_model_is_fitted,
    experiment,
    lagrangian_lambdas_by_budget=lagrangian_lambdas_by_budget,
)
test_scores = score_on_test(final_result["model"], experiment, config)
(
    final_action_results,
    final_episode_showcase,
    final_reward_comparison,
) = build_final_episode_tables(final_result, test_scores, config)

print_experiment_report(
    episode_results,
    action_results,
    replay_buffer,
    q_model_is_fitted,
    final_action_results,
    final_episode_showcase,
    final_reward_comparison,
    config,
)

# %%
final_episode_showcase

plot_episode_step_diagnostics(
    final_result["step_diagnostics"], config, experiment=experiment
)

# %% [markdown]
# ## Acquisition batch-fraction scenario
#
# Batch fraction controls rows per `Acquire`; the three budgets control the
# maximum number of times each action can run in this scenario.

# %%
SCENARIO_BATCH_FRACTION = 0.005
SCENARIO_ACQUISITION_BUDGET = 50
SCENARIO_RETRAIN_BUDGET = 20
SCENARIO_EVALUATION_BUDGET = 20

scenario_batch_size = max(1, int(round(SCENARIO_BATCH_FRACTION * config["total_rows"])))
scenario_config = {
    **config,
    "batch_fraction": SCENARIO_BATCH_FRACTION,
    "batch_size": scenario_batch_size,
    "acquisition_budget": SCENARIO_ACQUISITION_BUDGET,
    "retrain_budget": SCENARIO_RETRAIN_BUDGET,
    "evaluation_budget": SCENARIO_EVALUATION_BUDGET,
    # Do not add an unbudgeted fit after the scenario terminates.
    "final_retrain_pending_rows": False,
}
scenario_experiment = {**experiment, "config": scenario_config}

batch_fraction_scenario = run_greedy_episode_with_budgets(
    q_models=q_models,
    q_model_is_fitted=q_model_is_fitted,
    experiment=scenario_experiment,
    acquisition_budget=scenario_config["acquisition_budget"],
    retrain_budget=scenario_config["retrain_budget"],
    evaluation_budget=scenario_config["evaluation_budget"],
    lagrangian_lambdas_by_budget=lagrangian_lambdas_by_budget,
)
assert (
    batch_fraction_scenario["acquisition_cost"] <= scenario_config["acquisition_budget"]
)
assert batch_fraction_scenario["retrain_cost"] <= scenario_config["retrain_budget"]
assert (
    batch_fraction_scenario["evaluation_cost"] <= scenario_config["evaluation_budget"]
)

scenario_actions = pd.DataFrame(batch_fraction_scenario["action_history"])
print("Action sequence: " + " → ".join(scenario_actions["action"]))
print(
    f"Batch fraction={SCENARIO_BATCH_FRACTION:.4f}; "
    f"rows per Acquire={scenario_batch_size}"
)
print(
    f"Budgets: acquire={scenario_config['acquisition_budget']}, "
    f"retrain={scenario_config['retrain_budget']}, "
    f"evaluate={scenario_config['evaluation_budget']}"
)
print(
    f"Hidden validation {metric_label(config['target_metric'])}: "
    f"{batch_fraction_scenario['terminal_score']:.4f}"
)

plot_episode_step_diagnostics(
    batch_fraction_scenario["step_diagnostics"],
    scenario_config,
    experiment=scenario_experiment,
)

# %% [markdown]
# ## Custom-budget rollout
#
# The saved Q-policy is not refitted. The task classifier can still retrain when
# the policy selects `Train`. Budget-size state features distinguish interpolation
# inside the training ranges from extrapolation beyond them.

# %%
CUSTOM_ACQUISITION_BUDGET = 15
CUSTOM_RETRAIN_BUDGET = 5
CUSTOM_EVALUATION_BUDGET = 5

custom_result = run_greedy_episode_with_budgets(
    q_models=q_models,
    q_model_is_fitted=q_model_is_fitted,
    experiment=experiment,
    acquisition_budget=CUSTOM_ACQUISITION_BUDGET,
    retrain_budget=CUSTOM_RETRAIN_BUDGET,
    evaluation_budget=CUSTOM_EVALUATION_BUDGET,
    lagrangian_lambdas_by_budget=lagrangian_lambdas_by_budget,
)
custom_test_scores = score_on_test(custom_result["model"], experiment)
(
    custom_action_results,
    custom_episode_showcase,
    custom_reward_comparison,
) = build_final_episode_tables(custom_result, custom_test_scores, config)

print("Action sequence: " + " → ".join(custom_action_results["action"]))
print(
    f"Budgets: acquire={CUSTOM_ACQUISITION_BUDGET}, "
    f"retrain={CUSTOM_RETRAIN_BUDGET}, evaluate={CUSTOM_EVALUATION_BUDGET}"
)
print(
    f"Hidden validation {metric_label(config['target_metric'])}: "
    f"{custom_result['terminal_score']:.4f}"
)
print(f"Test scores: {custom_test_scores}")

custom_config = {
    **config,
    "acquisition_budget": CUSTOM_ACQUISITION_BUDGET,
    "retrain_budget": CUSTOM_RETRAIN_BUDGET,
    "evaluation_budget": CUSTOM_EVALUATION_BUDGET,
}
plot_episode_step_diagnostics(
    custom_result["step_diagnostics"],
    custom_config,
    experiment=experiment,
)

# %%
custom_cost_long = custom_action_results.melt(
    id_vars=["action_number", "action"],
    value_vars=[
        "acquisition_cost_so_far",
        "retrain_cost_so_far",
        "evaluation_cost_so_far",
    ],
    var_name="cost_type",
    value_name="cost_so_far",
)
custom_cost_long["cost_type"] = custom_cost_long["cost_type"].str.replace(
    "_cost_so_far", ""
)

px.line(
    custom_cost_long,
    x="action_number",
    y="cost_so_far",
    color="cost_type",
    markers=True,
    title="Custom-budget rollout: spend over the episode",
    labels={
        "action_number": "Step",
        "cost_so_far": "Cumulative cost",
        "cost_type": "Budget",
    },
)

# %%
px.line(
    custom_action_results,
    x="action_number",
    y=["q_value_at_action", "cumulative_actual_reward"],
    markers=True,
    title="Custom rollout: perceived Q vs cumulative actual reward",
    labels={
        "action_number": "Step",
        "value": "Value",
        "variable": "Series",
    },
)

# %% [markdown]
# ## SHAP interpretation
#
# Uses the replay buffer and Q-models from training above.

# %%
states_by_action = {
    action: states_from_replay_buffer(replay_buffer, action=action)
    for action in Q_ACTIONS
}

pd.DataFrame(
    [
        {"action": action, "transition_count": len(states_by_action[action])}
        for action in Q_ACTIONS
    ]
)

# %%
SHAP_BACKGROUND_SIZE = 100
SHAP_EXPLAIN_SIZE = 200

rng = np.random.default_rng(config["random_seed"])


def sample_rows(states, max_rows, random_generator):
    if len(states) == 0:
        return states
    if len(states) <= max_rows:
        return states
    row_indices = random_generator.choice(len(states), size=max_rows, replace=False)
    return states[row_indices]


shap_explainers = {}
shap_values_by_action = {}

for action in Q_ACTIONS:
    if not q_model_is_fitted[action]:
        print(f"Skipping {action}: Q-model not fitted yet")
        continue

    action_states = states_by_action[action]
    background_states = sample_rows(action_states, SHAP_BACKGROUND_SIZE, rng)
    explain_states = sample_rows(action_states, SHAP_EXPLAIN_SIZE, rng)

    shap_explainers[action] = shap.Explainer(
        q_models[action].predict, background_states
    )
    shap_values_by_action[action] = shap_explainers[action](explain_states)
    print(f"{action}: explained {len(explain_states)} states")

# %%
importance_rows = []
for action, shap_values in shap_values_by_action.items():
    mean_abs = np.mean(np.abs(shap_values.values), axis=0)
    for feature_name, importance in zip(state_feature_names, mean_abs):
        importance_rows.append(
            {
                "action": action,
                "feature": feature_name,
                "mean_abs_shap": float(importance),
            }
        )

importance_table = pd.DataFrame(importance_rows).sort_values(
    ["action", "mean_abs_shap"], ascending=[True, False]
)
importance_table

# %%
fig = px.bar(
    importance_table,
    x="mean_abs_shap",
    y="feature",
    color="action",
    barmode="group",
    orientation="h",
    title="Mean |SHAP| per state feature (by action Q-model)",
    labels={"mean_abs_shap": "mean |SHAP value|", "feature": "state feature"},
)
fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
fig

# %% [markdown]
# ### Single-state policy view

# %%
example_transition = replay_buffer[len(replay_buffer) // 2]
example_state = example_transition["state"]
example_action_taken = example_transition["action"]

q_predictions = {
    action: predict_q(example_state, action, q_models, q_model_is_fitted)
    for action in Q_ACTIONS
    if q_model_is_fitted[action]
}
best_action = max(q_predictions, key=q_predictions.get)

state_table = pd.DataFrame(
    {"feature": state_feature_names, "value": example_state}
).assign(action_taken=example_action_taken, greedy_best_action=best_action)

q_table = pd.DataFrame(
    [
        {"action": action, "predicted_q": value}
        for action, value in q_predictions.items()
    ]
).sort_values("predicted_q", ascending=False)

display(state_table)
display(q_table)

# %%
comparison_rows = []
for action, explainer in shap_explainers.items():
    shap_values = explainer(example_state.reshape(1, -1))
    for feature_name, shap_value in zip(state_feature_names, shap_values.values[0]):
        comparison_rows.append(
            {
                "action": action,
                "feature": feature_name,
                "shap_value": float(shap_value),
            }
        )

comparison_frame = pd.DataFrame(comparison_rows)

fig = px.bar(
    comparison_frame,
    x="shap_value",
    y="feature",
    color="action",
    barmode="group",
    orientation="h",
    title="Local SHAP contributions for one example state",
    labels={"shap_value": "SHAP value", "feature": "state feature"},
)
fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=550)
fig

# %% [markdown]
# ### Feature sweep: when does the greedy action switch?

# %%
SWEEP_FEATURE = "pending_rows_frac"
SWEEP_VALUES = np.arange(0, 5, 0.25)

feature_index = state_feature_names.index(SWEEP_FEATURE)
sweep_rows = []

for sweep_value in SWEEP_VALUES:
    swept_state = example_state.copy()
    swept_state[feature_index] = sweep_value
    for action in Q_ACTIONS:
        if not q_model_is_fitted[action]:
            continue
        sweep_rows.append(
            {
                SWEEP_FEATURE: sweep_value,
                "action": action,
                "predicted_q": predict_q(
                    swept_state, action, q_models, q_model_is_fitted
                ),
            }
        )

sweep_frame = pd.DataFrame(sweep_rows)
best_by_sweep = sweep_frame.loc[
    sweep_frame.groupby(SWEEP_FEATURE)["predicted_q"].idxmax()
].rename(columns={"action": "greedy_action"})[[SWEEP_FEATURE, "greedy_action"]]

px.line(
    sweep_frame,
    x=SWEEP_FEATURE,
    y="predicted_q",
    color="action",
    markers=True,
    title=f"Predicted Q vs {SWEEP_FEATURE} (other features held fixed)",
)

# %%
best_by_sweep


# %%
