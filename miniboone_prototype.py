# %% Imports
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


# %% Editable experiment configuration
RANDOM_SEED = 42

TOTAL_ROWS = 25_100
INITIAL_TRAIN_ROWS = 100
ACQUISITION_POOL_ROWS = 16_000
EVALUATION_ROWS = 3_000
HIDDEN_VALIDATION_ROWS = 3_000
TEST_ROWS = 3_000

BATCH_SIZE = 50
PREVIEW_SIZE = 20
ACQUISITION_BUDGET = 8
RETRAIN_BUDGET = 4
EVALUATION_BUDGET = 4

NUMBER_OF_TRAINING_EPISODES = 30
Q_LEARNING_RATE = 0.2
DISCOUNT_FACTOR = 0.95
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.1
STEP_PENALTY = 0.0

MODEL_PARAMETERS = {
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "random_state": RANDOM_SEED,
}


# %% Download MiniBooNE and prepare fixed experiment partitions
expected_total_rows = (
    INITIAL_TRAIN_ROWS
    + ACQUISITION_POOL_ROWS
    + EVALUATION_ROWS
    + HIDDEN_VALIDATION_ROWS
    + TEST_ROWS
)
assert TOTAL_ROWS == expected_total_rows
assert ACQUISITION_POOL_ROWS % BATCH_SIZE == 0
assert PREVIEW_SIZE <= BATCH_SIZE

dataset_url = (
    "https://archive.ics.uci.edu/static/public/199/"
    "miniboone+particle+identification.zip"
)
dataset_cache = (
    Path.home()
    / ".cache"
    / "data-acquisition-research"
    / "miniboone_particle_identification.zip"
)
dataset_cache.parent.mkdir(parents=True, exist_ok=True)
if not dataset_cache.exists():
    print(f"Downloading MiniBooNE to {dataset_cache}")
    urlretrieve(dataset_url, dataset_cache)

with ZipFile(dataset_cache) as dataset_archive:
    with dataset_archive.open("MiniBooNE_PID.txt") as dataset_file:
        signal_rows, background_rows = map(
            int,
            dataset_file.readline().decode("utf-8").split(),
        )
    with dataset_archive.open("MiniBooNE_PID.txt") as dataset_file:
        all_features = pd.read_csv(
            dataset_file,
            sep=r"\s+",
            header=None,
            skiprows=1,
            dtype="float32",
        )

all_targets = pd.Series(
    np.concatenate(
        [
            np.ones(signal_rows, dtype="int8"),
            np.zeros(background_rows, dtype="int8"),
        ]
    ),
    name="is_signal",
)
if len(all_features) != len(all_targets):
    raise ValueError(
        "MiniBooNE header counts do not match the number of feature rows."
    )
if all_features.shape[1] != 50:
    raise ValueError(
        f"Expected 50 MiniBooNE features, received {all_features.shape[1]}."
    )

all_row_ids = np.arange(len(all_features))
selected_row_ids, _ = train_test_split(
    all_row_ids,
    train_size=TOTAL_ROWS,
    stratify=all_targets,
    random_state=RANDOM_SEED,
)
remaining_row_ids, test_row_ids = train_test_split(
    selected_row_ids,
    test_size=TEST_ROWS,
    stratify=all_targets.iloc[selected_row_ids],
    random_state=RANDOM_SEED + 1,
)
remaining_row_ids, hidden_validation_row_ids = train_test_split(
    remaining_row_ids,
    test_size=HIDDEN_VALIDATION_ROWS,
    stratify=all_targets.iloc[remaining_row_ids],
    random_state=RANDOM_SEED + 2,
)
remaining_row_ids, evaluation_row_ids = train_test_split(
    remaining_row_ids,
    test_size=EVALUATION_ROWS,
    stratify=all_targets.iloc[remaining_row_ids],
    random_state=RANDOM_SEED + 3,
)
initial_train_row_ids, acquisition_pool_row_ids = train_test_split(
    remaining_row_ids,
    test_size=ACQUISITION_POOL_ROWS,
    stratify=all_targets.iloc[remaining_row_ids],
    random_state=RANDOM_SEED + 4,
)

split_row_ids = [
    initial_train_row_ids,
    acquisition_pool_row_ids,
    evaluation_row_ids,
    hidden_validation_row_ids,
    test_row_ids,
]
assert [len(row_ids) for row_ids in split_row_ids] == [
    INITIAL_TRAIN_ROWS,
    ACQUISITION_POOL_ROWS,
    EVALUATION_ROWS,
    HIDDEN_VALIDATION_ROWS,
    TEST_ROWS,
]
assert len(np.unique(np.concatenate(split_row_ids))) == TOTAL_ROWS

initial_train_features = all_features.iloc[initial_train_row_ids].reset_index(
    drop=True
)
initial_train_targets = all_targets.iloc[initial_train_row_ids].reset_index(
    drop=True
)
acquisition_pool_features = all_features.iloc[
    acquisition_pool_row_ids
].reset_index(drop=True)
acquisition_pool_targets = all_targets.iloc[
    acquisition_pool_row_ids
].reset_index(drop=True)
evaluation_features = all_features.iloc[evaluation_row_ids].reset_index(
    drop=True
)
evaluation_targets = all_targets.iloc[evaluation_row_ids].reset_index(
    drop=True
)
hidden_validation_features = all_features.iloc[
    hidden_validation_row_ids
].reset_index(drop=True)
hidden_validation_targets = all_targets.iloc[
    hidden_validation_row_ids
].reset_index(drop=True)
test_features = all_features.iloc[test_row_ids].reset_index(drop=True)
test_targets = all_targets.iloc[test_row_ids].reset_index(drop=True)


# %% Q-learning episode
def run_episode(q_values, episode_seed, epsilon, update_q_values):
    episode_random = np.random.default_rng(episode_seed)
    shuffled_pool_rows = episode_random.permutation(ACQUISITION_POOL_ROWS)

    available_batches = []
    for batch_number, batch_start in enumerate(
        range(0, ACQUISITION_POOL_ROWS, BATCH_SIZE)
    ):
        batch_rows = shuffled_pool_rows[
            batch_start : batch_start + BATCH_SIZE
        ]
        preview_rows = episode_random.choice(
            batch_rows,
            size=PREVIEW_SIZE,
            replace=False,
        )
        available_batches.append(
            {
                "batch_number": batch_number,
                "rows": batch_rows,
                "preview_rows": preview_rows,
            }
        )

    training_features = initial_train_features.copy()
    training_targets = initial_train_targets.copy()
    pending_batches = []
    action_history = []

    remaining_acquisition_budget = ACQUISITION_BUDGET
    remaining_retrain_budget = RETRAIN_BUDGET
    remaining_evaluation_budget = EVALUATION_BUDGET

    model = HistGradientBoostingClassifier(**MODEL_PARAMETERS)
    model.fit(training_features, training_targets)
    model_version = 0
    latest_evaluation_auc = None
    acquisitions_since_retrain = 0
    retrainings_since_evaluation = 0

    while True:
        batch_scores = []
        if available_batches:
            for batch in available_batches:
                preview_probabilities = model.predict_proba(
                    acquisition_pool_features.iloc[batch["preview_rows"]]
                )[:, 1]
                preview_probabilities = np.clip(
                    preview_probabilities,
                    1e-12,
                    1.0 - 1e-12,
                )
                batch_scores.append(
                    float(
                        np.mean(
                            -preview_probabilities
                            * np.log(preview_probabilities)
                            - (1.0 - preview_probabilities)
                            * np.log(1.0 - preview_probabilities)
                        )
                    )
                )

        mean_batch_utility = float(np.mean(batch_scores)) if batch_scores else 0.0
        maximum_batch_utility = float(np.max(batch_scores)) if batch_scores else 0.0
        state = (
            len(training_features) // BATCH_SIZE,
            len(pending_batches),
            len(available_batches),
            remaining_acquisition_budget,
            remaining_retrain_budget,
            remaining_evaluation_budget,
            -1
            if latest_evaluation_auc is None
            else int(latest_evaluation_auc * 20),
            acquisitions_since_retrain,
            retrainings_since_evaluation,
            int(mean_batch_utility * 10),
            int(maximum_batch_utility * 10),
        )

        feasible_actions = ["Stop"]
        if (
            available_batches
            and remaining_acquisition_budget > 0
            and remaining_retrain_budget > 0
        ):
            feasible_actions.append("Acquire")
        if pending_batches and remaining_retrain_budget > 0:
            feasible_actions.append("Retrain")
        if remaining_evaluation_budget > 0:
            feasible_actions.append("Evaluate")

        for action in feasible_actions:
            q_values.setdefault((state, action), 0.0)

        if update_q_values and episode_random.random() < epsilon:
            selected_action = episode_random.choice(feasible_actions)
        else:
            best_q_value = max(
                q_values[(state, action)] for action in feasible_actions
            )
            best_actions = [
                action
                for action in feasible_actions
                if q_values[(state, action)] == best_q_value
            ]
            selected_action = episode_random.choice(best_actions)

        selected_batch_number = None
        selected_batch_score = None
        evaluation_auc = None
        reward = STEP_PENALTY
        terminal = selected_action == "Stop"

        if selected_action == "Acquire":
            selected_batch_index = int(np.argmax(batch_scores))
            selected_batch = available_batches.pop(selected_batch_index)
            selected_batch_number = selected_batch["batch_number"]
            selected_batch_score = batch_scores[selected_batch_index]
            pending_batches.append(selected_batch)
            remaining_acquisition_budget -= 1
            acquisitions_since_retrain += 1

        elif selected_action == "Retrain":
            pending_rows = np.concatenate(
                [batch["rows"] for batch in pending_batches]
            )
            training_features = pd.concat(
                [
                    training_features,
                    acquisition_pool_features.iloc[pending_rows],
                ],
                ignore_index=True,
            )
            training_targets = pd.concat(
                [
                    training_targets,
                    acquisition_pool_targets.iloc[pending_rows],
                ],
                ignore_index=True,
            )
            pending_batches = []
            remaining_retrain_budget -= 1
            acquisitions_since_retrain = 0
            retrainings_since_evaluation += 1
            model = HistGradientBoostingClassifier(**MODEL_PARAMETERS)
            model.fit(training_features, training_targets)
            model_version += 1

        elif selected_action == "Evaluate":
            evaluation_auc = roc_auc_score(
                evaluation_targets,
                model.predict_proba(evaluation_features)[:, 1],
            )
            latest_evaluation_auc = evaluation_auc
            remaining_evaluation_budget -= 1
            retrainings_since_evaluation = 0

        else:
            reward = roc_auc_score(
                hidden_validation_targets,
                model.predict_proba(hidden_validation_features)[:, 1],
            )

        action_history.append(
            {
                "action_number": len(action_history) + 1,
                "action": selected_action,
                "reward": reward,
                "batch_number": selected_batch_number,
                "batch_score": selected_batch_score,
                "evaluation_auc": evaluation_auc,
                "training_rows": len(training_features),
                "pending_batches": len(pending_batches),
                "remaining_batches": len(available_batches),
                "model_version": model_version,
                "remaining_acquisition_budget": remaining_acquisition_budget,
                "remaining_retrain_budget": remaining_retrain_budget,
                "remaining_evaluation_budget": remaining_evaluation_budget,
            }
        )

        if terminal:
            if update_q_values:
                old_q_value = q_values[(state, selected_action)]
                q_values[(state, selected_action)] = old_q_value + (
                    Q_LEARNING_RATE * (reward - old_q_value)
                )
            terminal_auc = reward
            break

        next_batch_scores = []
        if available_batches:
            for batch in available_batches:
                preview_probabilities = model.predict_proba(
                    acquisition_pool_features.iloc[batch["preview_rows"]]
                )[:, 1]
                preview_probabilities = np.clip(
                    preview_probabilities,
                    1e-12,
                    1.0 - 1e-12,
                )
                next_batch_scores.append(
                    float(
                        np.mean(
                            -preview_probabilities
                            * np.log(preview_probabilities)
                            - (1.0 - preview_probabilities)
                            * np.log(1.0 - preview_probabilities)
                        )
                    )
                )

        next_mean_utility = (
            float(np.mean(next_batch_scores)) if next_batch_scores else 0.0
        )
        next_maximum_utility = (
            float(np.max(next_batch_scores)) if next_batch_scores else 0.0
        )
        next_state = (
            len(training_features) // BATCH_SIZE,
            len(pending_batches),
            len(available_batches),
            remaining_acquisition_budget,
            remaining_retrain_budget,
            remaining_evaluation_budget,
            -1
            if latest_evaluation_auc is None
            else int(latest_evaluation_auc * 20),
            acquisitions_since_retrain,
            retrainings_since_evaluation,
            int(next_mean_utility * 10),
            int(next_maximum_utility * 10),
        )

        next_feasible_actions = ["Stop"]
        if (
            available_batches
            and remaining_acquisition_budget > 0
            and remaining_retrain_budget > 0
        ):
            next_feasible_actions.append("Acquire")
        if pending_batches and remaining_retrain_budget > 0:
            next_feasible_actions.append("Retrain")
        if remaining_evaluation_budget > 0:
            next_feasible_actions.append("Evaluate")
        for action in next_feasible_actions:
            q_values.setdefault((next_state, action), 0.0)

        if update_q_values:
            old_q_value = q_values[(state, selected_action)]
            next_q_value = max(
                q_values[(next_state, action)]
                for action in next_feasible_actions
            )
            q_values[(state, selected_action)] = old_q_value + (
                Q_LEARNING_RATE
                * (
                    reward
                    + DISCOUNT_FACTOR * next_q_value
                    - old_q_value
                )
            )

    acquisition_cost = (
        ACQUISITION_BUDGET - remaining_acquisition_budget
    )
    retrain_cost = RETRAIN_BUDGET - remaining_retrain_budget
    evaluation_cost = EVALUATION_BUDGET - remaining_evaluation_budget
    assert acquisition_cost <= ACQUISITION_BUDGET
    assert retrain_cost <= RETRAIN_BUDGET
    assert evaluation_cost <= EVALUATION_BUDGET
    assert all(
        "hidden_validation_auc" not in action
        and "test_auc" not in action
        for action in action_history
    )

    return {
        "model": model,
        "terminal_auc": terminal_auc,
        "acquisition_cost": acquisition_cost,
        "retrain_cost": retrain_cost,
        "evaluation_cost": evaluation_cost,
        "action_history": action_history,
    }


# %% Train the hard-budget Q-learning policy
q_values = {}
episode_records = []
all_action_records = []

for episode_number in tqdm(
    range(NUMBER_OF_TRAINING_EPISODES),
    desc="Training Q-learning policy",
):
    if NUMBER_OF_TRAINING_EPISODES == 1:
        epsilon = FINAL_EPSILON
    else:
        epsilon = INITIAL_EPSILON + (
            (FINAL_EPSILON - INITIAL_EPSILON)
            * episode_number
            / (NUMBER_OF_TRAINING_EPISODES - 1)
        )
    episode_result = run_episode(
        q_values=q_values,
        episode_seed=RANDOM_SEED + 10_000 + episode_number,
        epsilon=epsilon,
        update_q_values=True,
    )
    episode_records.append(
        {
            "episode": episode_number + 1,
            "epsilon": epsilon,
            "terminal_auc": episode_result["terminal_auc"],
            "acquisition_cost": episode_result["acquisition_cost"],
            "retrain_cost": episode_result["retrain_cost"],
            "evaluation_cost": episode_result["evaluation_cost"],
            "number_of_actions": len(episode_result["action_history"]),
        }
    )
    for action in episode_result["action_history"]:
        all_action_records.append(
            {
                "episode": episode_number + 1,
                **action,
            }
        )

episode_results = pd.DataFrame(episode_records)
action_results = pd.DataFrame(all_action_records)
assert len(episode_results) == NUMBER_OF_TRAINING_EPISODES
assert (episode_results["acquisition_cost"] <= ACQUISITION_BUDGET).all()
assert (episode_results["retrain_cost"] <= RETRAIN_BUDGET).all()
assert (episode_results["evaluation_cost"] <= EVALUATION_BUDGET).all()


# %% Evaluate the learned greedy policy once on the held-out test set
final_result = run_episode(
    q_values=q_values,
    episode_seed=RANDOM_SEED + 100_000,
    epsilon=0.0,
    update_q_values=False,
)
final_test_probabilities = final_result["model"].predict_proba(
    test_features
)[:, 1]
final_test_auc = roc_auc_score(
    test_targets,
    final_test_probabilities,
)
final_test_accuracy = accuracy_score(
    test_targets,
    final_test_probabilities >= 0.5,
)
final_summary = pd.DataFrame(
    [
        {
            "hidden_validation_auc": final_result["terminal_auc"],
            "test_auc": final_test_auc,
            "test_accuracy": final_test_accuracy,
            "acquisition_cost": final_result["acquisition_cost"],
            "retrain_cost": final_result["retrain_cost"],
            "evaluation_cost": final_result["evaluation_cost"],
            "number_of_actions": len(final_result["action_history"]),
        }
    ]
)
final_action_results = pd.DataFrame(final_result["action_history"])

print("\nTraining episodes")
print(episode_results.to_string(index=False))
print("\nFinal greedy-policy actions")
print(final_action_results.to_string(index=False))
print("\nFinal held-out result")
print(final_summary.to_string(index=False))


# %% Visualize policy learning and action decisions
terminal_utility_figure = px.line(
    episode_results,
    x="episode",
    y="terminal_auc",
    markers=True,
    title="Terminal hidden-validation AUROC by training episode",
)
terminal_utility_figure.show()

action_count_results = (
    action_results.groupby(["episode", "action"])
    .size()
    .reset_index(name="count")
)
action_count_figure = px.bar(
    action_count_results,
    x="episode",
    y="count",
    color="action",
    title="Actions selected in each training episode",
)
action_count_figure.show()

cost_results = episode_results.melt(
    id_vars="episode",
    value_vars=[
        "acquisition_cost",
        "retrain_cost",
        "evaluation_cost",
    ],
    var_name="cost_type",
    value_name="cost",
)
cost_figure = px.line(
    cost_results,
    x="episode",
    y="cost",
    color="cost_type",
    markers=True,
    title="Resource use by training episode",
)
cost_figure.show()

acquisition_results = action_results[
    action_results["action"] == "Acquire"
].copy()
acquisition_results["acquisition_number"] = acquisition_results.groupby(
    "episode"
).cumcount() + 1
acquisition_figure = px.scatter(
    acquisition_results,
    x="episode",
    y="acquisition_number",
    color="batch_score",
    hover_data=[
        "batch_number",
        "model_version",
        "pending_batches",
    ],
    title="Acquisition decisions during Q-learning",
    labels={
        "acquisition_number": "Acquisition within episode",
        "batch_score": "Selected batch entropy",
    },
)
acquisition_figure.update_yaxes(tickmode="linear", dtick=1)
acquisition_figure.show()

