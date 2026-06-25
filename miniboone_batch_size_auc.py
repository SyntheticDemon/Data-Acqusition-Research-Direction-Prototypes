# %% Imports
import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.datasets import fetch_openml
from tqdm.auto import tqdm


# %% Editable experiment configuration
RANDOM_SEED = 42
VALIDATION_FRACTION = 0.25
BATCH_SIZES = list(range(100, 5_001, 100))

MODEL_PARAMETERS = {
    "learning_rate": 0.01,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "random_state": RANDOM_SEED,
}


# %% Download and load Adult Income
adult = fetch_openml(
    name="adult",
    version=2,
    as_frame=True,
)

all_features = adult.data.copy()

all_targets = (
    adult.target
    .astype(str)
    .str.strip()
    .str.replace(".", "", regex=False)
    .eq(">50K")
    .astype("int8")
    .rename("income_above_50k")
)

print(f"Rows: {len(all_features):,}")
print(f"Features: {all_features.shape[1]}")
print(f"Positive fraction: {all_targets.mean():.3f}")


# %% Identify numerical and categorical columns
numerical_columns = all_features.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = all_features.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("Numerical columns:", numerical_columns)
print("Categorical columns:", categorical_columns)


# %% Preprocessing
# Ordinal encoding is used because HistGradientBoosting requires numeric input.
# Unknown validation categories are encoded as -1.
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
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    ),
                ]
            ),
            categorical_columns,
        ),
    ],
    remainder="drop",
)


# %% Make one fixed 75% training and 25% validation split
(
    training_features,
    validation_features,
    training_targets,
    validation_targets,
) = train_test_split(
    all_features,
    all_targets,
    test_size=VALIDATION_FRACTION,
    stratify=all_targets,
    random_state=RANDOM_SEED,
)

training_features = training_features.reset_index(drop=True)
training_targets = training_targets.reset_index(drop=True)
validation_features = validation_features.reset_index(drop=True)
validation_targets = validation_targets.reset_index(drop=True)

print(
    f"Training rows: {len(training_features):,}; "
    f"validation rows: {len(validation_features):,}"
)


# %% Fit preprocessing only on the training partition
preprocessor.fit(training_features)

training_features_encoded = preprocessor.transform(training_features)
validation_features_encoded = preprocessor.transform(validation_features)


# %% Create nested stratified training subsets
random_generator = np.random.default_rng(RANDOM_SEED)

positive_training_rows = random_generator.permutation(
    np.flatnonzero(training_targets.to_numpy() == 1)
)

negative_training_rows = random_generator.permutation(
    np.flatnonzero(training_targets.to_numpy() == 0)
)

positive_fraction = training_targets.mean()

training_rows_by_batch_size = {}

for batch_size in BATCH_SIZES:
    positive_batch_rows = int(round(batch_size * positive_fraction))
    positive_batch_rows = min(
        max(positive_batch_rows, 1),
        batch_size - 1,
    )

    negative_batch_rows = batch_size - positive_batch_rows

    training_rows_by_batch_size[batch_size] = np.concatenate(
        [
            positive_training_rows[:positive_batch_rows],
            negative_training_rows[:negative_batch_rows],
        ]
    )


# %% Train one model for each available-label budget
auc_records = []

for batch_size in tqdm(
    BATCH_SIZES,
    desc="Training batch-size models",
):
    batch_rows = training_rows_by_batch_size[batch_size]

    model = HistGradientBoostingClassifier(
        **MODEL_PARAMETERS
    )

    model.fit(
        training_features_encoded[batch_rows],
        training_targets.iloc[batch_rows],
    )

    validation_probabilities = model.predict_proba(
        validation_features_encoded
    )[:, 1]

    auc_records.append(
        {
            "batch_size": batch_size,
            "validation_auc": roc_auc_score(
                validation_targets,
                validation_probabilities,
            ),
            "training_set": (
                f"{min(BATCH_SIZES)} to "
                f"{max(BATCH_SIZES)} rows"
            ),
        }
    )


# %% Full-training baseline
full_training_model = HistGradientBoostingClassifier(
    **MODEL_PARAMETERS
)

full_training_model.fit(
    training_features_encoded,
    training_targets,
)

full_validation_probabilities = (
    full_training_model.predict_proba(
        validation_features_encoded
    )[:, 1]
)

full_validation_auc = roc_auc_score(
    validation_targets,
    full_validation_probabilities,
)

auc_records.append(
    {
        "batch_size": len(training_features),
        "validation_auc": full_validation_auc,
        "training_set": "Full training partition",
    }
)


# %% Results
batch_size_auc_results = pd.DataFrame(auc_records)

print(
    batch_size_auc_results.to_string(
        index=False,
        formatters={
            "validation_auc": "{:.4f}".format
        },
    )
)

batch_size_auc_results


# %% Plot validation AUC by training-set size
auc_figure = px.line(
    batch_size_auc_results[:-1],
    x="batch_size",
    y="validation_auc",
    markers=True,
    # log_x=True,
    color="training_set",
    title="Adult Income validation AUC by training-set size",
    labels={
        "batch_size": "Training rows, log scale",
        "validation_auc": "Validation ROC AUC",
        "training_set": "Training set",
    },
)

auc_figure.update_yaxes(range=[0.5, 1.0])

auc_figure.add_hline(
    y=full_validation_auc,
    line_dash="dash",
    annotation_text=(
        f"Full-training AUC: {full_validation_auc:.4f}"
    ),
)

auc_figure.show()


# %% Inspect strongest measured model
small_batch_results = batch_size_auc_results[
    batch_size_auc_results["training_set"]
    != "Full training partition"
]

best_batch_result = small_batch_results.loc[
    small_batch_results["validation_auc"].idxmax()
]

print(
    f"Best sampled validation AUC: "
    f"{best_batch_result['validation_auc']:.4f} "
    f"with {int(best_batch_result['batch_size']):,} "
    f"training rows"
)
# %%
