from pathlib import Path
import json

import pandas as pd
import numpy as np
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)


def main():

    # 1. Locate your Task 3 output
    root = Path(__file__).resolve().parent

    input_file = (
        root / "results" / "task3_delay_recovery"
        / "delay_recovery_features.csv"
    )

    output_folder = root / "results" / "task4_delay_recovery"
    output_folder.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file)

    df["FlightDate"] = pd.to_datetime(
        df["FlightDate"], errors="coerce"
    )

    if df["FlightDate"].isna().any():
        raise ValueError("Invalid flight dates found.")

    # Keep a reference to the row in the Task 3 CSV.
    # This is metadata, not a predictor.
    df["Task3_Row"] = np.arange(len(df))

    # 2. Define your features and target
    numeric_features = [
        "DepDelay",
        "Distance",
        "CRSElapsedTime"
    ]

    categorical_features = [
        "Reporting_Airline",
        "Origin",
        "Dest",
        "DayOfWeek",
        "Scheduled_Departure_Hour",
        "Scheduled_Arrival_Hour"
    ]

    feature_columns = numeric_features + categorical_features
    target = "Delay_Recovery_Status"

    if not df[target].isin([0, 1, 2]).all():
        raise ValueError("The target must contain only 0, 1 and 2.")

    if not pd.to_numeric(
        df["DepDelay"], errors="coerce"
    ).ge(15).all():
        raise ValueError("Only flights with DepDelay >= 15 are eligible.")

    # 3. Separate development data from the final test data
    months = df["FlightDate"].dt.to_period("M").astype(str)

    if set(months) != {"2025-01", "2025-02", "2025-12"}:
        raise ValueError("Expected January, February and December 2025.")

    train_df = (
        df.loc[months.isin(["2025-01", "2025-02"])]
        .sort_values("FlightDate")
        .reset_index(drop=True)
    )

    test_df = (
        df.loc[months.eq("2025-12")]
        .sort_values("FlightDate")
        .reset_index(drop=True)
    )

    X_train = train_df[feature_columns]
    y_train = train_df[target]

    X_test = test_df[feature_columns]
    y_test = test_df[target]

    print("Training/development rows:", len(X_train), flush=True)
    print("December test rows:", len(X_test), flush=True)

    print("\nTraining class percentages:")
    print(
        (y_train.value_counts(normalize=True).sort_index() * 100)
        .round(2)
    )

    # 4. Create chronological cross-validation folds
    # Split dates so flights on the same date stay together.
    # Leave seven days between training and validation.
    unique_dates = np.sort(train_df["FlightDate"].unique())

    date_splitter = TimeSeriesSplit(n_splits=3, gap=7)

    folds = []
    fold_details = []

    for fold_number, (train_dates, validation_dates) in enumerate(
        date_splitter.split(unique_dates), start=1
    ):
        train_indices = np.flatnonzero(
            train_df["FlightDate"].isin(unique_dates[train_dates])
        )

        validation_indices = np.flatnonzero(
            train_df["FlightDate"].isin(
                unique_dates[validation_dates]
            )
        )

        if (
            set(y_train.iloc[train_indices]) != {0, 1, 2}
            or set(y_train.iloc[validation_indices]) != {0, 1, 2}
        ):
            raise ValueError("Each fold must contain all three classes.")

        folds.append((train_indices, validation_indices))

        fold_details.append({
            "Fold": fold_number,
            "Train_start": unique_dates[train_dates[0]],
            "Train_end": unique_dates[train_dates[-1]],
            "Validation_start": unique_dates[validation_dates[0]],
            "Validation_end": unique_dates[validation_dates[-1]],
            "Train_rows": len(train_indices),
            "Validation_rows": len(validation_indices)
        })

    pd.DataFrame(fold_details).to_csv(
        output_folder / "cross_validation_splits.csv",
        index=False
    )

    # 5. Build preprocessing
    # A new preprocessor is created for every model.
    # GridSearchCV fits it only on each fold's training data.
    def make_preprocessor():

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ])

        return ColumnTransformer([
            ("numeric", numeric_pipeline, numeric_features),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ])

    # 6. Define three algorithms and small tuning grids
    model_settings = {
        "LogisticRegression": (
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ),
            {"model__C": [0.1, 1.0]}
        ),

        "DecisionTree": (
            DecisionTreeClassifier(
                min_samples_leaf=30,
                class_weight="balanced",
                random_state=42
            ),
            {"model__max_depth": [8, 16]}
        ),

        "RandomForest": (
            RandomForestClassifier(
                n_estimators=80,
                min_samples_leaf=10,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=2
            ),
            {"model__max_depth": [12, 20]}
        )
    }

    fitted_models = {}
    validation_results = []

    # 7. Tune models using January and February only
    for name, (classifier, parameters) in model_settings.items():

        print(f"\nTraining and tuning {name}...", flush=True)

        pipeline = Pipeline([
            ("preprocessing", make_preprocessor()),
            ("model", classifier)
        ])

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameters,
            scoring="f1_macro",
            cv=folds,
            refit=True,
            n_jobs=1,
            error_score="raise"
        )

        search.fit(X_train, y_train)

        fitted_models[name] = search.best_estimator_

        validation_results.append({
            "Model": name,
            "CV_Macro_F1": search.best_score_,
            "Best_parameters": json.dumps(search.best_params_)
        })

        pd.DataFrame(search.cv_results_).to_csv(
            output_folder / f"{name}_tuning_results.csv",
            index=False
        )

        print("Best parameters:", search.best_params_, flush=True)
        print(
            f"Validation macro F1: {search.best_score_:.4f}",
            flush=True
        )

    # 8. Select the model BEFORE evaluating December
    validation_summary = pd.DataFrame(validation_results)

    best_row = validation_summary["CV_Macro_F1"].idxmax()
    selected_name = validation_summary.loc[best_row, "Model"]

    validation_summary.to_csv(
        output_folder / "validation_model_comparison.csv",
        index=False
    )

    print("\nSelected using validation:", selected_name, flush=True)

    (output_folder / "selected_model_name.txt").write_text(
        selected_name, encoding="utf-8"
    )

    joblib.dump(
        fitted_models[selected_name],
        output_folder / "selected_model.joblib"
    )

    # 9. Add a majority-class baseline
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    fitted_models["DummyBaseline"] = baseline

    # 10. Evaluate all fixed models on December
    test_results = []

    predictions = test_df[
        ["Task3_Row", "FlightDate", target]
    ].copy()

    class_names = [
        "Recovered",
        "Partially Recovered",
        "Not Recovered or Worsened"
    ]

    for name, model in fitted_models.items():

        predicted = model.predict(X_test)
        predictions[f"{name}_Prediction"] = predicted

        test_results.append({
            "Model": name,
            "Selected_by_validation": name == selected_name,
            "Accuracy": accuracy_score(y_test, predicted),
            "Balanced_accuracy": balanced_accuracy_score(
                y_test, predicted
            ),
            "Macro_F1": f1_score(
                y_test, predicted, average="macro"
            ),
            "Weighted_F1": f1_score(
                y_test, predicted, average="weighted"
            )
        })

        report = classification_report(
            y_test,
            predicted,
            labels=[0, 1, 2],
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )

        pd.DataFrame(report).transpose().to_csv(
            output_folder / f"{name}_classification_report.csv"
        )

        display = ConfusionMatrixDisplay.from_predictions(
            y_test,
            predicted,
            labels=[0, 1, 2],
            display_labels=[0, 1, 2],
            cmap="Blues",
            values_format="d"
        )

        display.ax_.set_title(f"{name}: December test")
        display.figure_.savefig(
            output_folder / f"{name}_confusion_matrix.png",
            bbox_inches="tight",
            dpi=150
        )

        plt.close(display.figure_)

    # 11. Save final results
    results = pd.DataFrame(test_results)

    results.to_csv(
        output_folder / "test_model_comparison.csv",
        index=False
    )

    predictions.to_csv(
        output_folder / "december_predictions.csv",
        index=False
    )

    print("\nDecember test results:")
    print(results.round(4).to_string(index=False))

    print("\nTask 4 completed.")
    print("Results saved to:", output_folder)


if __name__ == "__main__":
    main()