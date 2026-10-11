"""Run the full UCI-to-model workflow and save genuine evaluation outputs."""
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from .data_loader import load_data
from .preprocessing import clean_data, split_features_target
from .models import model_specs, make_pipeline
from .evaluate import regression_metrics, compare_predictions

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
MODEL_DIR = ROOT / "models"


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_eda(df):
    FIG.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", palette="deep")
    fig, ax = plt.subplots(figsize=(8, 5)); sns.histplot(df.G3, bins=21, discrete=True, ax=ax, color="#3569a8")
    ax.set(title="Final Grade (G3) Distribution", xlabel="Final grade (0–20)", ylabel="Number of students")
    _save(fig, "01_g3_distribution.png")
    fig, ax = plt.subplots(figsize=(8, 5)); sns.regplot(data=df, x="G2", y="G3", scatter_kws={"alpha": .45}, line_kws={"color":"#b34b42"}, ax=ax)
    ax.set(title="Second-period Grade and Final Grade", xlabel="G2: second-period grade", ylabel="G3: final grade")
    _save(fig, "02_g2_vs_g3.png")
    numeric = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(12, 9)); sns.heatmap(numeric.corr(), cmap="vlag", center=0, ax=ax, square=True)
    ax.set_title("Correlation of Numeric Variables (association, not causation)")
    _save(fig, "03_numeric_correlations.png")
    fig, ax = plt.subplots(figsize=(8, 5)); sns.boxplot(data=df, x="studytime", y="G3", ax=ax)
    ax.set(title="Final Grade by Self-reported Weekly Study-time Category", xlabel="Studytime category (1=lowest, 4=highest)", ylabel="Final grade (0–20)")
    _save(fig, "04_studytime_g3.png")


def run():
    df0 = load_data()
    before = len(df0)
    df = clean_data(df0)
    X, y = split_features_target(df)
    if {"G1", "G2"}.issubset(X.columns) is False:
        raise ValueError("Expected prior-period grades G1 and G2")
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=.20, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=.25, random_state=42)
    make_eda(df)
    fitted, val_predictions, predictions = {}, {}, {}
    for name, estimator in model_specs(X_train).items():
        pipe = make_pipeline(X_train, estimator)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        val_predictions[name] = pipe.predict(X_val)
        predictions[name] = pipe.predict(X_test)
    validation = compare_predictions(y_val, val_predictions)
    comparison = compare_predictions(y_test, predictions)
    winner = validation.iloc[0]["model"]
    best = fitted[winner]
    # Comparison chart follows the metric table order.
    fig, ax = plt.subplots(figsize=(9, 5)); plot = comparison.set_index("model")[["MAE", "RMSE"]]
    plot.plot(kind="bar", ax=ax, color=["#3569a8", "#d58b35"])
    ax.set(title="Held-out Model Error Comparison (lower is better)", ylabel="Error in G3 grade points", xlabel="Model")
    _save(fig, "05_model_comparison.png")
    fig, ax = plt.subplots(figsize=(7, 7)); ax.scatter(y_test, predictions[winner], alpha=.65)
    ax.plot([0,20],[0,20], "--", color="#b34b42"); ax.set(xlim=(0,20), ylim=(0,20), title=f"Actual vs Predicted — {winner}", xlabel="Actual G3", ylabel="Predicted G3")
    _save(fig, "06_actual_vs_predicted.png")
    perm = permutation_importance(best, X_test, y_test, n_repeats=12, random_state=42, scoring="r2", n_jobs=-1)
    imp = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(8, 6)); imp.plot.barh(ax=ax, color="#448a78")
    ax.set(title="Top Permutation Importances on Test Set", xlabel="Mean decrease in R² after shuffling", ylabel="Input feature")
    _save(fig, "07_permutation_importance.png")
    metrics = {"dataset":"UCI Student Performance, Portuguese course", "source":"https://archive.ics.uci.edu/dataset/320/student+performance", "target":"G3", "prediction_timing":"After G2 is available", "rows_source":int(before), "rows_after_duplicate_and_target_check":int(len(df)), "duplicates_removed":int(before-len(df)), "features":list(X.columns), "split":{"train_fraction":.60,"validation_fraction":.20,"test_fraction":.20,"random_state":42,"train_rows":len(X_train),"validation_rows":len(X_val),"test_rows":len(X_test)}, "selection_criterion":"lowest validation-set MAE; ties resolved by lower validation RMSE; test set used only for final reporting", "validation_metrics":validation.to_dict(orient="records"), "models":comparison.to_dict(orient="records"), "selected_model":winner}
    (ROOT/"reports").mkdir(exist_ok=True)
    (ROOT/"reports"/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    MODEL_DIR.mkdir(exist_ok=True); joblib.dump(best, MODEL_DIR/"selected_model.joblib")
    sample = X_test.iloc[[0]]
    estimate = float(best.predict(sample)[0])
    summary = f"Selected model: {winner}\nSelection: lowest validation-set MAE, tie-break lower validation RMSE\nFinal untouched test metrics: {regression_metrics(y_test, predictions[winner])}\nSample prediction: {estimate:.2f}/20 (estimate, not guaranteed result)\nSample inputs:\n{sample.to_string(index=False)}\n"
    (ROOT/"reports"/"run_summary.txt").write_text(summary)
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSelected: {winner}\nExample G3 estimate: {estimate:.2f}/20\nOutputs: reports/ and models/selected_model.joblib")
    return comparison

if __name__ == "__main__":
    run()
