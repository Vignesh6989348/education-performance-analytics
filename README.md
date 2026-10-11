# Education Performance Analytics — Student Success Prediction

A reproducible regression project estimating Portuguese-course final grade (**G3**, scale 0–20) from the UCI Student Performance dataset. It demonstrates acquisition, data checks, EDA, pipelines, model comparison, interpretation, and prediction.

Project overview site: [`docs/index.html`](docs/index.html) (configured for GitHub Pages from the `/docs` folder on `main`).

## Dataset and design
- **Source:** [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance), DOI [10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T), CC BY 4.0. Citation: Cortez, P. (2008), *Student Performance*.
- **Subset:** Portuguese course (`student-por.csv`), 649 rows; source has 33 columns including target.
- **Target:** `G3`, final-year grade issued at period 3 (0–20).
- **Prediction time:** after second-period grades are known; `G1` and `G2` are eligible inputs. `G3` is excluded from inputs. This is not an enrolment-time prediction.
- **Data availability:** source includes demographics, family/social factors, absences, and period grades. No explicit attendance-rate, internal-mark, assignment-score, or engagement-telemetry columns. `absences` is a count, not attendance rate; `studytime` is a self-reported category, not tracked engagement.
- **Limitations/responsibility:** historical cases from two Portuguese secondary schools, only 649 in this subject subset; transfer to other contexts is unverified. Correlation is not causation. Student data are sensitive—respect consent, access, retention and de-identification requirements. Estimates should support educators, never automatically determine outcomes.

Raw data is downloaded from UCI on first run; no synthetic observations are used.

## 🌐 Live Project Demo

<p align="center">
  <a href="https://eduperf-g7avvd4w.manus.space/">
    <img src="https://img.shields.io/badge/🚀_Launch-Live_Demo-00C853?style=for-the-badge&logoColor=white" alt="Launch Live Demo"/>
  </a>
</p>

<p align="center">
  <b>Education Performance Analytics — Student Success Prediction</b>
  <br/>
  An interactive web experience for exploring student performance data and estimating final academic grades using machine learning.
</p>

<p align="center">
  <a href="https://eduperf-g7avvd4w.manus.space/">
    <b>🔗 Open Live Website →</b>
  </a>
</p>

### ✨ Explore the Platform

* 📊 **Model Comparison** — Review the performance of the evaluated regression models.
* 🎯 **Student Grade Estimation** — Explore the prediction interface for estimating final course grades.
* 📈 **Data-Driven Insights** — Understand how historical academic data can support student performance analysis.
* 🧠 **Machine Learning Workflow** — Explore the project’s approach to data preparation, model training, and evaluation.

> **Note:** Predictions are estimates based on historical data and should not be treated as guaranteed academic outcomes.

**🔗 Live Demo:** https://eduperf-g7avvd4w.manus.space/
**📦 Source Code:** https://github.com/Vignesh6989348/education-performance-analytics


## Setup
Python 3.9+:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

## Run
From repository root:
```bash
python -m src.train
python -m pytest -q
```
The pipeline uses reproducible 60/20/20 train/validation/test partitions (`random_state=42`), train-only fitted preprocessing, a mean `DummyRegressor` baseline, Linear Regression, Random Forest and Extra Trees. Model selection uses lowest validation MAE (then lower validation RMSE); the untouched test partition is reserved for final reporting. The comparison table reports actual test MAE/RMSE/R² for all candidates.

Example CLI (all features required):
```bash
python -m src.predict --school GP --sex F --age 17 --address U --famsize GT3 --Pstatus T --Medu 4 --Fedu 3 --Mjob teacher --Fjob other --reason course --guardian mother --traveltime 1 --studytime 2 --failures 0 --schoolsup no --famsup yes --paid no --activities yes --nursery yes --higher yes --internet yes --romantic no --famrel 4 --freetime 3 --goout 2 --Dalc 1 --Walc 2 --health 5 --absences 4 --G1 12 --G2 13
```
Output estimate is clipped to the documented score range for display and is not a guaranteed examination result. Run `python -m src.predict --help` for all fields. Guided notebook: `notebooks/education_performance_analytics.ipynb`.

## Outputs
- `reports/figures/`: score distribution (shape of G3), G2/G3 relationship (prior grade vs target), numeric correlation heatmap (association only), studytime/G3 groups, test-set model-error comparison, actual-vs-predicted with identity reference line, and test-set permutation importance (shuffle-based predictive reliance, not causality).
- `reports/metrics.json`, `reports/run_summary.txt`, `models/selected_model.joblib`.

Chart guide: **01** shows how final grades are distributed; **02** shows the observed relationship between second-period and final grades; **03** summarizes pairwise numeric correlations (not causal effects); **04** compares grade distributions across self-reported study-time categories; **05** compares each model's held-out MAE/RMSE (lower is better); **06** compares predictions to actual test grades, with the dashed diagonal representing exact predictions; **07** shows how much held-out R² changes when each input is permuted, a measure of model reliance rather than causal feature effect.

MAE is mean absolute error in score points; RMSE penalizes larger errors; R² can be negative. Permutation importance measures held-out score change under feature shuffling; correlated inputs complicate attribution. Neither metric nor importance establishes causality.
