# Dataset

On first run, `src.data_loader` retrieves `student.zip` from the official UCI repository (`https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip`) and reads `student-por.csv` (semicolon delimiter). This is the Portuguese course subset: 649 cases and 33 source columns, including G1, G2 and G3. UCI documents no missing values. Raw data is not checked into this project; obtain it directly from UCI.

Attribution: Cortez, P. (2008). *Student Performance*. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T. CC BY 4.0.

G3 is the period-3 final grade. This project predicts it after G1/G2 are observed; do not use the design to claim an enrolment-time forecast. UCI documents demographic, social, school, absence and period-grade attributes. It does not provide explicit attendance rates, assignment/internal scores, or engagement telemetry. Absences are a count; studytime is questionnaire-category data.
