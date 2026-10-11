"""Predict G3 for one explicitly specified feature record."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from .data_loader import load_data
from .preprocessing import clean_data, split_features_target
from .train import run

ROOT = Path(__file__).resolve().parents[1]

def parser():
    p = argparse.ArgumentParser(description="Estimate final Portuguese-course grade G3 (0–20); not a guaranteed result.")
    p.add_argument("--school", choices=["GP","MS"], required=True); p.add_argument("--sex", choices=["F","M"], required=True)
    p.add_argument("--age", type=int, choices=range(15,23), required=True); p.add_argument("--address", choices=["U","R"], required=True)
    p.add_argument("--famsize", choices=["LE3","GT3"], required=True); p.add_argument("--Pstatus", choices=["T","A"], required=True)
    for x in ["Medu","Fedu"]: p.add_argument("--"+x, type=int, choices=range(5), required=True)
    for x, choices in {"Mjob":["teacher","health","services","at_home","other"],"Fjob":["teacher","health","services","at_home","other"],"reason":["home","reputation","course","other"],"guardian":["mother","father","other"]}.items(): p.add_argument("--"+x, choices=choices, required=True)
    for x in ["traveltime","studytime"]: p.add_argument("--"+x, type=int, choices=range(1,5), required=True)
    p.add_argument("--failures", type=int, choices=range(4), required=True)
    for x in ["schoolsup","famsup","paid","activities","nursery","higher","internet","romantic"]: p.add_argument("--"+x, choices=["yes","no"], required=True)
    for x in ["famrel","freetime","goout","Dalc","Walc","health"]: p.add_argument("--"+x, type=int, choices=range(1,6), required=True)
    p.add_argument("--absences", type=int, choices=range(94), required=True)
    for x in ["G1","G2"]: p.add_argument("--"+x, type=int, choices=range(21), required=True)
    return p

def predict(values):
    path = ROOT/"models"/"selected_model.joblib"
    if not path.exists():
        print("No trained model found; running end-to-end training first.")
        run()
    model = joblib.load(path)
    X, _ = split_features_target(clean_data(load_data()))
    record = pd.DataFrame([values]).reindex(columns=X.columns)
    raw = float(model.predict(record)[0])
    return max(0.0, min(20.0, raw))

if __name__ == "__main__":
    args = vars(parser().parse_args())
    value = predict(args)
    print(f"Estimated G3: {value:.2f}/20. This is an estimate, not a guaranteed examination result.")
