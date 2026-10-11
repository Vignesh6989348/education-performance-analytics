"""Download and validate the public UCI Student Performance dataset."""
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import pandas as pd

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
REQUIRED = {"G1", "G2", "G3", "absences", "studytime"}
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "data" / "raw" / "student-por.csv"


def load_data(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    """Load cached CSV or fetch it from UCI; fail loudly on unexpected schema."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(URL, timeout=30) as response:
            archive = ZipFile(BytesIO(response.read()))
        member = next((n for n in archive.namelist() if n.endswith("student-por.csv")), None)
        if member is None:
            raise RuntimeError("UCI archive did not contain student-por.csv")
        path.write_bytes(archive.read(member))
    df = pd.read_csv(path, sep=";")
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Dataset contains no rows")
    return df
