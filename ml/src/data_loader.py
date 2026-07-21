from pathlib import Path
import pandas as pd

COLUMNS = (
    ["engine_id", "cycle",
     "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

def load_train_data(dataset="FD001"):
    base_path = Path(__file__).resolve().parent.parent / "data" / "raw"
    file_path = base_path / f"train_{dataset}.txt"

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )

    return df