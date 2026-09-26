"""Extract and scale features from the dataset."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    from .feature_extraction import extract_features, iter_audio_files
except ImportError:
    from feature_extraction import extract_features, iter_audio_files


ROOT = Path(__file__).parents[1]


def preprocess(dataset_dir: str | Path = ROOT / "dataset", model_path: str | Path = ROOT / "models" / "scaler.pkl") -> int:
    rows = []
    for instrument, audio_path in iter_audio_files(dataset_dir):
        try:
            rows.append((instrument, audio_path, extract_features(audio_path)))
        except Exception as error:
            print(f"Skipping {audio_path}: {error}")

    if not rows:
        print("No supported audio files found.")
        return 0

    scaler = StandardScaler()
    scaler.fit(np.stack([features for _, _, features in rows]))
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, model_path)
    print(f"Saved scaler for {len(rows)} audio files to {model_path}")
    return len(rows)


if __name__ == "__main__":
    preprocess()
