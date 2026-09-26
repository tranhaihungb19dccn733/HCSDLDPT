"""Build the feature database from dataset audio files."""

import argparse
import json
import sqlite3
from pathlib import Path

import joblib

try:
    from .create_database import create_database
    from .feature_extraction import extract_features, iter_audio_files
except ImportError:
    from create_database import create_database
    from feature_extraction import extract_features, iter_audio_files


ROOT = Path(__file__).parents[1]


def build_database(dataset_dir: str | Path = ROOT / "dataset", database_path: str | Path = ROOT / "database" / "music.db", scaler_path: str | Path = ROOT / "models" / "scaler.pkl") -> int:
    create_database()
    scaler = joblib.load(scaler_path)
    inserted = 0
    with sqlite3.connect(database_path) as connection:
        connection.execute("DELETE FROM audio_features")
        connection.execute("DELETE FROM tracks")
        for instrument, audio_path in iter_audio_files(dataset_dir):
            try:
                features = scaler.transform([extract_features(audio_path)])[0]
                cursor = connection.execute(
                    "INSERT INTO tracks(filename, filepath, instrument, file_format) VALUES (?, ?, ?, ?)",
                    (audio_path.name, str(audio_path), instrument, audio_path.suffix.lower()),
                )
                connection.execute(
                    "INSERT INTO audio_features(track_id, feature_vector, feature_dimension) VALUES (?, ?, ?)",
                    (cursor.lastrowid, json.dumps(features.tolist()), features.size),
                )
                inserted += 1
            except Exception as error:
                print(f"Skipping {audio_path}: {error}")
    print(f"Inserted {inserted} records into {database_path}")
    return inserted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=ROOT / "dataset")
    parser.add_argument("--database", default=ROOT / "database" / "music.db")
    parser.add_argument("--scaler", default=ROOT / "models" / "scaler.pkl")
    args = parser.parse_args()
    build_database(args.dataset, args.database, args.scaler)
