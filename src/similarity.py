"""Nearest-neighbor search over stored audio features."""

import sqlite3
import json
from pathlib import Path

import joblib
import numpy as np

try:
    from .feature_extraction import extract_features
except ImportError:
    from feature_extraction import extract_features


def search_similar(query_path: str | Path, database_path: str | Path, scaler_path: str | Path, limit: int = 5) -> list[dict]:
    scaler = joblib.load(scaler_path)
    query = scaler.transform([extract_features(query_path)])[0]
    results = []
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT tracks.filepath, tracks.instrument, audio_features.feature_vector, audio_features.feature_dimension "
            "FROM tracks JOIN audio_features ON tracks.id = audio_features.track_id"
        ).fetchall()
    for file_path, instrument, feature_vector, feature_size in rows:
        vector = np.asarray(json.loads(feature_vector), dtype=np.float32)
        distance = float(np.linalg.norm(query - vector))
        results.append({"file_path": file_path, "instrument": instrument, "distance": distance})
    return sorted(results, key=lambda item: item["distance"])[:limit]
