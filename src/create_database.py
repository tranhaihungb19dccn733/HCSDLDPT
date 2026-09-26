"""Create SQLite database for the music similarity search system."""

import sqlite3
from pathlib import Path


# Thu muc goc cua project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# database/music.db
DATABASE_DIR = PROJECT_ROOT / "database"
DATABASE_PATH = DATABASE_DIR / "music.db"


def create_database():

    # Tao folder database neu chua ton tai
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    # Ket noi SQLite
    conn = sqlite3.connect(DATABASE_PATH)

    # Bat rang buoc khoa ngoai
    conn.execute("PRAGMA foreign_keys = ON")

    cursor = conn.cursor()

    existing_columns = {
        row[1] for row in cursor.execute("PRAGMA table_info(audio_features)")
    }
    if existing_columns and "track_id" not in existing_columns:
        cursor.execute("DROP TABLE audio_features")

    # ======================================================
    # TABLE 1: TRACKS
    # Luu thong tin cua file audio
    # ======================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            filename TEXT NOT NULL,
            filepath TEXT NOT NULL UNIQUE,
            instrument TEXT NOT NULL,

            duration REAL,
            sample_rate INTEGER,
            file_format TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ======================================================
    # TABLE 2: AUDIO_FEATURES
    # Luu vector 90 dac trung cua moi file
    # ======================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audio_features (
            track_id INTEGER PRIMARY KEY,

            feature_vector TEXT NOT NULL,
            feature_dimension INTEGER NOT NULL DEFAULT 90,

            FOREIGN KEY (track_id)
                REFERENCES tracks(id)
                ON DELETE CASCADE
        )
    """)

    # Index de tim theo nhac cu nhanh hon
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_tracks_instrument
        ON tracks(instrument)
    """)

    conn.commit()
    conn.close()

    print("=" * 60)
    print("DATABASE CREATED SUCCESSFULLY")
    print("=" * 60)
    print(f"Database: {DATABASE_PATH}")
    print("Tables:")
    print("  1. tracks")
    print("  2. audio_features")
    print("=" * 60)


if __name__ == "__main__":
    create_database()