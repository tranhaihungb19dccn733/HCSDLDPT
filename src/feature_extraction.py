"""Audio feature extraction utilities."""

from pathlib import Path

import librosa
import numpy as np


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 20


def mean_std(feature: np.ndarray) -> np.ndarray:
    """
    Tinh mean va standard deviation cua feature
    theo truc thoi gian.
    """
    mean = np.mean(feature, axis=1)
    std = np.std(feature, axis=1)

    return np.concatenate([mean, std])


def extract_features(
    audio_path: str | Path,
    sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """
    Trich xuat vector 90 dac trung tu mot file audio.

    Feature vector:
    - MFCC: 40
    - Chroma: 24
    - Spectral Contrast: 14
    - Spectral Centroid: 2
    - Spectral Bandwidth: 2
    - Spectral Rolloff: 2
    - Spectral Flatness: 2
    - Zero Crossing Rate: 2
    - RMS Energy: 2

    Tong = 90 features
    """

    # ==========================================
    # 1. LOAD AUDIO
    # ==========================================

    signal, sr = librosa.load(
        audio_path,
        sr=sample_rate,
        mono=True,
        duration=DURATION
    )

    if signal.size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")

    # Dam bao moi audio deu dai dung 3 giay
    target_length = sample_rate * DURATION

    signal = librosa.util.fix_length(
        signal,
        size=target_length
    )

    # ==========================================
    # 2. MFCC
    # 20 mean + 20 std = 40
    # ==========================================

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=N_MFCC
    )

    mfcc_features = mean_std(mfcc)

    # ==========================================
    # 3. CHROMA
    # 12 mean + 12 std = 24
    # ==========================================

    chroma = librosa.feature.chroma_stft(
        y=signal,
        sr=sr
    )

    chroma_features = mean_std(chroma)

    # ==========================================
    # 4. SPECTRAL CONTRAST
    # 7 mean + 7 std = 14
    # ==========================================

    spectral_contrast = librosa.feature.spectral_contrast(
        y=signal,
        sr=sr
    )

    contrast_features = mean_std(spectral_contrast)

    # ==========================================
    # 5. SPECTRAL CENTROID
    # ==========================================

    centroid = librosa.feature.spectral_centroid(
        y=signal,
        sr=sr
    )

    centroid_features = np.array([
        np.mean(centroid),
        np.std(centroid)
    ])

    # ==========================================
    # 6. SPECTRAL BANDWIDTH
    # ==========================================

    bandwidth = librosa.feature.spectral_bandwidth(
        y=signal,
        sr=sr
    )

    bandwidth_features = np.array([
        np.mean(bandwidth),
        np.std(bandwidth)
    ])

    # ==========================================
    # 7. SPECTRAL ROLLOFF
    # ==========================================

    rolloff = librosa.feature.spectral_rolloff(
        y=signal,
        sr=sr
    )

    rolloff_features = np.array([
        np.mean(rolloff),
        np.std(rolloff)
    ])

    # ==========================================
    # 8. SPECTRAL FLATNESS
    # ==========================================

    flatness = librosa.feature.spectral_flatness(
        y=signal
    )

    flatness_features = np.array([
        np.mean(flatness),
        np.std(flatness)
    ])

    # ==========================================
    # 9. ZERO CROSSING RATE
    # ==========================================

    zcr = librosa.feature.zero_crossing_rate(
        signal
    )

    zcr_features = np.array([
        np.mean(zcr),
        np.std(zcr)
    ])

    # ==========================================
    # 10. RMS ENERGY
    # ==========================================

    rms = librosa.feature.rms(
        y=signal
    )

    rms_features = np.array([
        np.mean(rms),
        np.std(rms)
    ])

    # ==========================================
    # 11. GHEP THANH VECTOR 90 CHIEU
    # ==========================================

    features = np.concatenate([
        mfcc_features,          # 40
        chroma_features,        # 24
        contrast_features,      # 14
        centroid_features,      # 2
        bandwidth_features,     # 2
        rolloff_features,       # 2
        flatness_features,      # 2
        zcr_features,           # 2
        rms_features            # 2
    ])

    return features.astype(np.float32)


def iter_audio_files(dataset_dir: str | Path):
    """
    Duyet tat ca cac file audio trong dataset.

    Tra ve:
        instrument_name, audio_path
    """

    root = Path(dataset_dir)

    for instrument_dir in sorted(
        path for path in root.iterdir()
        if path.is_dir()
    ):

        for audio_path in sorted(
            instrument_dir.rglob("*")
        ):

            if audio_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield instrument_dir.name, audio_path


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    test_file = Path("dataset/piano/1.wav")

    if not test_file.exists():
        print(f"Khong tim thay file test: {test_file}")

    else:
        features = extract_features(test_file)

        print("=" * 60)
        print("FEATURE EXTRACTION TEST")
        print("=" * 60)

        print(f"File: {test_file}")
        print(f"So luong dac trung: {len(features)}")

        print("\nFeature vector:")
        print(features)

        print("=" * 60)