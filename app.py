"""Web demo for the music search database and similarity engine."""

from pathlib import Path
import sqlite3
import tempfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.feature_extraction import extract_features
from src.similarity import search_similar


ROOT = Path(__file__).parent
DATABASE_PATH = ROOT / "database" / "music.db"
SCALER_PATH = ROOT / "models" / "scaler.pkl"
AUDIO_TYPES = ["wav", "mp3", "flac", "ogg", "m4a"]
FEATURE_GROUPS = {
    "MFCC": (0, 40),
    "Chroma": (40, 64),
    "Độ tương phản phổ": (64, 78),
    "Tâm trọng phổ": (78, 80),
    "Độ rộng phổ": (80, 82),
    "Ngưỡng cuộn phổ": (82, 84),
    "Độ phẳng phổ": (84, 86),
    "Tỉ lệ qua điểm 0": (86, 88),
    "Năng lượng RMS": (88, 90),
}

st.set_page_config(page_title="Resonance | Tìm kiếm âm thanh", page_icon="R", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    :root { --ink: #152323; --muted: #65716f; --mint: #b9e8cf; --coral: #ff8068; --line: #d6e0d8; }
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; color: var(--ink); }
    .stApp { background: radial-gradient(circle at 85% 3%, #d9f0e0 0, transparent 28%), linear-gradient(135deg, #f8f6f0 0%, #edf4ee 100%); }
    .block-container { padding-top: 2rem; max-width: 1240px; }
    h1, h2, h3 { letter-spacing: 0; }
    h1 { font-size: clamp(2.4rem, 5vw, 5.3rem) !important; line-height: .92 !important; max-width: 780px; margin-bottom: .45rem !important; }
    .eyebrow { font-family: 'DM Mono', monospace; color: #df5c48; font-size: .76rem; letter-spacing: .12em; text-transform: uppercase; }
    .lede { color: var(--muted); font-size: 1.08rem; max-width: 640px; line-height: 1.55; margin-top: .75rem; }
    .status-line { display: inline-flex; align-items: center; gap: .5rem; margin-top: 1.1rem; padding: .48rem .72rem; border: 1px solid var(--line); border-radius: 999px; background: rgba(255,255,255,.58); color: #445250; font-family: 'DM Mono', monospace; font-size: .7rem; letter-spacing: .04em; text-transform: uppercase; }
    .status-dot { width: .5rem; height: .5rem; border-radius: 50%; background: #4eaf78; box-shadow: 0 0 0 .22rem rgba(78,175,120,.15); }
    .hero-rule { height: 1px; background: linear-gradient(90deg, var(--ink), var(--line) 65%, transparent); margin: 1.5rem 0 .25rem; }
    .metric-card { border-top: 2px solid var(--ink); padding: .8rem 0; }
    .metric-value { font-size: 2.1rem; font-weight: 700; letter-spacing: -.04em; }
    .metric-label { color: var(--muted); font-size: .82rem; }
    .result-card { background: rgba(255,255,255,.72); border: 1px solid #dce5dc; border-radius: 8px; padding: 1rem 1.15rem; margin: .55rem 0; }
    .rank { font-family: 'DM Mono', monospace; color: #df5c48; font-size: .85rem; }
    .result-title { font-size: 1.15rem; font-weight: 700; }
    .path { color: var(--muted); font-family: 'DM Mono', monospace; font-size: .72rem; overflow-wrap: anywhere; }
    div[data-testid="stMetric"] { background: rgba(255,255,255,.62); border: 1px solid #dce5dc; padding: .8rem; border-radius: 8px; }
    div[data-testid="stFileUploader"] section { border: 1px dashed #9eb4a8; border-radius: 10px; background: rgba(255,255,255,.5); padding: 1.25rem; transition: border-color .2s, background .2s; }
    div[data-testid="stFileUploader"] section:hover { border-color: #df5c48; background: rgba(255,255,255,.85); }
    div[data-baseweb="tab-list"] { gap: .35rem; border-bottom: 1px solid var(--line); }
    button[data-baseweb="tab"] { padding: .65rem .9rem; }
    .section-kicker { color: #df5c48; font-family: 'DM Mono', monospace; font-size: .7rem; letter-spacing: .1em; text-transform: uppercase; margin-bottom: -.55rem; }
    @media (max-width: 700px) { .block-container { padding: 1.25rem 1rem 2rem; } h1 { font-size: 3.15rem !important; } .lede { font-size: .98rem; } .metric-value { font-size: 1.6rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def database_summary() -> tuple[int, int, pd.DataFrame]:
    if not DATABASE_PATH.exists():
        return 0, 0, pd.DataFrame(columns=["instrument", "total"])
    with sqlite3.connect(DATABASE_PATH) as connection:
        total = connection.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
        feature_total = connection.execute("SELECT COUNT(*) FROM audio_features").fetchone()[0]
        labels = pd.read_sql_query(
            "SELECT instrument, COUNT(*) AS total FROM tracks GROUP BY instrument ORDER BY total DESC",
            connection,
        )
    return total, feature_total, labels


def feature_table(vector: np.ndarray) -> pd.DataFrame:
    rows = []
    for group, (start, end) in FEATURE_GROUPS.items():
        values = vector[start:end]
        rows.append({"Nhóm đặc trưng": group, "Số chiều": end - start, "Trung bình": float(values.mean()), "Độ lệch chuẩn": float(values.std())})
    return pd.DataFrame(rows)


def feature_structure_table(vector: np.ndarray | None = None) -> pd.DataFrame:
    rows = []
    for group, (start, end) in FEATURE_GROUPS.items():
        row = {
            "Nhóm đặc trưng": group,
            "Vị trí trong vector": f"{start + 1} - {end}",
            "Số chiều": end - start,
            "Trạng thái": "Đã tính từ truy vấn" if vector is not None else "Chờ dữ liệu",
        }
        if vector is not None:
            values = vector[start:end]
            row["Giá trị trung bình"] = round(float(values.mean()), 5)
            row["Độ lệch chuẩn"] = round(float(values.std()), 5)
        rows.append(row)
    return pd.DataFrame(rows)


def render_result(rank: int, result: dict) -> None:
    path = Path(result["file_path"])
    score = 100 * np.exp(-result["distance"] / 10)
    st.markdown(
        f'<div class="result-card"><span class="rank">RANK {rank:02d}</span> &nbsp; '
        f'<span class="result-title">{result["instrument"].replace("_", " ").title()}</span>'
        f'<br><span class="path">{result["file_path"]}</span></div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([3, 1])
    with left:
        if path.exists():
            st.audio(str(path))
    with right:
        st.metric("Độ tương đồng", f"{score:.1f}%")
        st.caption(f"Khoảng cách Euclid: {result['distance']:.3f}")


st.markdown('<div class="eyebrow">Hệ thống truy vấn thông tin âm thanh / 2026</div>', unsafe_allow_html=True)
st.title("Tìm kiếm đoạn nhạc\ntương đồng.")
st.markdown('<p class="lede">Cơ sở dữ liệu âm thanh gồm 500 đoạn nhạc không lời. Tải lên một file mới để hệ thống xếp hạng 5 bản ghi gần nhất dựa trên chữ ký âm thanh 90 chiều.</p>', unsafe_allow_html=True)
st.markdown('<div class="status-line"><span class="status-dot"></span> Cơ sở dữ liệu đang hoạt động · 500 bản ghi · 10 nhạc cụ</div><div class="hero-rule"></div>', unsafe_allow_html=True)

total_tracks, feature_total, label_counts = database_summary()
metric_columns = st.columns(4)
for column, value, label in zip(metric_columns, [total_tracks, len(label_counts), 90, feature_total], ["Đoạn nhạc đã lập chỉ mục", "Loại nhạc cụ", "Chiều đặc trưng", "Vector đã lưu"]):
    with column:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

search_tab, evidence_tab, system_tab = st.tabs(["Tìm kiếm", "Dữ liệu trung gian", "Thông tin hệ thống"])

with search_tab:
    st.markdown('<div class="section-kicker">Không gian truy vấn</div>', unsafe_allow_html=True)
    st.subheader("Tìm kiếm bằng file âm thanh mới")
    query_file = st.file_uploader("Kéo thả file âm thanh vào đây", type=AUDIO_TYPES, help="Định dạng hỗ trợ: WAV, MP3, FLAC, OGG, M4A")
    if query_file:
        st.audio(query_file)
        with tempfile.NamedTemporaryFile(suffix=Path(query_file.name).suffix, delete=False) as temporary_file:
            temporary_file.write(query_file.getbuffer())
            query_path = Path(temporary_file.name)
        if st.button("Bắt đầu tìm kiếm", type="primary", use_container_width=True):
            if not DATABASE_PATH.exists() or not SCALER_PATH.exists():
                st.error("Thiếu database hoặc scaler. Hãy chạy pipeline tiền xử lý trước.")
            else:
                with st.spinner("Đang trích xuất đặc trưng và xếp hạng database..."):
                    try:
                        raw_vector = extract_features(query_path)
                        scaler = joblib.load(SCALER_PATH)
                        scaled_vector = scaler.transform([raw_vector])[0]
                        results = search_similar(query_path, DATABASE_PATH, SCALER_PATH, 5)
                        st.session_state["last_query"] = {"name": query_file.name, "raw": raw_vector, "scaled": scaled_vector, "results": results}
                    except Exception as error:
                        st.error(f"Không thể xử lý file này: {error}")

    query_state = st.session_state.get("last_query")
    if query_state:
        st.divider()
        st.subheader(f"5 kết quả gần nhất cho {query_state['name']}")
        for rank, result in enumerate(query_state["results"], start=1):
            render_result(rank, result)

with evidence_tab:
    st.subheader("Dữ liệu mà mô hình sử dụng")
    query_state = st.session_state.get("last_query")
    raw_vector = query_state["raw"] if query_state else None
    structure_columns = st.columns(3)
    structure_values = [
        ("Cấu trúc vector", "90 chiều"),
        ("Nhóm đặc trưng", str(len(FEATURE_GROUPS))),
        ("Trạng thái", "Đã cập nhật" if raw_vector is not None else "Mô hình chờ dữ liệu"),
    ]
    for column, (label, value) in zip(structure_columns, structure_values):
        with column:
            st.metric(label, value)
    st.markdown("**Mô hình cấu trúc đặc trưng**")
    st.caption("Mỗi nhóm chiếm một vùng cố định trong vector 90 chiều. Bảng sẽ tự thay đổi giá trị khi có file truy vấn.")
    st.dataframe(feature_structure_table(raw_vector), use_container_width=True, hide_index=True)
    if not query_state:
        st.info("Mô hình cấu trúc đang ở trạng thái chờ dữ liệu. Hãy thực hiện một lần tìm kiếm để cập nhật các giá trị thực tế.")
    else:
        scaled_vector = query_state["scaled"]
        st.caption(f"Vector truy vấn: {len(raw_vector)} chiều | âm thanh chuẩn hóa về 3 giây, tần số 22.050 Hz")
        st.dataframe(feature_table(raw_vector), use_container_width=True, hide_index=True)
        chart_frame = pd.DataFrame({"Giá trị gốc": raw_vector, "Giá trị chuẩn hóa": scaled_vector})
        st.line_chart(chart_frame, height=300)
        st.subheader("Tính toán xếp hạng")
        distance_frame = pd.DataFrame([
            {"Thứ hạng": index, "Nhạc cụ": item["instrument"], "Khoảng cách": item["distance"], "Điểm tương đồng": 100 * np.exp(-item["distance"] / 10)}
            for index, item in enumerate(query_state["results"], start=1)
        ])
        st.dataframe(distance_frame, use_container_width=True, hide_index=True)
        st.caption("Điểm tương đồng được quy đổi từ khoảng cách Euclid theo công thức 100 x exp(-distance / 10). Hệ thống xếp hạng theo khoảng cách nhỏ nhất.")

with system_tab:
    st.subheader("Sơ đồ khối hệ thống")
    st.graphviz_chart(
        """digraph { rankdir=LR; node [shape=box style=rounded color=\"#152323\" fontname=\"Space Grotesk\"]; A [label=\"File am thanh truy van\"]; B [label=\"Tien xu ly\\nmono / 3s / 22.05kHz\"]; C [label=\"Trich xuat dac trung\\nMFCC + pho + chroma\"]; D [label=\"StandardScaler\"]; E [label=\"CSDL dac trung SQLite\\n500 vector\"]; F [label=\"Xep hang bang Euclid\"]; G [label=\"5 ket qua\"]; A -> B -> C -> D -> F -> G; E -> F; }"""
    )
    st.markdown("**Quy trình xử lý**")
    st.write("1. Âm thanh được chuyển về mono, lấy mẫu lại ở 22.050 Hz và cắt hoặc bổ sung để dài 3 giây.")
    st.write("2. Bộ trích xuất tính MFCC, chroma, độ tương phản phổ, tâm trọng phổ, độ rộng phổ, ngưỡng cuộn phổ, độ phẳng phổ, tỉ lệ qua điểm 0 và năng lượng RMS; trung bình và độ lệch chuẩn tạo thành 90 giá trị.")
    st.write("3. StandardScaler đưa các chiều về cùng thang đo. SQLite lưu thông tin file và vector đặc trưng liên kết.")
    st.write("4. Vector truy vấn được so sánh với mọi vector trong database bằng khoảng cách Euclid. Năm kết quả có khoảng cách nhỏ nhất được trả về.")
    st.subheader("Thống kê bộ dữ liệu")
    if not label_counts.empty:
        st.bar_chart(label_counts.set_index("instrument"), height=280)
        st.dataframe(label_counts, use_container_width=True, hide_index=True)
    st.subheader("Đánh giá nhanh")
    st.write("Bộ dữ liệu gồm 10 loại nhạc cụ cân bằng, mỗi loại 50 đoạn nhạc và 500 vector đặc trưng. Kiểm tra tự truy vấn đạt file sẽ đặt hạng 1 với khoảng cách gần 0; chất lượng tìm kiếm liên lớp có thể đánh giá bằng precision@5 và MRR trong src/evaluation.py.")
