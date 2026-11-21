"""Giao diện tìm kiếm âm thanh tương đồng."""

from pathlib import Path
import sqlite3
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

from src.similarity import search_similar


ROOT = Path(__file__).parent
DATABASE_PATH = ROOT / "database" / "music.db"
SCALER_PATH = ROOT / "models" / "scaler.pkl"
AUDIO_TYPES = ["wav", "mp3", "flac", "ogg", "m4a"]
INSTRUMENT_NAMES = {
    "accordion": "Phím kéo",
    "acoustic_guitar": "Guitar acoustic",
    "banjo": "Banjo",
    "drum": "Trống",
    "flute": "Sáo",
    "piano": "Piano",
    "saxophone": "Saxophone",
    "trumpet": "Kèn trumpet",
    "violin": "Violin",
}

st.set_page_config(
    page_title="Tìm kiếm âm thanh",
    page_icon="",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
    :root {
        --navy: #17232f;
        --blue: #3157d5;
        --blue-soft: #edf2ff;
        --paper: #FFDEAD;
        --line: #e3e7ef;
        --muted: #697586;
        --green: #18a56f;
    }
    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }
    .stApp,
    .stApp * {
        color: #000000 !important;
    }
    .stButton > button[kind="primary"],
    .stButton > button[kind="primary"] * {
        color: #ffffff !important;
    }
    input, textarea {
        color: #000000 !important;
    }
    .stApp {
        background: var(--paper);
    }
    .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }
    header[data-testid="stHeader"] {
        background: transparent;
    }
    h1, h2, h3 {
        color: var(--navy);
        letter-spacing: -0.035em;
    }
    h1 {
        font-size: 3.35rem !important;
        line-height: 1.05 !important;
        margin-bottom: .35rem !important;
    }
    h2 {
        font-size: 1.55rem !important;
    }
    .brand {
        display: flex;
        align-items: center;
        gap: .7rem;
        color: var(--navy);
        font-weight: 800;
        font-size: 1.05rem;
        margin-bottom: 2.2rem;
    }
    .brand-mark {
        width: 38px;
        height: 38px;
        display: grid;
        place-items: center;
        border-radius: 12px;
        background: var(--navy);
        color: white;
        font-size: 1.1rem;
    }
    .hero {
        display: grid;
        grid-template-columns: 1fr 380px;
        gap: 2.5rem;
        align-items: end;
        padding-bottom: 2rem;
        border-bottom: 1px solid var(--line);
    }
    .hero-copy {
        max-width: 680px;
        color: var(--muted);
        font-size: 1.03rem;
        line-height: 1.65;
    }
    .stats-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        background: white;
        border: 1px solid var(--line);
        border-radius: 18px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(32, 48, 74, .05);
    }
    .stat {
        padding: 1.15rem 1.2rem;
        border-right: 1px solid var(--line);
    }
    .stat:last-child {
        border-right: none;
    }
    .stat-value {
        color: var(--navy);
        font-size: 1.55rem;
        font-weight: 800;
        line-height: 1;
    }
    .stat-label {
        color: var(--muted);
        font-size: .74rem;
        margin-top: .45rem;
    }
    div[data-testid="stTabs"] {
        margin-top: 1.25rem;
    }
    div[data-baseweb="tab-list"] {
        gap: .35rem;
        border-bottom: 1px solid var(--line);
    }
    button[data-baseweb="tab"] {
        color: var(--muted);
        font-weight: 700;
        padding: .75rem 1rem;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--blue);
    }
    div[data-testid="stFileUploader"] section {
        border: 1.5px dashed #cbd4e5;
        border-radius: 14px;
        background: #fafbfe;
        min-height: 150px;
        display: grid;
        align-content: center;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: var(--blue);
        background: var(--blue-soft);
    }
    .stButton > button {
        border-radius: 10px;
        min-height: 46px;
        font-weight: 800;
        border: none;
    }
    .stButton > button[kind="primary"] {
        background: var(--blue);
        color: white;
        box-shadow: 0 8px 18px rgba(49, 87, 213, .18);
    }
    .results-heading {
        display: flex;
        align-items: end;
        justify-content: space-between;
        margin: 1.9rem 0 .75rem;
    }
    .results-heading h2 {
        margin: 0;
    }
    .result-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: .8rem;
    }
    .result-card {
        background: white;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1.1rem;
        box-shadow: 0 8px 24px rgba(32, 48, 74, .04);
    }
    .result-card:first-child {
        grid-column: 1 / -1;
        border-left: 4px solid var(--green);
    }
    .rank-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
    }
    .rank {
        color: var(--blue);
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
    }
    .score {
        color: var(--green);
        background: #ecf8f3;
        border-radius: 999px;
        padding: .35rem .65rem;
        font-size: .78rem;
        font-weight: 800;
    }
    .instrument {
        color: var(--navy);
        font-size: 1.15rem;
        font-weight: 800;
        margin: .55rem 0 .25rem;
    }
    .file-name {
        color: var(--muted);
        font-size: .78rem;
        overflow-wrap: anywhere;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: .85rem 1rem;
    }
    .process-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: .7rem;
        margin-top: .8rem;
    }
    .process-step {
        background: white;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1rem;
    }
    .process-number {
        color: var(--blue);
        font-size: .72rem;
        font-weight: 800;
    }
    .process-title {
        color: var(--navy);
        font-weight: 800;
        margin: .45rem 0 .25rem;
    }
    .process-text {
        color: var(--muted);
        font-size: .78rem;
        line-height: 1.5;
    }
    @media (max-width: 800px) {
        .block-container { padding: 1.25rem 1rem 3rem; }
        h1 { font-size: 2.45rem !important; }
        .hero { grid-template-columns: 1fr; gap: 1.25rem; }
        .stats-row { max-width: none; }
        .result-grid, .process-grid { grid-template-columns: 1fr; }
        .result-card:first-child { grid-column: auto; }
    }
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


def render_result(rank: int, result: dict) -> None:
    path = Path(result["file_path"])
    score = 100 * np.exp(-result["distance"] / 10)
    instrument = INSTRUMENT_NAMES.get(
        result["instrument"],
        result["instrument"].replace("_", " ").title(),
    )

    st.markdown(
        f"""
        <div class="result-card">
            <div class="rank-row">
                <span class="rank">Kết quả #{rank:02d}</span>
                <span class="score">{score:.1f}% tương đồng</span>
            </div>
            <div class="instrument">{instrument}</div>
            <div class="file-name">{path.name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if path.exists():
        st.audio(str(path))


total_tracks, feature_total, label_counts = database_summary()

st.markdown(
    '<div class="brand"><span class="brand-mark">♪</span><span>TÌM KIẾM ÂM THANH</span></div>',
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
        <div>
            <h1>Tìm kiếm âm thanh<br>tương đồng</h1>
            <div class="hero-copy">Tải một đoạn nhạc lên để xác định nhanh những bản ghi có đặc trưng âm thanh gần giống nhất trong hệ thống.</div>
        </div>
        <div class="stats-row">
            <div class="stat"><div class="stat-value">{total_tracks}</div><div class="stat-label">Bản ghi</div></div>
            <div class="stat"><div class="stat-value">{len(label_counts)}</div><div class="stat-label">Nhạc cụ</div></div>
            <div class="stat"><div class="stat-value">90</div><div class="stat-label">Đặc trưng</div></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

search_tab, system_tab = st.tabs(["Tìm kiếm", "Thông tin hệ thống"])

with search_tab:
    left, right = st.columns([1.08, 1.92], gap="large")

    with left:
        st.subheader("Thêm file truy vấn")
        query_file = st.file_uploader(
            "Tải lên file âm thanh",
            type=AUDIO_TYPES,
            help="Hỗ trợ WAV, MP3, FLAC, OGG và M4A",
        )
        if query_file:
            st.audio(query_file)
        search_clicked = st.button(
            "Tìm kiếm bản ghi giống nhất",
            type="primary",
            width="stretch",
            disabled=query_file is None,
        )

    with right:
        st.subheader("Kết quả")
        query_state = st.session_state.get("last_query")

        if search_clicked and query_file:
            if not DATABASE_PATH.exists() or not SCALER_PATH.exists():
                st.error("Thiếu cơ sở dữ liệu hoặc bộ chuẩn hóa. Hãy chạy pipeline tiền xử lý trước.")
            else:
                with tempfile.NamedTemporaryFile(
                    suffix=Path(query_file.name).suffix,
                    delete=False,
                ) as temporary_file:
                    temporary_file.write(query_file.getbuffer())
                    query_path = Path(temporary_file.name)

                with st.spinner("Đang phân tích và so khớp âm thanh..."):
                    try:
                        results = search_similar(
                            query_path,
                            DATABASE_PATH,
                            SCALER_PATH,
                            5,
                        )
                        st.session_state["last_query"] = {
                            "name": query_file.name,
                            "results": results,
                        }
                        query_state = st.session_state["last_query"]
                    except Exception as error:
                        st.error(f"Không thể xử lý file này: {error}")
                    finally:
                        query_path.unlink(missing_ok=True)

        if query_state:
            st.caption(f"Top 5 bản ghi cho {query_state['name']}")
            for rank, result in enumerate(query_state["results"], start=1):
                render_result(rank, result)
        else:
            st.markdown(
                """
                <div style="height:280px;display:grid;place-items:center;text-align:center;
                border:1px dashed #dce2ec;border-radius:16px;background:white;color:#697586;">
                    <div>
                        <div style="font-size:2rem;margin-bottom:.5rem;">♪</div>
                        <div style="font-weight:700;color:#17232f;">Sẵn sàng tìm kiếm</div>
                        <div style="font-size:.82rem;margin-top:.35rem;">Tải lên một file âm thanh để bắt đầu</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with system_tab:
    st.subheader("Tổng quan dữ liệu")
    metric_columns = st.columns(4)
    metrics = [
        ("Bản ghi", total_tracks),
        ("Nhạc cụ", len(label_counts)),
        ("Vector đặc trưng", feature_total),
        ("Chiều vector", 90),
    ]
    for column, (label, value) in zip(metric_columns, metrics):
        with column:
            st.metric(label, value)

    st.subheader("Quy trình tìm kiếm")
    st.markdown(
        """
        <div class="process-grid">
            <div class="process-step"><div class="process-number">01</div><div class="process-title">Tải file</div><div class="process-text">Nhận một file âm thanh nguồn từ giao diện.</div></div>
            <div class="process-step"><div class="process-number">02</div><div class="process-title">Tiền xử lý</div><div class="process-text">Chuyển mono, lấy mẫu và chuẩn hóa thành 3 giây.</div></div>
            <div class="process-step"><div class="process-number">03</div><div class="process-title">Trích đặc trưng</div><div class="process-text">Tạo vector 90 chiều từ các đặc trưng âm thanh.</div></div>
            <div class="process-step"><div class="process-number">04</div><div class="process-title">So khớp</div><div class="process-text">Xếp hạng bằng khoảng cách Euclid và trả về top 5.</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not label_counts.empty:
        st.subheader("Phân bố nhạc cụ")
        left, right = st.columns([1.7, 1])
        with left:
            chart_frame = label_counts.copy()
            chart_frame["instrument"] = chart_frame["instrument"].map(
                lambda name: INSTRUMENT_NAMES.get(name, name)
            )
            chart_frame = chart_frame.set_index("instrument")
            st.bar_chart(chart_frame, height=260)
        with right:
            display_counts = label_counts.copy()
            display_counts["instrument"] = display_counts["instrument"].map(
                lambda name: INSTRUMENT_NAMES.get(name, name)
            )
            display_counts.columns = ["Nhạc cụ", "Số lượng"]
            st.dataframe(display_counts, width="stretch", hide_index=True)
