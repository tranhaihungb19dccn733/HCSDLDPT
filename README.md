# Music Search Project

Project tim kiem nhac theo do tuong dong am thanh. He thong trich xuat dac trung MFCC tu cac file audio, luu vao SQLite va tim cac mau gan nhat.

## Cau truc

- `dataset/`: dat file `.wav`, `.mp3` hoac `.flac` vao thu muc nhac cu tuong ung.
- `database/music.db`: SQLite database chua metadata va vector dac trung.
- `models/scaler.pkl`: bo chuan hoa dac trung sau khi chay preprocessing.
- `src/`: cac module xu ly du lieu, database, tim kiem va danh gia.
- `app.py`: giao dien Streamlit de upload file truy van.

## Cai dat

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Chay pipeline

```bash
python "src/preprocess(ChuanHoa).py"
python src/build_database.py
streamlit run app.py
```

Co the xem danh sach tham so:

```bash
python src/build_database.py --help
```
