import base64
import json
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from io import BytesIO

import av
import numpy as np
import requests
import streamlit as st
from PIL import Image
from streamlit_webrtc import RTCConfiguration, webrtc_streamer


# ==================================================
# KONFIGURASI
# ==================================================

st.set_page_config(
    page_title="RoadDamageAI",
    page_icon="🛣️",
    layout="wide",
)

API_URL = "https://hospitality-vitamins-peterson-wallpapers.trycloudflare.com/"
DATABASE_PATH = (
    Path(__file__).resolve().parent
    / "riwayat_roaddamageai.db"
)


# ==================================================
# DATABASE SQLITE
# ==================================================

def koneksi_database():
    return sqlite3.connect(DATABASE_PATH)


def siapkan_database():
    with koneksi_database() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS riwayat_deteksi (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                waktu TEXT NOT NULL,
                nama_file TEXT NOT NULL,
                jumlah_kerusakan INTEGER NOT NULL,
                rata_rata_confidence REAL NOT NULL
            )
            """
        )


def simpan_riwayat(nama_file, jumlah, rata_rata):
    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with koneksi_database() as conn:
        conn.execute(
            """
            INSERT INTO riwayat_deteksi
            (waktu, nama_file, jumlah_kerusakan,
             rata_rata_confidence)
            VALUES (?, ?, ?, ?)
            """,
            (waktu, nama_file, jumlah, rata_rata),
        )


def ambil_riwayat():
    with koneksi_database() as conn:
        conn.row_factory = sqlite3.Row
        data = conn.execute(
            """
            SELECT id, waktu, nama_file,
                   jumlah_kerusakan, rata_rata_confidence
            FROM riwayat_deteksi
            ORDER BY id DESC
            """
        ).fetchall()

    return [dict(baris) for baris in data]


def hapus_riwayat():
    with koneksi_database() as conn:
        conn.execute("DELETE FROM riwayat_deteksi")


siapkan_database()


# ==================================================
# PEMROSES KAMERA REAL-TIME
# ==================================================

class PemrosesKamera:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame_terakhir = None
        self.jumlah_kerusakan = 0
        self.rata_rata_confidence = 0.0
        self.waktu_proses_terakhir = 0.0
        self.error = None

    def proses_frame(self, frame):
        gambar = frame.to_ndarray(format="bgr24")
        waktu_sekarang = time.time()

        # Batasi permintaan ke API agar server tidak
        # dibebani permintaan pada setiap frame.
        if waktu_sekarang - self.waktu_proses_terakhir >= 0.5:
            self.waktu_proses_terakhir = waktu_sekarang

            try:
                gambar_pil = Image.fromarray(
                    gambar[:, :, ::-1]
                )

                buffer = BytesIO()
                gambar_pil.save(buffer, format="JPEG", quality=80)

                respons = requests.post(
                    f"{API_URL}/predict",
                    files={
                        "file": (
                            "kamera.jpg",
                            buffer.getvalue(),
                            "image/jpeg",
                        )
                    },
                    timeout=20,
                )
                respons.raise_for_status()
                hasil = respons.json()

                daftar_deteksi = hasil.get("detections", [])

                confidence = [
                    float(item["confidence_score"])
                    for item in daftar_deteksi
                    if item.get("confidence_score") is not None
                ]

                rata_rata = (
                    sum(confidence) / len(confidence)
                    if confidence
                    else 0.0
                )

                gambar_base64 = hasil.get("annotated_image")

                if gambar_base64:
                    gambar_hasil = Image.open(
                        BytesIO(
                            base64.b64decode(gambar_base64)
                        )
                    ).convert("RGB")

                    gambar_bgr = np.array(
                        gambar_hasil
                    )[:, :, ::-1].copy()
                else:
                    gambar_bgr = gambar

                with self.lock:
                    self.frame_terakhir = gambar_bgr
                    self.jumlah_kerusakan = int(
                        hasil.get("total_detections", 0)
                    )
                    self.rata_rata_confidence = rata_rata
                    self.error = None

            except Exception as error:
                with self.lock:
                    self.error = str(error)

        with self.lock:
            if self.frame_terakhir is not None:
                gambar_tampil = self.frame_terakhir.copy()
            else:
                gambar_tampil = gambar

        return av.VideoFrame.from_ndarray(
            gambar_tampil,
            format="bgr24",
        )


# ==================================================
# TAMPILAN
# ==================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1220;
        color: #f1f5f9;
    }

    h1, h2, h3 {
        color: #f8fafc;
    }

    .subjudul {
        color: #a7b8d3;
        font-size: 15px;
    }

    div.stButton > button {
        border-radius: 9px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("RoadDamageAI")
st.markdown(
    '<p class="subjudul">'
    "Sistem Deteksi dan Analisis Kerusakan Jalan "
    "Menggunakan YOLO"
    "</p>",
    unsafe_allow_html=True,
)

st.divider()


# ==================================================
# SIDEBAR DAN STATUS API
# ==================================================

with st.sidebar:
    st.header("Panel Sistem")

    try:
        status = requests.get(
            f"{API_URL}/health",
            timeout=3,
        )

        if status.ok:
            st.success("API terhubung")
        else:
            st.warning("API tidak merespons dengan normal")

    except requests.RequestException:
        st.error("API belum terhubung")
        st.caption(
            "Jalankan server FastAPI terlebih dahulu."
        )

    st.divider()
    st.caption("Model: YOLO")
    st.caption("Backend: FastAPI")
    st.caption("Database: SQLite")


# ==================================================
# MENU UTAMA
# ==================================================

menu = st.radio(
    "Pilih Menu",
    [
        "Deteksi Kerusakan",
        "Kamera Real-Time",
        "Riwayat Deteksi",
    ],
    horizontal=True,
)


# ==================================================
# MENU 1: UNGGAH GAMBAR
# ==================================================

if menu == "Deteksi Kerusakan":

    st.subheader("📷 Unggah Gambar Jalan")

    file_gambar = st.file_uploader(
        "Pilih gambar JPG, JPEG, atau PNG",
        type=["jpg", "jpeg", "png"],
        key="unggah_gambar",
    )

    if file_gambar is not None:
        st.image(
            file_gambar,
            caption="Gambar asli",
            use_container_width=True,
        )

        if st.button(
            "Analisis Kerusakan",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner("Menganalisis gambar..."):
                    respons = requests.post(
                        f"{API_URL}/predict",
                        files={
                            "file": (
                                file_gambar.name,
                                file_gambar.getvalue(),
                                file_gambar.type,
                            )
                        },
                        timeout=180,
                    )
                    respons.raise_for_status()
                    hasil = respons.json()

                    deteksi = hasil.get("detections", [])
                    jumlah = int(
                        hasil.get("total_detections", 0)
                    )

                    confidence = [
                        float(item["confidence_score"])
                        for item in deteksi
                        if item.get("confidence_score") is not None
                    ]

                    rata_rata = (
                        sum(confidence) / len(confidence)
                        if confidence
                        else 0.0
                    )

                    simpan_riwayat(
                        file_gambar.name,
                        jumlah,
                        rata_rata,
                    )

                    st.session_state["hasil_terakhir"] = hasil

                st.success(
                    "Analisis berhasil dan riwayat tersimpan."
                )

            except requests.RequestException as error:
                st.error(f"Gagal menghubungi API: {error}")

            except (ValueError, KeyError, TypeError) as error:
                st.error(f"Format hasil tidak sesuai: {error}")

            except sqlite3.Error as error:
                st.error(f"Kesalahan database: {error}")

    if "hasil_terakhir" in st.session_state:
        hasil = st.session_state["hasil_terakhir"]
        deteksi = hasil.get("detections", [])
        jumlah = int(hasil.get("total_detections", 0))

        confidence = [
            float(item["confidence_score"])
            for item in deteksi
            if item.get("confidence_score") is not None
        ]

        rata_rata = (
            sum(confidence) / len(confidence)
            if confidence
            else 0.0
        )

        st.divider()
        st.subheader("Hasil Analisis")

        kolom1, kolom2, kolom3 = st.columns(3)

        kolom1.metric("Total Kerusakan", jumlah)
        kolom2.metric(
            "Rata-rata Confidence",
            f"{rata_rata * 100:.2f}%",
        )
        kolom3.metric("Objek Terdeteksi", len(deteksi))

        gambar_base64 = hasil.get("annotated_image")

        if gambar_base64:
            try:
                gambar_bytes = base64.b64decode(gambar_base64)

                st.image(
                    gambar_bytes,
                    caption="Hasil Deteksi YOLO",
                    use_container_width=True,
                )

                st.download_button(
                    "⬇️ Unduh Gambar Hasil",
                    data=gambar_bytes,
                    file_name="hasil_deteksi_roaddamageai.jpg",
                    mime="image/jpeg",
                )
            except (ValueError, TypeError):
                st.warning("Gambar hasil tidak dapat dibaca.")

        if deteksi:
            st.subheader("Rincian Kerusakan")

            for nomor, item in enumerate(deteksi, start=1):
                with st.expander(
                    f"{nomor}. {item.get('class_label', 'Tidak diketahui')}"
                ):
                    st.write(
                        "Confidence: "
                        f"{float(item.get('confidence_score', 0)) * 100:.2f}%"
                    )
                    st.write(
                        "Bounding box: "
                        f"{item.get('bounding_box', [])}"
                    )
        else:
            st.info("Tidak ditemukan kerusakan pada gambar.")

        st.download_button(
            "Unduh Laporan JSON",
            data=json.dumps(
                hasil,
                indent=4,
                ensure_ascii=False,
            ).encode("utf-8"),
            file_name="laporan_roaddamageai.json",
            mime="application/json",
        )


# ==================================================
# MENU 2: KAMERA REAL-TIME
# ==================================================

elif menu == "Kamera Real-Time":

    st.subheader("Deteksi Kerusakan melalui Kamera")

    st.write(
        "Izinkan akses kamera pada browser. "
        "Hasil video diproses oleh YOLO melalui FastAPI."
    )

    pilihan_kamera = st.selectbox(
        "Pilih kamera",
        [
            "Kamera depan / webcam laptop",
            "Kamera belakang HP",
        ],
    )

    facing_mode = (
        "environment"
        if pilihan_kamera == "Kamera belakang HP"
        else "user"
    )

    if "pemroses_kamera" not in st.session_state:
        st.session_state["pemroses_kamera"] = (
            PemrosesKamera()
        )

    pemroses = st.session_state["pemroses_kamera"]

    st.info(
        "Pada laptop, pilihan kamera menggunakan webcam "
        "yang dipilih browser. Pada HP, pilih kamera belakang "
        "untuk mengarahkan kamera ke permukaan jalan."
    )

    kamera = webrtc_streamer(
        key="kamera-roaddamageai",
        video_frame_callback=pemroses.proses_frame,
        media_stream_constraints={
            "video": {
                "facingMode": {"ideal": facing_mode},
                "width": {"ideal": 640},
                "height": {"ideal": 480},
            },
            "audio": False,
        },
        rtc_configuration=RTCConfiguration(
            {
                "iceServers": [
                    {
                        "urls": [
                            "stun:stun.l.google.com:19302"
                        ]
                    }
                ]
            }
        ),
        async_processing=True,
    )

    st.caption(
        "Tekan START untuk memulai kamera dan STOP "
        "untuk menghentikannya."
    )

    with pemroses.lock:
        jumlah_live = pemroses.jumlah_kerusakan
        confidence_live = pemroses.rata_rata_confidence
        error_live = pemroses.error
        frame_live = (
            pemroses.frame_terakhir.copy()
            if pemroses.frame_terakhir is not None
            else None
        )

    kolom1, kolom2 = st.columns(2)

    kolom1.metric(
        "Kerusakan pada frame terakhir",
        jumlah_live,
    )

    kolom2.metric(
        "Rata-rata Confidence",
        f"{confidence_live * 100:.2f}%",
    )

    if error_live:
        st.warning(
            "Pemrosesan kamera mengalami masalah. "
            "Pastikan FastAPI aktif dan periksa koneksi API."
        )

    if kamera.state.playing and frame_live is not None:
        st.caption(
            "Angka statistik diperbarui ketika aplikasi "
            "melakukan pemrosesan ulang."
        )

    st.divider()

    st.markdown("**Simpan hasil kamera ke riwayat**")

    if frame_live is not None:
        st.image(
            frame_live[:, :, ::-1],
            caption="Frame hasil pemrosesan terakhir",
            use_container_width=True,
        )

        if st.button(
            "Simpan hasil deteksi kamera",
            use_container_width=True,
        ):
            try:
                with pemroses.lock:
                    jumlah_simpan = (
                        pemroses.jumlah_kerusakan
                    )
                    confidence_simpan = (
                        pemroses.rata_rata_confidence
                    )
                    gambar_simpan = (
                        pemroses.frame_terakhir.copy()
                    )

                simpan_riwayat(
                    f"kamera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                    jumlah_simpan,
                    confidence_simpan,
                )

                buffer = BytesIO()
                Image.fromarray(
                    gambar_simpan[:, :, ::-1]
                ).save(buffer, format="JPEG")

                st.download_button(
                    "Unduh frame kamera",
                    data=buffer.getvalue(),
                    file_name="hasil_kamera_roaddamageai.jpg",
                    mime="image/jpeg",
                )

                st.success("Riwayat deteksi kamera tersimpan.")

            except sqlite3.Error as error:
                st.error(f"Gagal menyimpan riwayat: {error}")


# ==================================================
# MENU 3: RIWAYAT
# ==================================================

elif menu == "Riwayat Deteksi":

    st.subheader("Riwayat Deteksi")

    try:
        riwayat = ambil_riwayat()

        if riwayat:
            kolom1, kolom2 = st.columns(2)

            kolom1.metric(
                "Total Analisis",
                len(riwayat),
            )

            kolom2.metric(
                "Total Kerusakan",
                sum(
                    item["jumlah_kerusakan"]
                    for item in riwayat
                ),
            )

            st.dataframe(
                [
                    {
                        "Waktu": item["waktu"],
                        "Nama File": item["nama_file"],
                        "Jumlah Kerusakan": (
                            item["jumlah_kerusakan"]
                        ),
                        "Rata-rata Confidence": (
                            f"{item['rata_rata_confidence'] * 100:.2f}%"
                        ),
                    }
                    for item in riwayat
                ],
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Unduh Riwayat JSON",
                data=json.dumps(
                    riwayat,
                    indent=4,
                    ensure_ascii=False,
                ).encode("utf-8"),
                file_name="riwayat_roaddamageai.json",
                mime="application/json",
            )

            with st.expander("Hapus riwayat"):
                konfirmasi = st.checkbox(
                    "Saya yakin ingin menghapus seluruh riwayat."
                )

                if st.button(
                    "Hapus Seluruh Riwayat",
                    disabled=not konfirmasi,
                ):
                    hapus_riwayat()
                    st.success("Riwayat berhasil dihapus.")
                    st.rerun()

        else:
            st.info("Belum ada riwayat deteksi.")

    except sqlite3.Error as error:
        st.error(f"Gagal membaca database: {error}")


# ==================================================
# PENUTUP
# ==================================================

st.divider()

st.caption(
    "RoadDamageAI | YOLO • FastAPI • Streamlit • SQLite"
)
