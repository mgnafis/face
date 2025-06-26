import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="Face Detection App",
    page_icon="👤",
    layout="wide"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .detection-stats {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🎯 Face Detection App</h1>', unsafe_allow_html=True)

# Fungsi untuk memuat classifier
@st.cache_resource
def load_face_classifier():
    """Load OpenCV face classifier"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

# Fungsi untuk deteksi wajah
def detect_faces(image, face_cascade):
    """Deteksi wajah pada gambar"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces

# Fungsi untuk menggambar kotak di sekitar wajah
def draw_faces(image, faces):
    """Menggambar kotak di sekitar wajah yang terdeteksi"""
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

# Load face classifier
face_cascade = load_face_classifier()

# Sidebar untuk pengaturan
st.sidebar.header("⚙️ Pengaturan")
detection_mode = st.sidebar.selectbox(
    "Pilih Mode Deteksi:",
    ["📸 Upload Gambar", "📹 Webcam Real-time", "🎥 Upload Video"]
)

# Parameter deteksi
st.sidebar.subheader("Parameter Deteksi")
scale_factor = st.sidebar.slider("Scale Factor", 1.05, 2.0, 1.1, 0.05)
min_neighbors = st.sidebar.slider("Min Neighbors", 3, 10, 5)
min_size = st.sidebar.slider("Min Size", 20, 100, 30)

if detection_mode == "📸 Upload Gambar":
    st.markdown('<div class="info-box">📷 <strong>Upload gambar untuk deteksi wajah</strong></div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Pilih gambar...", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Format yang didukung: JPG, JPEG, PNG, BMP"
    )
    
    if uploaded_file is not None:
        # Baca gambar
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Konversi RGB ke BGR untuk OpenCV
        if len(image_np.shape) == 3:
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_cv = image_np
        
        # Deteksi wajah
        faces = face_cascade.detectMultiScale(
            cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY),
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(min_size, min_size)
        )
        
        # Gambar hasil deteksi
        result_image = image_cv.copy()
        result_image = draw_faces(result_image, faces)
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        
        # Tampilkan hasil
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Gambar asli")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Hasil Deteksi")
            st.image(result_image_rgb, use_column_width=True)
        
        # Statistik deteksi
        st.markdown(f'''
        <div class="detection-stats">
            <h3>📊 Statistik Deteksi</h3>
            <p><strong>Jumlah wajah terdeteksi:</strong> {len(faces)}</p>
            <p><strong>Ukuran gambar:</strong> {image.size[0]} x {image.size[1]} pixels</p>
            <p><strong>Parameter yang digunakan:</strong></p>
            <ul>
                <li>Scale Factor: {scale_factor}</li>
                <li>Min Neighbors: {min_neighbors}</li>
                <li>Min Size: {min_size}x{min_size}</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)

elif detection_mode == "📹 Webcam Real-time":
    st.markdown('<div class="info-box">🎥 <strong>Deteksi wajah real-time dari webcam</strong></div>', unsafe_allow_html=True)
    
    # Tombol untuk memulai/berhenti
    start_button = st.button("🚀 Mulai Deteksi Webcam")
    stop_button = st.button("⏹️ Berhenti")
    
    # Placeholder untuk video
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    if start_button:
        # Buka webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Tidak dapat mengakses webcam. Pastikan webcam terhubung dan tidak digunakan aplikasi lain.")
        else:
            st.success("✅ Webcam berhasil diakses!")
            
            # Session state untuk kontrol
            if 'webcam_running' not in st.session_state:
                st.session_state.webcam_running = True
            
            frame_count = 0
            
            while st.session_state.webcam_running:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("❌ Gagal membaca frame dari webcam")
                    break
                
                # Deteksi wajah
                faces = face_cascade.detectMultiScale(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=(min_size, min_size)
                )
                
                # Gambar kotak di sekitar wajah
                frame_with_faces = draw_faces(frame.copy(), faces)
                
                # Konversi BGR ke RGB untuk Streamlit
                frame_rgb = cv2.cvtColor(frame_with_faces, cv2.COLOR_BGR2RGB)
                
                # Tampilkan frame
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Update statistik
                frame_count += 1
                stats_placeholder.markdown(f'''
                <div class="detection-stats">
                    <p><strong>Frame:</strong> {frame_count}</p>
                    <p><strong>Wajah terdeteksi:</strong> {len(faces)}</p>
                    <p><strong>Resolusi:</strong> {frame.shape[1]}x{frame.shape[0]}</p>
                </div>
                ''', unsafe_allow_html=True)
                
                # Berhenti jika tombol stop ditekan
                if stop_button:
                    break
            
            cap.release()
            st.session_state.webcam_running = False

elif detection_mode == "🎥 Upload Video":
    st.markdown('<div class="info-box">🎬 <strong>Upload video untuk deteksi wajah</strong></div>', unsafe_allow_html=True)
    
    uploaded_video = st.file_uploader(
        "Pilih file video...", 
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Format yang didukung: MP4, AVI, MOV, MKV"
    )
    
    if uploaded_video is not None:
        # Simpan video sementara
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close()
        
        # Buka video
        cap = cv2.VideoCapture(tfile.name)
        
        if not cap.isOpened():
            st.error("❌ Tidak dapat membuka file video")
        else:
            # Informasi video
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            st.info(f"📹 Video: {fps} FPS, {frame_count} frames, {duration:.2f} detik")
            
            # Proses video
            process_video = st.button("🎯 Proses Video")
            
            if process_video:
                progress_bar = st.progress(0)
                frame_placeholder = st.empty()
                
                processed_frames = 0
                total_faces_detected = 0
                
                while True:
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                    
                    # Deteksi wajah
                    faces = face_cascade.detectMultiScale(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                        scaleFactor=scale_factor,
                        minNeighbors=min_neighbors,
                        minSize=(min_size, min_size)
                    )
                    
                    total_faces_detected += len(faces)
                    
                    # Gambar kotak di sekitar wajah
                    frame_with_faces = draw_faces(frame.copy(), faces)
                    frame_rgb = cv2.cvtColor(frame_with_faces, cv2.COLOR_BGR2RGB)
                    
                    # Tampilkan setiap 10 frame untuk performa
                    if processed_frames % 10 == 0:
                        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                    
                    processed_frames += 1
                    progress_bar.progress(processed_frames / frame_count)
                
                # Hasil akhir
                st.success("✅ Video berhasil diproses!")
                st.markdown(f'''
                <div class="detection-stats">
                    <h3>📊 Hasil Pemrosesan Video</h3>
                    <p><strong>Total frame diproses:</strong> {processed_frames}</p>
                    <p><strong>Total wajah terdeteksi:</strong> {total_faces_detected}</p>
                    <p><strong>Rata-rata wajah per frame:</strong> {total_faces_detected/processed_frames:.2f}</p>
                </div>
                ''', unsafe_allow_html=True)
        
        cap.release()
        # Hapus file sementara
        os.unlink(tfile.name)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🔧 Dibuat dengan Python, OpenCV, dan Streamlit</p>
    <p>💡 Tips: Pastikan pencahayaan cukup untuk hasil deteksi yang optimal</p>
</div>
""", unsafe_allow_html=True)
