import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# Configure page
st.set_page_config(
    page_title="Face Detection App",
    page_icon="👤",
    layout="wide"
)

# Load OpenCV face detection classifier
@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_cascade = load_face_cascade()
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Add face count
        cv2.putText(img, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("🎯 Face Detection App")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Settings")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This app uses OpenCV to detect faces in real-time through your webcam. "
        "Green rectangles will appear around detected faces."
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Face Detection")
        
        # WebRTC configuration for better connectivity
        RTC_CONFIGURATION = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })
        
        # Start webcam stream
        webrtc_ctx = webrtc_streamer(
            key="face-detection",
            video_processor_factory=VideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        if webrtc_ctx.video_processor:
            st.success("✅ Face detection is active!")
        else:
            st.warning("⚠️ Click 'START' to begin face detection")
    
    with col2:
        st.subheader("📊 Information")
        
        st.markdown("### How it works:")
        st.markdown("""
        1. **Click START** to activate your webcam
        2. **Allow camera access** when prompted
        3. **Position your face** in front of the camera
        4. **Green rectangles** will appear around detected faces
        5. **Face count** is displayed in the top-left corner
        """)
        
        st.markdown("### Features:")
        st.markdown("""
        - ✅ Real-time face detection
        - ✅ Multiple face detection
        - ✅ Face counting
        - ✅ Responsive design
        - ✅ Privacy-focused (no data stored)
        """)
        
        st.markdown("### Requirements:")
        st.markdown("""
        - 📷 Working webcam
        - 🌐 Modern web browser
        - 💡 Good lighting conditions
        """)

if __name__ == "__main__":
    main()

