import streamlit as st
import cv2
import tempfile
import os
from yt_dlp import YoutubeDL
from processor import VideoProcessor

# YOLO 설정 디렉토리를 쓰기 가능한 /tmp로 변경 (로그의 경고 해결)
os.environ['YOLO_CONFIG_DIR'] = '/tmp/Ultralytics'

st.set_page_config(page_title="TacticalEye - YouTube 분석", layout="wide")
st.title("⚽ TacticalEye: 유튜브 경기 분석")

if 'processor' not in st.session_state:
    st.session_state.processor = VideoProcessor()

input_source = st.sidebar.radio("영상 소스 선택", ["YouTube 링크", "파일 업로드"])
video_path = None

if input_source == "YouTube 링크":
    url = st.text_input("유튜브 URL 입력")
    if url:
        with st.spinner("영상을 최적화하여 가져오는 중..."):
            try:
                ydl_opts = {
                    # 'worst'로 설정하여 메모리 절약 (분석용이므로 고화질 불필요)
                    'format': 'worst[ext=mp4]/bestvideo[height<=360][ext=mp4]',
                    'outtmpl': '/tmp/temp_video.%(ext)s', # 경로를 /tmp로 명시
                    'quiet': True,
                    'no_warnings': True
                }
                with YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    video_path = ydl.prepare_filename(info)
            except Exception as e:
                st.error(f"유튜브 에러: {e}")

# ... (중략) ...

if video_path:
    cap = cv2.VideoCapture(video_path)
    # 영상 정보 확인 (FPS 조절용)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    st_frame = st.empty()
    stop_button = st.button("분석 중지")

    while cap.isOpened():
        if stop_button:
            break
            
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임 크기를 줄여서 처리 속도 및 메모리 확보 (360p 권장)
        frame = cv2.resize(frame, (640, 360))
        
        annotated_frame, current_ids = st.session_state.processor.process_frame(frame)
        
        st_frame.image(annotated_frame, channels="BGR")
        # caption이 너무 자주 업데이트되면 렉이 걸릴 수 있음
        
    cap.release()
    if os.path.exists(video_path):
        os.remove(video_path)
