import streamlit as st
import cv2
import tempfile
import os
from yt_dlp import YoutubeDL
from processor import VideoProcessor

st.set_page_config(page_title="TacticalEye - YouTube 분석", layout="wide")
st.title("⚽ TacticalEye: 유튜브 경기 분석")

# AI 프로세서 초기화
if 'processor' not in st.session_state:
    st.session_state.processor = VideoProcessor()

# 사이드바: 입력 방식 선택
input_source = st.sidebar.radio("영상 소스 선택", ["YouTube 링크", "파일 업로드"])

video_path = None

if input_source == "YouTube 링크":
    url = st.text_input("유튜브 URL을 입력하세요 (예: https://www.youtube.com/watch?v=...)")
    if url:
        with st.spinner("유튜브 영상을 가져오는 중..."):
            try:
                # yt-dlp 설정 (가장 낮은 화질로 가져와서 속도 최적화)
                ydl_opts = {
                    'format': 'best[ext=mp4]/best',
                    'outtmpl': 'temp_video.%(ext)s',
                    'quiet': True
                }
                with YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    video_path = ydl.prepare_filename(info)
            except Exception as e:
                st.error(f"유튜브 영상을 불러오지 못했습니다: {e}")

else:
    uploaded_file = st.file_uploader("영상을 업로드하세요", type=["mp4", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

# 분석 실행
if video_path:
    cap = cv2.VideoCapture(video_path)
    st_frame = st.empty()
    
    # 분석 중지 버튼
    if st.button("분석 중지"):
        st.rerun()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # AI 처리 (Pose + Tracking)
        annotated_frame, current_ids = st.session_state.processor.process_frame(frame)
        
        # 화면 출력
        st_frame.image(annotated_frame, channels="BGR")
        st.caption(f"감지된 선수 ID: {list(current_ids)}")

    cap.release()
    # 유튜브 임시 파일 삭제
    if input_source == "YouTube 링크" and os.path.exists(video_path):
        os.remove(video_path)
    st.success("분석 완료!")
