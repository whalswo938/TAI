import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from yt_dlp import YoutubeDL
from processor import VideoProcessor

# 1. 환경 설정 (로그에 나타난 권한 에러 방지)
os.environ['YOLO_CONFIG_DIR'] = '/tmp/Ultralytics'

st.set_page_config(page_title="TacticalEye - YouTube 분석", layout="wide")
st.title("⚽ TacticalEye: 유튜브 경기 분석")

# 2. AI 프로세서 초기화 (세션 상태 유지)
if 'processor' not in st.session_state:
    with st.spinner("AI 모델 로딩 중..."):
        st.session_state.processor = VideoProcessor()

# 3. 사이드바 구성
st.sidebar.header("설정")
input_source = st.sidebar.radio("영상 소스 선택", ["YouTube 링크", "파일 업로드"])
process_speed = st.sidebar.slider("분석 정밀도 (프레임 건너뛰기)", 1, 10, 2)

video_path = None

# 4. 영상 소스 처리
if input_source == "YouTube 링크":
    url = st.text_input("유튜브 URL을 입력하세요")
    if url:
        with st.spinner("유튜브 영상을 최적화하여 불러오는 중..."):
            try:
                # 메모리 절약을 위해 360p 이하 mp4 강제 선택
                ydl_opts = {
                    'format': 'worst[ext=mp4]/bestvideo[height<=360][ext=mp4]',
                    'outtmpl': '/tmp/temp_video.%(ext)s',
                    'quiet': True,
                    'no_warnings': True
                }
                with YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    video_path = ydl.prepare_filename(info)
            except Exception as e:
                st.error(f"유튜브 영상을 불러오지 못했습니다: {e}")

else:
    uploaded_file = st.file_uploader("영상을 업로드하세요", type=["mp4", "mov"])
    if uploaded_file:
        # 업로드 파일도 /tmp 디렉토리에 임시 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir='/tmp') as tfile:
            tfile.write(uploaded_file.read())
            video_path = tfile.name

# 5. 분석 실행 엔진
if video_path:
    cap = cv2.VideoCapture(video_path)
    st_frame = st.empty()
    st_info = st.empty()
    
    stop_btn = st.button("분석 중지")
    
    frame_count = 0
    
    while cap.isOpened():
        if stop_btn:
            st.warning("분석이 사용자에 의해 중단되었습니다.")
            break
            
        ret, frame = cap.read()
        if not ret:
            break
        
        # 성능 최적화: 설정한 값에 따라 프레임 건너뛰기
        frame_count += 1
        if frame_count % process_speed != 0:
            continue
            
        # 성능 최적화: 해상도 강제 축소 (서버 메모리 보호)
        frame = cv2.resize(frame, (640, 360))
        
        # AI 처리 (Pose + Tracking)
        try:
            annotated_frame, current_ids = st.session_state.processor.process_frame(frame)
            
            # 화면 출력
            st_frame.image(annotated_frame, channels="BGR", use_container_width=True)
            st_info.caption(f"분석 중... 현재 감지된 선수 ID: {list(current_ids)}")
        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")
            break

    cap.release()
    
    # 사용 완료된 임시 파일 삭제 (용량 확보)
    if os.path.exists(video_path):
        try:
            os.remove(video_path)
        except:
            pass
            
    st.success("모든 분석이 완료되었습니다!")
