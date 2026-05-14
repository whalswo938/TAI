import streamlit as st
import cv2
import tempfile
import os
import time
from yt_dlp import YoutubeDL
from processor import VideoProcessor

# Ultralytics 캐시 디렉토리 설정 (권한 문제 해결)
os.environ['YOLO_CONFIG_DIR'] = '/tmp/ultralytics_settings'
os.environ['ULTRALYTICS_CONFIG_DIR'] = '/tmp/ultralytics_config'

st.set_page_config(page_title="TacticalEye AI", layout="wide")

## ⚽ TacticalEye: 분석 시스템

# 프로세서 초기화 (에러 핸들링 추가)
if 'processor' not in st.session_state:
    try:
        with st.spinner("AI 모델(YOLOv8)을 초기화 중입니다..."):
            st.session_state.processor = VideoProcessor()
            st.success("모델 로드 성공!")
    except Exception as e:
        st.error(f"모델 로드 중 치명적 오류: {e}")
        st.stop()

# 사이드바 설정
st.sidebar.header("Control Panel")
input_source = st.sidebar.radio("소스 선택", ["YouTube 링크", "파일 업로드"])
skip_frames = st.sidebar.slider("프레임 스킵 (높을수록 빠름)", 1, 30, 5)

video_path = None

if input_source == "YouTube 링크":
    url = st.text_input("유튜브 URL (예: https://www.youtube.com/watch?v=...)")
    if url:
        with st.spinner("유튜브 서버에서 영상 추출 중..."):
            try:
                # /tmp 폴더를 명시적으로 사용
                save_path = f"/tmp/video_{int(time.time())}.mp4"
                ydl_opts = {
                    'format': 'best[height<=360][ext=mp4]/worst[ext=mp4]', # 최대한 저화질
                    'outtmpl': save_path,
                    'quiet': True,
                }
                with YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                    video_path = save_path
            except Exception as e:
                st.error(f"유튜브 다운로드 실패: {e}")

else:
    uploaded_file = st.file_uploader("영상 파일 업로드", type=["mp4", "mov"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir='/tmp') as tfile:
            tfile.write(uploaded_file.read())
            video_path = tfile.name

# 실행 엔진
if video_path:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("영상을 열 수 없습니다.")
    else:
        st_frame = st.empty()
        st_text = st.empty()
        stop_button = st.button("분석 중단")

        frame_idx = 0
        try:
            while cap.isOpened():
                if stop_button:
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 성능 최적화: 프레임 스킵
                frame_idx += 1
                if frame_idx % skip_frames != 0:
                    continue

                # 해상도 축소 (CPU 부하 감소)
                frame = cv2.resize(frame, (480, 270)) 
                
                # 분석
                annotated_frame, current_ids = st.session_state.processor.process_frame(frame)
                
                # 출력
                st_frame.image(annotated_frame, channels="BGR", use_container_width=True)
                st_text.info(f"분석 중... 감지된 ID 수: {len(current_ids)}")

        except Exception as e:
            st.error(f"분석 도중 오류 발생: {e}")
        finally:
            cap.release()
            if os.path.exists(video_path):
                os.remove(video_path)
            st.write("분석이 종료되었습니다.")
