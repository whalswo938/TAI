import streamlit as st
import cv2
import tempfile
import os
from processor import VideoProcessor

st.set_page_config(page_title="TacticalEye MVP", layout="wide")
st.title("⚽ TacticalEye MVP: AI 하이라이트 엔진")

# 세션 상태 초기화 (AI 모델 로드)
if 'processor' not in st.session_state:
    st.session_state.processor = VideoProcessor()

uploaded_file = st.file_uploader("경기 영상을 업로드하세요", type=["mp4", "mov"])

if uploaded_file:
    # 1. 파일 임시 저장
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    # 2. 영상 읽기
    cap = cv2.VideoCapture(tfile.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    st_frame = st.empty() # 영상 출력 화면
    
    # 3. 분석 루프
    stop_btn = st.button("중지")
    
    while cap.isOpened():
        if stop_btn: break
        
        ret, frame = cap.read()
        if not ret: break
        
        # AI 처리 (Pose + Tracking)
        annotated_frame, current_ids = st.session_state.processor.process_frame(frame)
        
        # 화면 업데이트
        st_frame.image(annotated_frame, channels="BGR")
        
        # 하단에 실시간 탐지된 선수 ID 표시
        st.caption(f"현재 탐지된 선수 ID: {list(current_ids)}")

    cap.release()
    st.success("분석이 완료되었습니다.")
