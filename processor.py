# processor.py 예시
from ultralytics import YOLO
import os

class VideoProcessor:
    def __init__(self):
        self.model = None

    def process_frame(self, frame):
        # 모델이 없을 때만 로드 (최초 1회)
        if self.model is None:
            # .pt 파일을 미리 레포에 업로드해두면 다운로드 과정을 생략해 안정적입니다.
            self.model = YOLO('yolov8n-pose.pt') 
            
        # 추론 시 half=True(소수점 정밀도 낮춤)를 사용하여 메모리 사용량 절감
        results = self.model(frame, verbose=False, half=True)
        # ... 이후 처리 로직
