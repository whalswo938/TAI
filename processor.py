import cv2
from ultralytics import YOLO

class VideoProcessor:
    def __init__(self, model_path='yolov8n-pose.pt'):
        # 체형 및 포즈 탐지 모델 로드
        self.model = YOLO(model_path)

    def process_frame(self, frame, conf=0.5):
        # persist=True로 설정해야 프레임 간 고유 ID가 유지됨 (Re-ID 효과)
        results = self.model.track(frame, persist=True, conf=conf, verbose=False)
        
        # 분석 결과 시각화
        annotated_frame = results[0].plot()
        
        # ID 정보 추출 (추후 하이라이트 타겟팅용)
        ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else []
        
        return annotated_frame, ids
