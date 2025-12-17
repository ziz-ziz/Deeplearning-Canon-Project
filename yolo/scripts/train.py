# scripts/train.py
from ultralytics import YOLO

def main():
    # YOLOv11 m-size segmentation 모델 불러오기
    model = YOLO("yolo11m-seg.pt")

    model.train(
        task="segment",
        data="yolo/configs/canon_seg.yml",  # 확장자 .yml/.yaml 실제 파일이랑 맞춰주세요
        imgsz=1024,
        epochs=100,
        batch=8,
        device=1,                      # A5000이 0번 GPU라고 가정 (사용할 번호 쓰면 되요)
        project="yolo/runs/segment",        # 결과 저장 루트
        name="canon_yolo11m_1203",     # exp 대신 폴더 이름 고정
        workers=4                      # dataloader worker 수 (필요하면 줄이기)
    )

if __name__ == "__main__":
    main()
