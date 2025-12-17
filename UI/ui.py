print("### NEW UI.PY LOADED ###")


import gradio as gr
from ultralytics import YOLO
import json
import os
from PIL import Image

# -------------------------------------------------
# 1) YOLO MODEL LOAD
# -------------------------------------------------
MODEL_PATH = "/home/injaejung/canon/yolo/runs/segment/canon_yolo11m_1203/weights/best.pt"
LAYOUT_STATS_PATH = "/home/injaejung/canon/yolo/pass_data_statitics/layout_stats.json"

model = YOLO(MODEL_PATH)

# -------------------------------------------------
# CASE REQUIRED
# -------------------------------------------------
CASE_REQUIRED = {
    1: {"home_eng", "monitor_eng", "back_eng",
        "home_window", "monitor_window", "back_window",
        "home_button", "monitor_button", "back_button", "screen"},
    2: {"home_eng", "monitor_eng", "id_eng",
        "home_window", "monitor_window", "id_window",
        "home_button", "monitor_button", "id_button", "screen"},
    3: {"home_chi_1", "monitor_chi_1", "back_chi",
        "home_window", "monitor_window", "back_window",
        "home_button", "monitor_button", "back_button", "screen"},
    4: {"home_chi_1", "monitor_chi_1", "id_chi",
        "home_window", "monitor_window", "id_window",
        "home_button", "monitor_button", "id_button", "screen"},
    5: {"home_chi_2", "monitor_chi_2", "back_chi",
        "home_window", "monitor_window", "back_window",
        "home_button", "monitor_button", "back_button", "screen"},
    6: {"home_jap", "monitor_jap", "back_jap",
        "home_window", "monitor_window", "back_window",
        "home_button", "monitor_button", "back_button", "screen"},
    7: {"home_kor", "monitor_kor", "back_kor",
        "home_window", "monitor_window", "back_window",
        "home_button", "monitor_button", "back_button", "screen"},
    8: {"home_window", "monitor_window", "back_window",
        "home_button", "monitor_button", "back_button", "screen"},
}

# -------------------------------------------------
# UTIL FUNCTIONS
# -------------------------------------------------
def classify_case(present_labels):
    matched = []
    for case_id, required in CASE_REQUIRED.items():
        if required.issubset(present_labels):
            matched.append(case_id)
    return sorted(matched)[0] if matched else None


def collect_boxes(res):
    boxes = {}
    if res.boxes is None or res.boxes.cls is None:
        return boxes

    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    xyxy = res.boxes.xyxy.cpu().numpy()

    for cid, box in zip(cls_ids, xyxy):
        label = res.names[int(cid)]
        boxes.setdefault(label, []).append(box.tolist())

    return boxes


def relative_box(window_box, button_box):
    wx1, wy1, wx2, wy2 = window_box
    bx1, by1, bx2, by2 = button_box
    w = max(wx2 - wx1, 1e-6)
    h = max(wy2 - wy1, 1e-6)

    return {
        "center_x": ((bx1 + bx2) / 2 - wx1) / w,
        "center_y": ((by1 + by2) / 2 - wy1) / h,
        "rel_width": (bx2 - bx1) / w,
        "rel_height": (by2 - by1) / h,
    }


def union_box(boxes):
    xs1 = min(b[0] for b in boxes)
    ys1 = min(b[1] for b in boxes)
    xs2 = max(b[2] for b in boxes)
    ys2 = max(b[3] for b in boxes)
    return [xs1, ys1, xs2, ys2]


def compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    if area_a + area_b - inter <= 0:
        return 0
    return inter / (area_a + area_b - inter)


def load_layout_stats():
    if os.path.exists(LAYOUT_STATS_PATH):
        with open(LAYOUT_STATS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# -------------------------------------------------
# 단일 이미지 분석
# -------------------------------------------------
def analyze_single(img, tol_k, check_wh, hand_iou_th):

    res = model.predict(img, task="segment", imgsz=1024, verbose=False)[0]

    # 🔒 안전 가드 (바운딩박스 핵심 수정)
    if res.boxes is None or len(res.boxes) == 0:
        plotted_img = img.copy()
    else:
        plotted_img = Image.fromarray(res.plot())

    boxes = collect_boxes(res)
    labels = set(boxes.keys())

    case_id = classify_case(labels)
    if case_id is None:
        return plotted_img, "❌ FAIL", "-", "CASE 분류 실패"

    stats = load_layout_stats()

    layout_ok = True
    if stats and str(case_id) in stats:
        for w_lbl, btns in stats[str(case_id)].items():
            if w_lbl not in boxes:
                layout_ok = False
                break
            for b_lbl, s in btns.items():
                if b_lbl not in boxes:
                    layout_ok = False
                    break
                rel = relative_box(boxes[w_lbl][0], boxes[b_lbl][0])
                keys = ["center_x", "center_y"]
                if check_wh:
                    keys += ["rel_width", "rel_height"]
                for key in keys:
                    mean = s["mean"][key]
                    std = s["std"][key]
                    tol = max(0.01, std * tol_k)
                    if abs(rel[key] - mean) > tol:
                        layout_ok = False

    hand_fail = False
    if "hand" in boxes:
        screen_box = union_box(boxes["screen"]) if "screen" in boxes else [0, 0, img.width, img.height]
        for h in boxes["hand"]:
            if compute_iou(screen_box, h) >= hand_iou_th:
                hand_fail = True
                break

    if layout_ok and not hand_fail:
        return plotted_img, "✅ PASS", f"CASE {case_id}", "정상"
    else:
        reason = []
        if not layout_ok:
            reason.append("버튼 위치/크기 불일치")
        if hand_fail:
            reason.append("손 가림")
        return plotted_img, "❌ FAIL", f"CASE {case_id}", ", ".join(reason)


# -----------------------------------------

from PIL import Image

def normalize_gallery_images(img_list):
    """
    Gradio Gallery 입력을 List[PIL.Image] 로 변환
    """
    if img_list is None:
        return []

    clean_imgs = []

    for item in img_list:
        if item is None:
            continue

        # Gallery dict 형태
        if isinstance(item, dict):
            if "data" in item and item["data"] is not None:
                clean_imgs.append(item["data"])

        # 이미 PIL.Image 인 경우
        elif isinstance(item, Image.Image):
            clean_imgs.append(item)

    return clean_imgs



# -------------------------------------------------
# 여러 장 이미지 분석
# -------------------------------------------------
def analyze_multi(files, tol_k, check_wh, hand_iou_th):

    if files is None or len(files) == 0:
        return [], []

    # 🔥 File → PIL.Image 변환
    img_list = [Image.open(f.name).convert("RGB") for f in files]

    print("DEBUG image count:", len(img_list))


    stats = load_layout_stats()
    plotted_images = []
    table_rows = []

    for img in img_list:

        # YOLO inference
        res = model.predict(img, task="segment", imgsz=1024, verbose=False)[0]

        # 🔹 Bounding Box 시각화
        if res.boxes is None or len(res.boxes) == 0:
            plotted_images.append(img)
        else:
            plotted_images.append(Image.fromarray(res.plot()))

        boxes = collect_boxes(res)
        labels = set(boxes.keys())

        # CASE 분류
        case_id = classify_case(labels)
        if case_id is None:
            table_rows.append(["FAIL", "-", "CASE 분류 실패"])
            continue

        # 버튼 layout 검사
        layout_ok = True
        if stats and str(case_id) in stats:
            for w_lbl, btns in stats[str(case_id)].items():
                if w_lbl not in boxes:
                    layout_ok = False
                    break
                for b_lbl, s in btns.items():
                    if b_lbl not in boxes:
                        layout_ok = False
                        break

                    rel = relative_box(boxes[w_lbl][0], boxes[b_lbl][0])
                    keys = ["center_x", "center_y"]
                    if check_wh:
                        keys += ["rel_width", "rel_height"]

                    for key in keys:
                        mean = s["mean"][key]
                        std = s["std"][key]
                        tol = max(0.01, std * tol_k)
                        if abs(rel[key] - mean) > tol:
                            layout_ok = False

        # 손 가림 검사
        hand_fail = False
        if "hand" in boxes:
            screen_box = union_box(boxes["screen"]) if "screen" in boxes else [0, 0, img.width, img.height]
            for h in boxes["hand"]:
                if compute_iou(screen_box, h) >= hand_iou_th:
                    hand_fail = True
                    break

        # 최종 결과
        if layout_ok and not hand_fail:
            table_rows.append(["PASS", f"CASE {case_id}", "정상"])
        else:
            reason = []
            if not layout_ok:
                reason.append("버튼 위치/크기 불일치")
            if hand_fail:
                reason.append("손 가림")
            table_rows.append(["FAIL", f"CASE {case_id}", ", ".join(reason)])

    return plotted_images, table_rows



# -------------------------------------------------
# UI
# -------------------------------------------------
with gr.Blocks(title="조작부 자동 검사 시스템") as demo:

    gr.Markdown("## 📘 조작부 자동 검사 시스템")

    # =========================
    # 단일 이미지 분석 TAB
    # =========================
    with gr.Tab("단일 이미지 분석"):
        with gr.Row():

            # -------- 왼쪽: 입력 --------
            with gr.Column(scale=1):
                img_single = gr.Image(type="pil", label="📷 사진 업로드")

                tol_k_s = gr.Slider(
                    1, 6, value=3, step=0.5,
                    label="버튼 위치 허용 범위 (k × STD)"
                )
                check_wh_s = gr.Checkbox(
                    value=True,
                    label="버튼 크기 검사 포함"
                )
                hand_iou_s = gr.Slider(
                    0.0, 0.5, value=0.15, step=0.01,
                    label="손 가림 IoU 임계값"
                )

                btn_single = gr.Button("🔍 분석하기", variant="primary")

            # -------- 오른쪽: 결과 --------
            with gr.Column(scale=1):
                out_img = gr.Image(label="🔍 Bounding Box 결과")

                pf = gr.Textbox(label="PASS / FAIL")
                cs = gr.Textbox(label="CASE")
                rs = gr.Textbox(label="설명")

        btn_single.click(
            fn=analyze_single,
            inputs=[img_single, tol_k_s, check_wh_s, hand_iou_s],
            outputs=[out_img, pf, cs, rs]
        )

    # =========================
    # 여러 장 자동 분석 TAB
    # =========================
    with gr.Tab("여러 장 자동 분석"):
        with gr.Row():

            # -------- 왼쪽: 입력 --------
            with gr.Column(scale=1):
                img_files = gr.File(
                label="📂 사진 여러 장 업로드",
                file_types=["image"],
                file_count="multiple"
                )


                tol_k_m = gr.Slider(
                    1, 6, value=3, step=0.5,
                    label="버튼 위치 허용 범위 (k × STD)"
                )
                check_wh_m = gr.Checkbox(
                    value=True,
                    label="버튼 크기 검사 포함"
                )
                hand_iou_m = gr.Slider(
                    0.0, 0.5, value=0.15, step=0.01,
                    label="손 가림 IoU 임계값"
                )

                btn_multi = gr.Button("📂 여러 장 분석하기", variant="primary")

            # -------- 오른쪽: 결과 --------
            with gr.Column(scale=1):
                out_gallery = gr.Gallery(
                    label="🔍 Bounding Box 결과",
                    columns=4,
                    height=400,
                    allow_preview=True
                )

                out_table = gr.Dataframe(
                    headers=["PASS/FAIL", "CASE", "설명"],
                    label="검사 결과 표"
                )

        btn_multi.click(
            fn=analyze_multi,
            inputs=[img_files, tol_k_m, check_wh_m, hand_iou_m],
            outputs=[out_gallery, out_table]
        )

demo.launch()
