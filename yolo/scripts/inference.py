import os
import re
import json

from ultralytics import YOLO
import torch  # hand mask 계산용
from typing import Dict, List


# ==========================
# 경로 설정
# ==========================
MODEL_PATH = "/home/injaejung/canon/yolo/runs/segment/canon_yolo11m_1203/weights/best.pt"

SOURCE = "/home/injaejung/canon/yolo/data/images/val"  # 혹은 pass 이미지 폴더
PROJECT = "/home/injaejung/canon/yolo/val_inference_results"
NAME = "canon_yolo11m_1213_val"

# 위치/크기 검증용 (PASS 데이터 기반 통계)
LAYOUT_STATS_PATH = "/home/injaejung/canon/yolo/pass_data_statitics/layout_stats.json"

# 🔹 결과 로그 저장 폴더
LOG_DIR = "/home/injaejung/canon/yolo/val_inference_results/result_logs"
os.makedirs(LOG_DIR, exist_ok=True)


def get_log_file_path(name: str) -> str:
    return os.path.join(LOG_DIR, f"{name}.txt")


# ==========================
# case 정의 (필수 라벨 집합)
# ==========================
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

# window / button 라벨셋 (hand가 가렸는지 확인용)
WINDOW_LABELS = {
    "home_window", "back_window", "id_window", "monitor_window",
}
BUTTON_LABELS = {
    "home_button", "back_button", "id_button", "monitor_button",
}

# 텍스트(문자) 라벨들 (hand가 이 bbox를 가려도 FAIL)
TEXT_LABELS = {
    # 영어
    "home_eng", "monitor_eng", "back_eng", "id_eng",
    # 중국어 1차
    "home_chi_1", "monitor_chi_1", "back_chi", "id_chi",
    # 중국어 2차
    "home_chi_2", "monitor_chi_2",
    # 일본어
    "home_jap", "monitor_jap", "back_jap",
    # 한국어
    "home_kor", "monitor_kor", "back_kor",
}

# 면적 판정에 사용할 전체 라벨 셋
AREA_LABELS = WINDOW_LABELS | BUTTON_LABELS | TEXT_LABELS


# ==========================
# 후처리 파라미터 기본값
# ==========================
BUTTON_TOL_STD = 3.0        # 버튼 위치 허용 범위: mean ± k*std 의 k 값
USE_WIDTH_HEIGHT = True     # 버튼 크기(rel_width, rel_height)도 검사할지 여부

HAND_IOU_THRESHOLD = 0.1      # hand vs screen mask IoU threshold
HAND_UI_IOU_THRESHOLD = 0.1  # hand vs window/button/text bbox IoU threshold

# 🔹 PASS 통계 기준으로, 라벨별 mask 면적이 평균 대비 얼마나 작으면 이상(가려짐)으로 볼지
AREA_OCCLUSION_RATIO = 0.6   # PASS 평균 면적 비율의 60%보다 작으면 이상치로 간주


# ==========================
# 유틸 함수들
# ==========================
def parse_gt_case_from_filename(path: str):
    """
    파일명에서 GT case 번호 추출.
    pass/fail 둘 다 지원.
    예) pass_case_5_211.jpeg, fail_1_case_5_211.jpeg -> 5 (int)
    실패 시 None
    """
    base = os.path.basename(path)
    m = re.search(r"case_(\d+)_", base)
    if m:
        return int(m.group(1))
    return None


def parse_gt_passfail_from_filename(path: str):
    """
    파일명에서 GT PASS/FAIL 추출.
    예) pass_case_1_101.jpeg -> 'PASS'
        fail_1_case_1_101.jpeg -> 'FAIL'
    실패 시 None
    """
    base = os.path.basename(path)
    m = re.match(r"(pass|fail)_", base)
    if m:
        return m.group(1).upper()  # 'PASS' or 'FAIL'
    return None


def classify_case(present_labels: set) -> int | None:
    """
    세그멘테이션에서 검출된 라벨 이름 집합으로 case 분류.
    required set ⊆ present_labels 를 만족하는 case 를 리턴.
    여러 개 매칭되면 case 번호가 작은 것 우선.
    하나도 없으면 None.
    """
    matched_cases = []
    for case_id, required in CASE_REQUIRED.items():
        if required.issubset(present_labels):
            matched_cases.append(case_id)

    if not matched_cases:
        return None

    return sorted(matched_cases)[0]


def collect_boxes_by_label(res) -> Dict[str, List[List[float]]]:
    """
    YOLO 결과(res)에서 라벨별 bounding box 리스트를 딕셔너리로 반환.
    { "home_window": [[x1, y1, x2, y2], ...], ... }
    """
    boxes_by_label: Dict[str, List[List[float]]] = {}

    if res.boxes is None or res.boxes.cls is None:
        return boxes_by_label

    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    xyxy = res.boxes.xyxy.cpu().numpy()  # shape: (N, 4)

    for cid, box in zip(cls_ids, xyxy):
        label_name = res.names[int(cid)]
        boxes_by_label.setdefault(label_name, []).append(box.tolist())

    return boxes_by_label


def relative_box_to_window(window_box, button_box):
    """
    window_box, button_box: [x1, y1, x2, y2]
    window 기준으로 버튼의 상대 중심/크기를 0~1 스케일로 환산.
    """
    wx1, wy1, wx2, wy2 = window_box
    bx1, by1, bx2, by2 = button_box

    w = max(wx2 - wx1, 1e-6)
    h = max(wy2 - wy1, 1e-6)

    center_x = ((bx1 + bx2) / 2.0 - wx1) / w
    center_y = ((by1 + by2) / 2.0 - wy1) / h
    rel_width = (bx2 - bx1) / w
    rel_height = (by2 - by1) / h

    return {
        "center_x": center_x,
        "center_y": center_y,
        "rel_width": rel_width,
        "rel_height": rel_height,
    }


def check_button_layout(case_id: int,
                        layout_stats: dict | None,
                        boxes_by_label: Dict[str, List[List[float]]],
                        tol_std_factor: float = BUTTON_TOL_STD,
                        use_wh: bool = USE_WIDTH_HEIGHT):
    """
    layout_stats.json 기반으로, 각 window-button 쌍의 상대 좌표/크기가
    정상 범위( mean ± k*std ) 안에 있는지 검사.

    면적(occlusion)은 여기서 판단하지 않고,
    mask 기반 별도 함수에서 처리한다.
    """
    case_key = str(case_id)
    if layout_stats is None or case_key not in layout_stats:
        return True, []

    case_stats = layout_stats[case_key]
    layout_ok = True
    details: list[dict] = []

    for window_label, button_dict in case_stats.items():
        for button_label, stats in button_dict.items():
            mean = stats["mean"]
            std = stats["std"]

            win_boxes = boxes_by_label.get(window_label)
            btn_boxes = boxes_by_label.get(button_label)

            if not win_boxes or not btn_boxes:
                layout_ok = False
                details.append({
                    "window": window_label,
                    "button": button_label,
                    "status": "FAIL",
                    "reason": "missing detection (window/button 미검출)",
                })
                continue

            rel = relative_box_to_window(win_boxes[0], btn_boxes[0])

            checks = {}
            keys = ["center_x", "center_y"]
            if use_wh:
                keys += ["rel_width", "rel_height"]

            for key in keys:
                m = mean[key]
                s = std[key]

                # # ⭐️⭐️⭐️⭐️⭐️⭐️
                # MIN_STD = 0.01  # 의미 있는 최소 분산 (relative scale)

                # if s < MIN_STD:
                #     tol = MIN_STD
                # else:
                #     tol = tol_std_factor * s
                # # ⭐️⭐️⭐️⭐️⭐️⭐️
                
                if s < 1e-6:
                    tol = 0.01
                else:
                    tol = tol_std_factor * s

                diff = abs(rel[key] - m)
                ok = diff <= tol

                checks[key] = {
                    "value": rel[key],
                    "mean": m,
                    "std": s,
                    "diff": diff,
                    "tol": tol,
                    "ok": ok,
                }

            pair_ok = all(v["ok"] for v in checks.values())
            if not pair_ok:
                layout_ok = False

            details.append({
                "window": window_label,
                "button": button_label,
                "status": "PASS" if pair_ok else "FAIL",
                "metrics": checks,
            })

    return layout_ok, details


def union_boxes(boxes: List[List[float]]) -> List[float]:
    xs1 = min(b[0] for b in boxes)
    ys1 = min(b[1] for b in boxes)
    xs2 = max(b[2] for b in boxes)
    ys2 = max(b[3] for b in boxes)
    return [xs1, ys1, xs2, ys2]


def build_screen_box(boxes_by_label: Dict[str, List[List[float]]],
                     img_w: int,
                     img_h: int):
    """
    screen 박스가 있으면 그 union,
    없으면 전체 이미지를 screen 박스로 사용.
    (monitor_window 기반으로 바꾸고 싶으면 아래 주석 해제해서 교체 가능)
    """
    if "screen" in boxes_by_label and boxes_by_label["screen"]:
        return union_boxes(boxes_by_label["screen"]), "screen"

    # if "monitor_window" in boxes_by_label and boxes_by_label["monitor_window"]:
    #     return union_boxes(boxes_by_label["monitor_window"]), "monitor_window"

    return [0.0, 0.0, float(img_w), float(img_h)], "image"


def iou(box1: List[float], box2: List[float]) -> float:
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2

    inter_x1 = max(x1, x1b)
    inter_y1 = max(y1, y1b)
    inter_x2 = min(x2, x2b)
    inter_y2 = min(y2, y2b)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area2 = max(0.0, x2b - x1b) * max(0.0, y2b - y1b)

    if area1 <= 0.0 or area2 <= 0.0:
        return 0.0

    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def check_hand_occlusion_screen(res,
                                boxes_by_label: Dict[str, List[List[float]]],
                                iou_threshold: float = HAND_IOU_THRESHOLD):
    """
    hand vs screen 영역 기준으로 occlusion 판단.
    (가능하면 mask 기반, 없으면 bbox 기반)
    """
    img_h, img_w = res.orig_shape[:2]

    # screen 영역(box) 확보
    screen_box, base_label = build_screen_box(boxes_by_label, img_w, img_h)

    # hand 인덱스 찾기 (boxes와 masks는 같은 순서라고 가정)
    hand_indices = []
    if res.boxes is None or res.boxes.cls is None:
        return False, 0.0, None

    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    for idx, cid in enumerate(cls_ids):
        label_name = res.names[int(cid)]
        if label_name == "hand":
            hand_indices.append(idx)

    # hand 자체가 없다면 Fail 아님
    if not hand_indices:
        return False, 0.0, None

    # mask가 없으면 bbox IoU fallback
    if res.masks is None or res.masks.data is None:
        hand_boxes = boxes_by_label.get("hand") or []
        if not hand_boxes:
            return False, 0.0, None
        max_iou_val = 0.0
        for hb in hand_boxes:
            val = iou(screen_box, hb)
            if val > max_iou_val:
                max_iou_val = val
        is_fail = max_iou_val >= iou_threshold
        return is_fail, max_iou_val, base_label

    # mask 기반 계산
    masks = res.masks.data  # shape: (N, H, W)
    _, h, w = masks.shape
    device = masks.device

    # screen rect를 마스크로 변환
    screen_mask = torch.zeros((h, w), dtype=torch.bool, device=device)

    sx1, sy1, sx2, sy2 = screen_box
    sx1 = max(0, min(int(sx1), w))
    sx2 = max(0, min(int(sx2), w))
    sy1 = max(0, min(int(sy1), h))
    sy2 = max(0, min(int(sy2), h))

    if sx2 > sx1 and sy2 > sy1:
        screen_mask[sy1:sy2, sx1:sx2] = True

    screen_area = screen_mask.sum().item()
    if screen_area <= 0:
        return False, 0.0, base_label

    max_iou_val = 0.0

    for idx in hand_indices:
        hand_mask = masks[idx] > 0.5
        hand_area = hand_mask.sum().item()
        if hand_area <= 0:
            continue

        inter = hand_mask & screen_mask
        inter_area = inter.sum().item()

        union_area = hand_area + screen_area - inter_area
        if union_area <= 0:
            iou_val = 0.0
        else:
            iou_val = inter_area / union_area

        if iou_val > max_iou_val:
            max_iou_val = iou_val

    is_fail = max_iou_val >= iou_threshold
    return is_fail, max_iou_val, base_label


def check_hand_occlusion_ui(boxes_by_label: Dict[str, List[List[float]]],
                            iou_threshold: float = HAND_UI_IOU_THRESHOLD):
    """
    hand bbox가 window/button/텍스트 bbox를 가렸는지 여부를 검사.
    - 단순 bbox IoU 사용
    반환:
      (is_fail: bool, occluded_labels: List[str])
    """
    hand_boxes = boxes_by_label.get("hand") or []
    if not hand_boxes:
        return False, []

    # window + button + text 라벨 전체를 occlusion 대상으로 사용
    target_labels = sorted(list(WINDOW_LABELS | BUTTON_LABELS | TEXT_LABELS))
    occluded_labels = set()

    for lbl in target_labels:
        ui_boxes = boxes_by_label.get(lbl) or []
        for ub in ui_boxes:
            for hb in hand_boxes:
                val = iou(ub, hb)
                if val >= iou_threshold:
                    occluded_labels.add(lbl)
                    break

    return (len(occluded_labels) > 0), sorted(list(occluded_labels))


def build_mask_area_stats(results) -> Dict[str, Dict[str, float]]:
    """
    PASS 이미지들에 대해, 라벨별 mask 면적 비율 통계를 계산.
    - 대상 라벨: AREA_LABELS (window/button/text)
    - 면적 비율: mask 픽셀 수 / 전체 이미지 픽셀 수

    반환:
      { label_name: {"mean": float, "std": float, "n": int}, ... }
    """
    accum: Dict[str, Dict[str, float]] = {}

    for res in results:
        img_path = res.path
        gt_final = parse_gt_passfail_from_filename(img_path)
        if gt_final != "PASS":
            continue

        if res.masks is None or res.masks.data is None:
            continue
        if res.boxes is None or res.boxes.cls is None:
            continue

        masks = res.masks.data  # (N, H, W)
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        _, h, w = masks.shape
        total_pixels = float(h * w)

        for idx, cid in enumerate(cls_ids):
            label = res.names[int(cid)]
            if label not in AREA_LABELS:
                continue

            mask = masks[idx] > 0.5
            area = mask.sum().item()
            if area <= 0:
                continue

            ratio = area / total_pixels

            if label not in accum:
                accum[label] = {"sum": 0.0, "sum2": 0.0, "n": 0}
            accum[label]["sum"] += ratio
            accum[label]["sum2"] += ratio * ratio
            accum[label]["n"] += 1

    area_stats: Dict[str, Dict[str, float]] = {}
    for label, s in accum.items():
        n = s["n"]
        if n <= 0:
            continue
        mean = s["sum"] / n
        mean2 = s["sum2"] / n
        var = max(0.0, mean2 - mean * mean)
        std = var ** 0.5
        area_stats[label] = {"mean": mean, "std": std, "n": n}

    return area_stats


def check_button_area_by_mask(res,
                              area_stats: Dict[str, Dict[str, float]],
                              ratio_threshold: float = AREA_OCCLUSION_RATIO):
    """
    mask 기반으로 라벨별 면적이 PASS 통계 대비 얼마나 줄어들었는지 검사.
    - 대상 라벨: AREA_LABELS (window/button/text)
    - 면적 비율: mask 픽셀 수 / 전체 이미지 픽셀 수

    반환:
      (area_ok: bool,
       details: Dict[str, Dict[str, float]])
        details[label] = {
            "current": 현재 면적 비율,
            "mean": PASS 평균 면적 비율,
            "ratio": current / mean
        }
    """
    if not area_stats:
        return True, {}

    if res.masks is None or res.masks.data is None:
        return True, {}

    if res.boxes is None or res.boxes.cls is None:
        return True, {}

    masks = res.masks.data  # (N, H, W)
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    _, h, w = masks.shape
    total_pixels = float(h * w)

    area_ok = True
    details: Dict[str, Dict[str, float]] = {}
    visited_labels = set()

    for idx, cid in enumerate(cls_ids):
        label = res.names[int(cid)]
        if label not in AREA_LABELS:
            continue

        # 한 이미지에 같은 라벨이 여러 번 있을 가능성은 거의 없지만,
        # 혹시를 대비해서 첫 번째만 사용.
        if label in visited_labels:
            continue

        if label not in area_stats:
            continue

        mask = masks[idx] > 0.5
        area = mask.sum().item()
        if area <= 0:
            continue

        cur_ratio = area / total_pixels
        mean = area_stats[label]["mean"]
        if mean <= 0:
            continue

        ratio = cur_ratio / mean

        details[label] = {
            "current": cur_ratio,
            "mean": mean,
            "ratio": ratio,
        }

        if ratio < ratio_threshold:
            area_ok = False

        visited_labels.add(label)

    return area_ok, details


# ==========================
# main
# ==========================
def main():
    global BUTTON_TOL_STD

    model = YOLO(MODEL_PATH)

    # PASS 데이터 기반 레이아웃 통계 로드 (bbox 기반 위치 검증 용도)
    if os.path.exists(LAYOUT_STATS_PATH):
        with open(LAYOUT_STATS_PATH, "r", encoding="utf-8") as f:
            layout_stats = json.load(f)
    else:
        layout_stats = None

    try:
        user_k = input(
            f"Button layout tolerance (k for mean±k*std, default {BUTTON_TOL_STD}): "
        ).strip()
        if user_k:
            BUTTON_TOL_STD = float(user_k)
    except Exception:
        pass

    # 1차 추론 (val 전체) 수행
    results = model.predict(
        task="segment",
        source=SOURCE,
        imgsz=1024,
        device=0,
        save=True,
        project=PROJECT,
        name=NAME,
        verbose=False,
    )

    # PASS 이미지들에 대해 mask 기반 라벨별 면적 통계 계산
    area_stats = build_mask_area_stats(results)
    print(f"mask 기반 PASS 통계가 계산된 라벨 수: {len(area_stats)}")

    total = 0

    # case 정확도 (pass + fail 모두 포함)
    total_case_gt = 0
    correct_case = 0

    # 최종 PASS/FAIL 판정 정확도
    total_final_gt = 0
    correct_final = 0

    # 통계 카운터
    button_layout_fail = 0
    button_area_fail = 0

    hand_screen_fail = 0
    hand_ui_fail = 0
    
    pred_case_none_fail = 0  # pred_case가 None이어서 FAIL된 경우

    final_pass = 0
    final_fail = 0

    # GT vs Pred 불일치 카운터
    gt_pass_pred_fail = 0  # GT는 PASS인데 FAIL로 예측
    gt_fail_pred_pass = 0  # GT는 FAIL인데 PASS로 예측
    mismatched_images: List[str] = []  # 불일치한 이미지 목록

    # case check만 fail인 이미지 카운터 및 로그
    case_only_fail_count = 0
    case_only_fail_log_lines: List[str] = []

    # 전체 로그
    all_log_lines: List[str] = []

    print("\n================ INFERENCE & CASE / POST-PROCESS CHECK ================\n")

    for res in results:
        img_path = res.path
        base = os.path.basename(img_path)

        boxes_by_label = collect_boxes_by_label(res)
        present_labels = set(boxes_by_label.keys())

        pred_case = classify_case(present_labels)
        gt_case = parse_gt_case_from_filename(img_path)
        gt_final = parse_gt_passfail_from_filename(img_path)

        total += 1

        # case 정확도
        if gt_case is not None:
            total_case_gt += 1
            if pred_case == gt_case:
                correct_case += 1

        # 버튼 레이아웃 검사 (bbox 기반 위치)
        if pred_case is not None and layout_stats is not None:
            layout_ok, layout_details = check_button_layout(
                pred_case,
                layout_stats,
                boxes_by_label,
                tol_std_factor=BUTTON_TOL_STD,
                use_wh=USE_WIDTH_HEIGHT,
            )
        else:
            layout_ok, layout_details = True, []

        if not layout_ok:
            button_layout_fail += 1

        # 버튼/창/텍스트 mask 면적 검사 (PASS 통계 vs 현재 mask 면적)
        area_ok, area_details = check_button_area_by_mask(
            res,
            area_stats,
            ratio_threshold=AREA_OCCLUSION_RATIO,
        )
        if not area_ok:
            button_area_fail += 1

        # hand occlusion (screen)
        hand_screen_is_fail, hand_screen_iou_val, screen_base = check_hand_occlusion_screen(
            res,
            boxes_by_label,
            iou_threshold=HAND_IOU_THRESHOLD,
        )
        if hand_screen_is_fail:
            hand_screen_fail += 1

        # hand occlusion (window/button/text)
        hand_ui_is_fail, occluded_labels = check_hand_occlusion_ui(
            boxes_by_label,
            iou_threshold=HAND_UI_IOU_THRESHOLD,
        )
        if hand_ui_is_fail:
            hand_ui_fail += 1

        # case check flag (GT 기준)
        if gt_case is not None:
            if pred_case == gt_case:
                case_ok = True
            else:
                case_ok = False
        else:
            case_ok = None

        # 최종 PASS/FAIL (생산용 판정)
        if (
            (pred_case is not None) and
            layout_ok and
            area_ok and
            (not hand_screen_is_fail) and
            (not hand_ui_is_fail)
        ):
            final_status = "PASS"
        else:
            final_status = "FAIL"

        if final_status == "PASS":
            final_pass += 1
        else:
            final_fail += 1
            # pred_case가 None인 경우 카운트
            if pred_case is None:
                pred_case_none_fail += 1

        # 최종 판정 정확도 (GT pass/fail 기준)
        if gt_final is not None:
            total_final_gt += 1
            if final_status == gt_final:
                correct_final += 1
            else:
                # GT와 예측이 불일치하는 경우
                mismatched_images.append(base)
                if gt_final == "PASS" and final_status == "FAIL":
                    gt_pass_pred_fail += 1
                elif gt_final == "FAIL" and final_status == "PASS":
                    gt_fail_pred_pass += 1

        # 🔹 "final FAIL + case check만 FAIL"인 이미지 선별
        only_case_fail = (
            final_status == "FAIL" and
            case_ok is False and
            layout_ok and
            area_ok and
            (not hand_screen_is_fail) and
            (not hand_ui_is_fail)
        )

        if only_case_fail:
            case_only_fail_count += 1
            lines: List[str] = []
            lines.append(f"[{base}] case check만 FAIL인 이미지")
            lines.append(f"  GT case : {gt_case}")
            lines.append(f"  Pred case : {pred_case}")

            if area_details:
                lines.append("  라벨별 mask 면적 (cur_ratio vs PASS mean, ratio=cur/mean):")
                for lbl, info in area_details.items():
                    lines.append(
                        f"    - {lbl}: cur={info['current']:.6f}, "
                        f"mean={info['mean']:.6f}, ratio={info['ratio']:.3f}"
                    )
            else:
                lines.append("  (해당 이미지에서 면적 비교에 사용된 라벨이 없었습니다.)")

            lines.append("")  # 구분용 빈 줄
            case_only_fail_log_lines.extend(lines)

        # ================== 콘솔/메인 로그 라인 모으기 ==================
        log_lines: List[str] = []

        # case 문장 (한국어 설명, case 용어는 그대로 유지)
        if pred_case is not None:
            if gt_case is not None:
                line = f"[{base}] 이 이미지는 case {pred_case} (ground truth: case {gt_case}) 로 분류되었습니다."
            else:
                line = f"[{base}] 이 이미지는 case {pred_case} 로 분류되었습니다."
        else:
            if gt_case is not None:
                line = (
                    f"[{base}] 이 이미지는 case를 결정하지 못했습니다 "
                    f"(ground truth: case {gt_case}, present labels: {sorted(present_labels)})."
                )
            else:
                line = (
                    f"[{base}] 이 이미지는 case를 결정하지 못했습니다 "
                    f"(present labels: {sorted(present_labels)})."
                )
        print(line)
        log_lines.append(line)

        # case check 라인 (GT가 있는 경우에만 의미 있음)
        if case_ok is True:
            line = "   - case check: OK (예측 case와 GT case가 일치합니다.)"
        elif case_ok is False:
            if pred_case is None:
                line = "   - case check: FAIL (GT case는 존재하지만, 모델이 case를 결정하지 못했습니다.)"
            else:
                line = "   - case check: FAIL (예측 case와 GT case가 서로 다릅니다.)"
        else:
            line = "   - case check: - (GT case 정보가 없습니다.)"
        print(line)
        log_lines.append(line)

        # 버튼 레이아웃 결과
        if layout_ok:
            line = "   - button layout: OK (버튼/창 위치가 PASS 통계 범위 내에 있습니다.)"
        else:
            line = "   - button layout: FAIL (버튼/창 위치가 PASS 통계를 벗어납니다.)"
        print(line)
        log_lines.append(line)

        # 버튼/창/텍스트 면적 결과 (PASS 통계 기준, mask 기반)
        if area_ok:
            line = (
                "   - button area (mask vs PASS 통계): OK "
                "(버튼/창/텍스트의 mask 면적 비율이 PASS case 통계 기준과 유사합니다.)"
            )
        else:
            line = (
                "   - button area (mask vs PASS 통계): FAIL "
                f"(일부 라벨의 mask 면적 비율이 PASS 평균의 {AREA_OCCLUSION_RATIO:.2f}배보다 작습니다.)"
            )
        print(line)
        log_lines.append(line)

        # hand occlusion (screen)
        if hand_screen_is_fail:
            line = (
                f"   - hand occlusion (screen): FAIL "
                f"(손 mask와 {screen_base} 영역의 IoU = {hand_screen_iou_val:.3f} ≥ threshold {HAND_IOU_THRESHOLD})"
            )
        else:
            line = "   - hand occlusion (screen): OK (손이 screen 영역을 유의미하게 가리지 않습니다.)"
        print(line)
        log_lines.append(line)

        # hand occlusion (icon/window/text)
        if hand_ui_is_fail:
            occ_str = ", ".join(occluded_labels)
            line = (
                f"   - hand occlusion (icon/window): FAIL "
                f"(손 bbox가 다음 UI 라벨 bbox를 가립니다: {occ_str}, IoU ≥ {HAND_UI_IOU_THRESHOLD})"
            )
        else:
            line = "   - hand occlusion (icon/window): OK (손이 window/button/text bbox를 유의미하게 가리지 않습니다.)"
        print(line)
        log_lines.append(line)

        # 최종 결과
        if final_status == "PASS":
            line = "   - FINAL: PASS (버튼 레이아웃, 버튼 면적, hand occlusion 조건을 모두 만족합니다.)"
        else:
            line = "   - FINAL: FAIL (위 조건 중 하나 이상이 기준을 만족하지 못합니다.)"
        print(line)
        log_lines.append(line)

        print()  # 콘솔 줄바꿈

        # 이미지 로그를 전체 로그에 추가
        all_log_lines.extend(log_lines)
        all_log_lines.append("")  # 이미지 간 구분을 위한 빈 줄

    # ---------------- summary ----------------
    summary_lines: List[str] = []
    summary_lines.append("\n================ SUMMARY ================")

    # case 정확도 (pass + fail 모두 포함)
    if total_case_gt > 0:
        case_acc = correct_case / total_case_gt * 100.0
        print("\n================ SUMMARY ================")
        # print(f"GT case가 있는 전체 이미지 수       : {total_case_gt}")
        # print(f"case를 정확히 맞춘 이미지 수        : {correct_case}")
        # print(f"Case Accuracy                        : {case_acc:.2f}%")

        # summary_lines.append(f"GT case가 있는 전체 이미지 수       : {total_case_gt}")
        # summary_lines.append(f"case를 정확히 맞춘 이미지 수        : {correct_case}")
        # summary_lines.append(f"Case Accuracy                        : {case_acc:.2f}%")
    else:
        msg = "No images with ground-truth case pattern (case_X_...)."
        print(f"\n{msg}")
        summary_lines.append(msg)

    # 최종 PASS/FAIL 판정 정확도
    if total_final_gt > 0:
        final_acc = correct_final / total_final_gt * 100.0
        print(f"GT PASS/FAIL 라벨이 있는 이미지 수           : {total_final_gt}")
        print(f"FINAL을 정확히 맞춘 이미지 수                : {correct_final}")
        print(f"Final Decision Accuracy                      : {final_acc:.2f}%")

        summary_lines.append(f"GT PASS/FAIL 라벨이 있는 이미지 수           : {total_final_gt}")
        summary_lines.append(f"FINAL을 정확히 맞춘 이미지 수                : {correct_final}")
        summary_lines.append(f"Final Decision Accuracy                      : {final_acc:.2f}%")
    else:
        msg = "No images with PASS/FAIL ground-truth prefix (pass_ / fail_)."
        print(msg)
        summary_lines.append(msg)

    print(f"Button layout FAIL images                    : {button_layout_fail}")
    print(f"Button area FAIL images (mask vs PASS 통계) : {button_area_fail}")
    print(f"Hand occlusion (screen) FAIL images          : {hand_screen_fail}")
    print(f"Hand occlusion (icon/window) FAIL            : {hand_ui_fail}")
    print(f"Case detection FAIL (pred_case=None)         : {pred_case_none_fail}")
    
    # FAIL 조건 개별 합계 계산
    fail_conditions_sum = button_layout_fail + button_area_fail + hand_screen_fail + hand_ui_fail + pred_case_none_fail
    other_fail = final_fail - fail_conditions_sum if final_fail > fail_conditions_sum else 0
    
    # print(f"Other FAIL (multiple conditions)     : {other_fail}")
    print(f"Final PASS images                            : {final_pass}")
    print(f"Final FAIL images                            : {final_fail}")
    # print(f"case check만 FAIL인 이미지 수         : {case_only_fail_count}")
    print(f"Button layout tolerance (k*std)              : {BUTTON_TOL_STD}")
    print(f"Hand IoU threshold (screen)                  : {HAND_IOU_THRESHOLD}")
    print(f"Hand IoU threshold (icon/window)             : {HAND_UI_IOU_THRESHOLD}")
    # print(f"Area-occlusion ratio (mean*ratio)    : {AREA_OCCLUSION_RATIO}")

    summary_lines.append(f"Button layout FAIL images                    : {button_layout_fail}")
    summary_lines.append(f"Button area FAIL images (mask vs PASS 통계) : {button_area_fail}")
    summary_lines.append(f"Hand occlusion (screen) FAIL images          : {hand_screen_fail}")
    summary_lines.append(f"Hand occlusion (icon/window) FAIL            : {hand_ui_fail}")
    summary_lines.append(f"Case detection FAIL (pred_case=None)         : {pred_case_none_fail}")
    # summary_lines.append(f"Other FAIL (multiple conditions)     : {other_fail}")
    summary_lines.append(f"Final PASS images                            : {final_pass}")
    summary_lines.append(f"Final FAIL images                            : {final_fail}")
    # summary_lines.append(f"case check만 FAIL인 이미지 수         : {case_only_fail_count}")
    summary_lines.append(f"Button layout tolerance (k*std)              : {BUTTON_TOL_STD}")
    summary_lines.append(f"Hand IoU threshold (screen)                  : {HAND_IOU_THRESHOLD}")
    summary_lines.append(f"Hand IoU threshold (icon/window)             : {HAND_UI_IOU_THRESHOLD}")
    # summary_lines.append(f"Area-occlusion ratio (mean*ratio)    : {AREA_OCCLUSION_RATIO}")
    
    # 불일치 통계
    if gt_pass_pred_fail > 0 or gt_fail_pred_pass > 0:
        print(f"\n[Mismatched Predictions]")
        print(f"GT=PASS but Pred=FAIL                : {gt_pass_pred_fail}")
        print(f"GT=FAIL but Pred=PASS                : {gt_fail_pred_pass}")
        print(f"Mismatched images: {', '.join(mismatched_images)}")
        
        summary_lines.append(f"\n[Mismatched Predictions]")
        summary_lines.append(f"GT=PASS but Pred=FAIL                : {gt_pass_pred_fail}")
        summary_lines.append(f"GT=FAIL but Pred=PASS                : {gt_fail_pred_pass}")
        summary_lines.append(f"Mismatched images: {', '.join(mismatched_images)}")

    # 🔹 전체 로그 (이미지 결과 + SUMMARY)를 하나의 파일에 저장
    all_log_lines.extend(summary_lines)
    log_file_path = get_log_file_path(NAME)

    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_log_lines))

    print(f"\n✅ 결과 로그 저장 완료: {log_file_path}")

    # 🔹 case check만 FAIL인 이미지 목록 별도 저장
    if case_only_fail_log_lines:
        case_only_path = os.path.join(LOG_DIR, f"{NAME}_case_only_fail.txt")
        with open(case_only_path, "w", encoding="utf-8") as f:
            f.write("\n".join(case_only_fail_log_lines))
        print(f"📄 case check만 FAIL인 이미지 로그 저장 완료: {case_only_path}")
    else:
        print("📄 case check만 FAIL인 이미지는 없습니다.")

    print("\nDone.")


if __name__ == "__main__":
    main()
