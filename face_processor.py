"""
人脸处理核心模块
- 主检测器：OpenCV FaceDetectorYN (YuNet ONNX) —— 深度学习，对集体照/小脸效果好
- 兜底检测器：Haar Cascade —— 应对模型未命中的情况
- 功能1：单人照片 - 用卡通头像覆盖人脸
- 功能2：集体合影 - 对人脸区域高斯模糊，其余高清
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw
import os
import sys

def get_resource_path(relative_path):
    """获取资源绝对路径，兼容 PyInstaller 的 _MEIPASS"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), relative_path)

# ─── 模型路径 ────────────────────────────────────────────────
_YUNET_PATH = get_resource_path(os.path.join("models", "face_detection_yunet_2023mar.onnx"))
_DATA_DIR   = cv2.data.haarcascades

# ─── Haar Cascade 兜底 ───────────────────────────────────────
_haar_default = cv2.CascadeClassifier(os.path.join(_DATA_DIR, "haarcascade_frontalface_default.xml"))
_haar_alt2    = cv2.CascadeClassifier(os.path.join(_DATA_DIR, "haarcascade_frontalface_alt2.xml"))
_haar_profile = cv2.CascadeClassifier(os.path.join(_DATA_DIR, "haarcascade_profileface.xml"))


# ─── IoU / NMS ───────────────────────────────────────────────
def _iou(a, b):
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0]+a[2], a[1]+a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0]+b[2], b[1]+b[3]
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter == 0:
        return 0.0
    return inter / float(a[2]*a[3] + b[2]*b[3] - inter)


def _nms(boxes, iou_thr=0.35):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
    kept, used = [], [False]*len(boxes)
    for i in range(len(boxes)):
        if used[i]:
            continue
        kept.append(boxes[i])
        for j in range(i+1, len(boxes)):
            if not used[j] and _iou(boxes[i], boxes[j]) > iou_thr:
                used[j] = True
    return kept


# ─── YuNet 检测 ───────────────────────────────────────────────
def _detect_yunet(image_bgr, conf_thr=0.6):
    """使用 YuNet 深度学习模型检测人脸，多尺度以覆盖小脸"""
    if not os.path.exists(_YUNET_PATH):
        return []

    h, w = image_bgr.shape[:2]
    results = []

    # 多尺度：原图 + 放大1.5倍 —— 让小脸也能被检测到
    scales = [1.0]
    if max(h, w) < 1200:
        scales.append(1.5)

    for scale in scales:
        nw = int(w * scale)
        nh = int(h * scale)
        img_s = cv2.resize(image_bgr, (nw, nh)) if scale != 1.0 else image_bgr

        detector = cv2.FaceDetectorYN.create(
            _YUNET_PATH, "", (nw, nh),
            score_threshold=conf_thr,
            nms_threshold=0.3,
            top_k=100
        )
        _, faces = detector.detect(img_s)
        if faces is None:
            continue
        for face in faces:
            x = int(face[0] / scale)
            y = int(face[1] / scale)
            fw = int(face[2] / scale)
            fh = int(face[3] / scale)
            # 边界检查
            x, y = max(0, x), max(0, y)
            fw = min(fw, w - x)
            fh = min(fh, h - y)
            if fw > 8 and fh > 8:
                results.append((x, y, fw, fh))

    return results


# ─── Haar 兜底检测 ────────────────────────────────────────────
def _detect_haar(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    h, w = image_bgr.shape[:2]
    min_sz = max(20, int(min(h, w) * 0.04))
    boxes = []

    for cascade, nbr in [(_haar_default, 4), (_haar_alt2, 3)]:
        rects = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=nbr,
            minSize=(min_sz, min_sz), flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(rects) > 0:
            for r in rects:
                boxes.append(tuple(int(v) for v in r))

    # 侧脸 + 镜像侧脸
    for flip in [False, True]:
        g = cv2.flip(gray, 1) if flip else gray
        rects = _haar_profile.detectMultiScale(
            g, scaleFactor=1.1, minNeighbors=3,
            minSize=(min_sz, min_sz), flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(rects) > 0:
            for (x, y, fw, fh) in rects:
                if flip:
                    x = w - x - fw
                boxes.append((int(x), int(y), int(fw), int(fh)))
    return boxes


def _select_primary_face(boxes, image_shape):
    """单人模式只保留最可信主脸：大、居中、形状正常。"""
    if not boxes:
        return []

    h_img, w_img = image_shape[:2]
    target_cx = w_img / 2.0
    target_cy = h_img * 0.38
    diag = max((w_img ** 2 + h_img ** 2) ** 0.5, 1.0)

    def score(box):
        x, y, fw, fh = box
        area_ratio = (fw * fh) / float(max(w_img * h_img, 1))
        cx = x + fw / 2.0
        cy = y + fh / 2.0
        center_dist = ((cx - target_cx) ** 2 + (cy - target_cy) ** 2) ** 0.5 / diag
        center_score = max(0.0, 1.0 - center_dist * 1.8)
        aspect = fw / float(max(fh, 1))
        aspect_score = max(0.0, 1.0 - abs(aspect - 0.85))
        top_bonus = 0.18 if cy < h_img * 0.72 else 0.0
        return area_ratio * 6.0 + center_score * 2.2 + aspect_score * 0.7 + top_bonus

    return [max(boxes, key=score)]


def _resolve_detect_mode(mode, detect_mode=None):
    """兼容旧调用：未显式指定时，头像=单人，模糊=多人。"""
    if detect_mode in ("single", "multi"):
        return detect_mode
    return "single" if mode == "avatar" else "multi"


def detect_single_face(image_bgr, yunet_boxes):
    boxes = yunet_boxes if yunet_boxes else _nms(_detect_haar(image_bgr), iou_thr=0.3)
    return _select_primary_face(boxes, image_bgr.shape)


def detect_multi_faces(image_bgr, yunet_boxes):
    boxes = yunet_boxes
    if len(boxes) == 0:
        boxes = _detect_haar(image_bgr)
    elif len(boxes) < 5:
        boxes = _nms(boxes + _detect_haar(image_bgr), iou_thr=0.3)
    else:
        boxes = _nms(boxes, iou_thr=0.3)
    return boxes


# ─── 主检测入口 ───────────────────────────────────────────────
def detect_faces(image_bgr, mode="blur", detect_mode=None,
                 pad_ratio_x=0.12, pad_ratio_top=0.28, pad_ratio_bot=0.10):
    """
    多级人脸检测：
    - `detect_mode=single`：优先 YuNet，仅保留最可信主脸，避免把背景/身体误判成额外人脸
    - `detect_mode=multi`：YuNet 为主，小数量时再用 Haar 补充漏检
    """
    detect_mode = _resolve_detect_mode(mode, detect_mode)
    h_img, w_img = image_bgr.shape[:2]
    yunet_boxes = _nms(_detect_yunet(image_bgr, conf_thr=0.55), iou_thr=0.3)

    if detect_mode == "single":
        boxes = detect_single_face(image_bgr, yunet_boxes)
    else:
        boxes = detect_multi_faces(image_bgr, yunet_boxes)

    # 扩大边界框，保证整张脸都在框内
    padded = []
    for (x, y, fw, fh) in boxes:
        px = int(fw * pad_ratio_x)
        pyt = int(fh * pad_ratio_top)
        pyb = int(fh * pad_ratio_bot)
        x2 = max(0, x - px)
        y2 = max(0, y - pyt)
        w2 = min(fw + 2 * px, w_img - x2)
        h2 = min(fh + pyt + pyb, h_img - y2)
        padded.append((x2, y2, w2, h2))

    return padded


# ─── 卡通头像生成 ─────────────────────────────────────────────
def generate_cartoon_avatar(size, index=0):
    palettes = [
        ("#FFD700", "#E07B00"),
        ("#87CEEB", "#1565C0"),
        ("#98FB98", "#2E7D32"),
        ("#FFB6C1", "#C2185B"),
        ("#DDA0DD", "#6A1B9A"),
        ("#FFA07A", "#BF360C"),
        ("#B0E0E6", "#0277BD"),
        ("#F0E68C", "#F57F17"),
    ]
    face_color, accent = palettes[index % len(palettes)]
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    m, cx = max(4, size//20), size//2

    draw.ellipse([m, m, size-m, size-m],
                 fill=face_color, outline=accent, width=max(2, size//40))

    ey, er, eox = int(size*.38), max(4, size//10), int(size*.20)
    for ex in [cx-eox, cx+eox]:
        draw.ellipse([ex-er, ey-er, ex+er, ey+er],
                     fill="white", outline=accent, width=max(1, size//60))
        pr = max(2, size//18)
        draw.ellipse([ex-pr, ey-pr, ex+pr, ey+pr], fill="#111")
        hl = max(1, size//40)
        draw.ellipse([ex-pr+hl, ey-pr+hl, ex-pr+hl*3, ey-pr+hl*3], fill="white")

    brow_y = ey - er - max(2, size//20)
    brow_w, brow_h = er + max(2, size//25), max(2, size//30)
    for ex in [cx-eox, cx+eox]:
        draw.ellipse([ex-brow_w, brow_y-brow_h, ex+brow_w, brow_y+brow_h], fill=accent)

    ny, nr = int(size*.56), max(2, size//28)
    draw.ellipse([cx-nr, ny-nr//2, cx+nr, ny+nr], fill=accent)

    sy, sw, sh = int(size*.67), int(size*.26), int(size*.09)
    draw.arc([cx-sw, sy-sh, cx+sw, sy+sh], start=12, end=168,
             fill=accent, width=max(2, size//28))

    blush_y, blush_r = int(size*.57), max(4, size//11)
    r, g, b = tuple(int(accent.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    blush = (r, g, b, 70)
    for ex in [cx-eox, cx+eox]:
        draw.ellipse([ex-blush_r, blush_y-blush_r//2,
                      ex+blush_r, blush_y+blush_r//2], fill=blush)
    return img


# ─── 功能 1：卡通头像替换 ─────────────────────────────────────
def apply_cartoon_avatar(image_bgr, faces, avatar_path=None):
    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGBA))
    for i, (x, y, w, h) in enumerate(faces):
        if avatar_path and os.path.exists(avatar_path):
            avatar = Image.open(avatar_path).convert("RGBA").resize((w, h), Image.LANCZOS)
        else:
            avatar = generate_cartoon_avatar(max(w, h), index=i).resize((w, h), Image.LANCZOS)
        px = x + (w - avatar.width) // 2
        py = y + (h - avatar.height) // 2
        pil_img.paste(avatar, (px, py), avatar)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGR)


# ─── 功能 2：人脸模糊 ─────────────────────────────────────────
def apply_face_blur(image_bgr, faces, blur_strength=55):
    """纯净高斯模糊，只处理人脸区域，其余完全高清，消除强烈的马赛克（二维码）感"""
    result = image_bgr.copy()
    ks = blur_strength | 1   # 保证奇数
    # 根据 blur_strength 动态调整模糊强度 (sigma)
    sigma = blur_strength / 3.0
    for (x, y, w, h) in faces:
        roi = result[y:y+h, x:x+w]
        blurred = cv2.GaussianBlur(roi, (ks, ks), sigma)
        result[y:y+h, x:x+w] = blurred
    return result


# ─── 主处理入口 ───────────────────────────────────────────────
def process_image(image_bytes, mode="blur", detect_mode=None, blur_strength=55, avatar_path=None):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("无法读取图片")

    detect_mode = _resolve_detect_mode(mode, detect_mode)
    faces = detect_faces(image_bgr, mode=mode, detect_mode=detect_mode)
    face_count = len(faces)

    if face_count == 0:
        _, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return buf.tobytes(), 0

    output = apply_cartoon_avatar(image_bgr, faces, avatar_path) \
             if mode == "avatar" else \
             apply_face_blur(image_bgr, faces, blur_strength)

    _, buf = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes(), face_count
