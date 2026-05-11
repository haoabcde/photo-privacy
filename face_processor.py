"""
人脸处理核心模块
- 主检测器：OpenCV FaceDetectorYN (YuNet ONNX) —— 深度学习，对集体照/小脸效果好
- 兜底检测器：Haar Cascade —— 应对模型未命中的情况
- 功能1：单人照片 - 用卡通头像覆盖人脸
- 功能2：集体合影 - 对人脸区域高斯模糊，其余高清
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import io
import os
import sys

def correct_image_orientation(image_bytes):
    """读取图片字节流，使用 PIL 修正 EXIF 旋转，并转为 BGR 的 numpy 数组供 cv2 使用"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)
        img = img.convert('RGB')
        # Convert RGB to BGR for OpenCV
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        # Fallback to direct cv2 decode if PIL fails
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def _resize_for_preview(image_bgr, max_side=900):
    h, w = image_bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return image_bgr
    scale = max_side / float(m)
    nw, nh = int(w * scale), int(h * scale)
    if nw < 1 or nh < 1:
        return image_bgr
    # 快速插值算法，缩短执行时间
    return cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_NEAREST)

def _map_strength(strength):
    try:
        s = int(float(strength))
    except Exception:
        s = 40
    s = max(0, min(s, 100))
    blur_strength = int(15 + (s / 100.0) * 40)
    global_blur_strength = int(max(0, s - 40) * 1)
    return blur_strength, global_blur_strength

def _encode_jpeg(image_bgr, quality=92):
    q = int(quality)
    q = max(10, min(q, 100))
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        raise ValueError("JPEG 编码失败")
    return buf.tobytes()

def get_resource_path(relative_path):
    """获取资源绝对路径，兼容 PyInstaller 的 _MEIPASS"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), relative_path)

# ─── 模型路径 ────────────────────────────────────────────────
_YUNET_PATH = get_resource_path(os.path.join("models", "face_detection_yunet_2023mar.onnx"))

# 懒加载全局变量，提高启动速度
_yunet_detector_cache = {}
_haar_default = None
_haar_alt2 = None
_haar_profile = None
_DATA_DIR = cv2.data.haarcascades

def _get_yunet_detector(nw, nh, conf_thr):
    """复用 YuNet 检测器实例，避免每次重新初始化"""
    key = (nw, nh, conf_thr)
    if key not in _yunet_detector_cache:
        _yunet_detector_cache[key] = cv2.FaceDetectorYN.create(
            _YUNET_PATH, "", (nw, nh),
            score_threshold=conf_thr,
            nms_threshold=0.3,
            top_k=100
        )
    return _yunet_detector_cache[key]

def _init_haar():
    """懒加载 Haar 模型"""
    global _haar_default, _haar_alt2, _haar_profile
    if _haar_default is None:
        _haar_default = cv2.CascadeClassifier(os.path.join(_DATA_DIR, "haarcascade_frontalface_default.xml"))
        _haar_alt2    = cv2.CascadeClassifier(os.path.join(_DATA_DIR, "haarcascade_frontalface_alt2.xml"))
        _haar_profile = cv2.CascadeClassifier(os.path.join(_DATA_DIR, "haarcascade_profileface.xml"))


# ─── NMS ───────────────────────────────────────────────
def _nms(boxes, iou_thr=0.35):
    if not boxes:
        return []
    
    # 为了避免各类数据类型造成的异常，统一转为普通的元组列表
    try:
        clean_boxes = []
        for b in boxes:
            if hasattr(b, '__len__') and len(b) >= 4:
                clean_boxes.append((float(b[0]), float(b[1]), float(b[2]), float(b[3])))
        if not clean_boxes:
            return []
            
        boxes_np = np.array(clean_boxes, dtype=np.float32)
        if len(boxes_np.shape) != 2 or boxes_np.shape[1] < 4:
            raise ValueError("Invalid boxes shape")
    except Exception:
        # 兜底：返回原始框的前几个（避免崩溃）
        return list(boxes)[:10]
        
    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    w = boxes_np[:, 2]
    h = boxes_np[:, 3]
    x2 = x1 + w
    y2 = y1 + h
    
    areas = w * h
    order = areas.argsort()[::-1]  # 按面积从大到小排序
    
    keep = []
    
    # 修复：确保 boxes 可以按原索引访问，即使它是 generator 或其他类型
    boxes_list = list(boxes)
    
    while order.size > 0:
        i = order[0]
        keep.append(boxes_list[i])
        
        if order.size == 1:
            break
            
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w_inter = np.maximum(0.0, xx2 - xx1)
        h_inter = np.maximum(0.0, yy2 - yy1)
        inter = w_inter * h_inter
        
        # 计算 IoU
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        # 找到 IoU 小于等于阈值的框，保留它们
        inds = np.where(ovr <= iou_thr)[0]
        order = order[inds + 1]
        
    return keep


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
        # 使用 INTER_NEAREST 或 INTER_LINEAR 加速 resize
        img_s = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_NEAREST) if scale != 1.0 else image_bgr

        detector = _get_yunet_detector(nw, nh, conf_thr)
        _, faces = detector.detect(img_s)
        if faces is None:
            continue
        for face in faces:
            try:
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
            except Exception:
                continue

    return results


# ─── Haar 兜底检测 ────────────────────────────────────────────
def _detect_haar(image_bgr):
    _init_haar()
    try:
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
    except Exception:
        return []


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


def detect_single_face(image_bgr, yunet_boxes, strict=False):
    if strict:
        boxes = list(yunet_boxes)
    else:
        if len(yunet_boxes) > 0:
            boxes = list(yunet_boxes)
        else:
            boxes = _nms(list(_detect_haar(image_bgr)), iou_thr=0.3)
    return _select_primary_face(boxes, image_bgr.shape)


def detect_multi_faces(image_bgr, yunet_boxes, strict=False):
    if strict:
        return list(yunet_boxes)
        
    boxes = list(yunet_boxes)
    if len(boxes) == 0:
        boxes = _nms(list(_detect_haar(image_bgr)), iou_thr=0.3)
    elif len(boxes) < 5:
        haar_res = list(_detect_haar(image_bgr))
        combined = boxes + haar_res
        boxes = _nms(combined, iou_thr=0.3)
    else:
        boxes = _nms(boxes, iou_thr=0.3)
    return boxes


# ─── 主检测入口 ───────────────────────────────────────────────
def detect_faces(image_bgr, mode="blur", detect_mode="multi",
                 pad_ratio_x=0.12, pad_ratio_top=0.28, pad_ratio_bot=0.10, strict=False):
    """
    多级人脸检测：
    - strict=True：仅使用 YuNet 并提高置信度阈值，禁用 Haar 兜底，防止批处理风景图误识别
    """
    h_img, w_img = image_bgr.shape[:2]
    
    conf_thr = 0.65 if strict else 0.55
    yunet_boxes = _nms(_detect_yunet(image_bgr, conf_thr=conf_thr), iou_thr=0.3)

    if detect_mode == "single":
        boxes = detect_single_face(image_bgr, yunet_boxes, strict)
    else:
        boxes = detect_multi_faces(image_bgr, yunet_boxes, strict)

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
_cartoon_avatar_cache = {}

def generate_cartoon_avatar(size, index=0):
    cache_key = (size, index)
    if cache_key in _cartoon_avatar_cache:
        return _cartoon_avatar_cache[cache_key]

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
    _cartoon_avatar_cache[cache_key] = img
    return img


# ─── 功能 1：卡通头像替换 ─────────────────────────────────────
def apply_cartoon_avatar(image_bgr, faces, avatar_path=None, avatar_bytes=None):
    if not faces:
        return image_bgr
        
    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGBA))
    for i, (x, y, w, h) in enumerate(faces):
        avatar = None
        if avatar_bytes:
            try:
                avatar = Image.open(io.BytesIO(avatar_bytes)).convert("RGBA")
            except Exception:
                avatar = None
        if avatar is None and avatar_path and os.path.exists(avatar_path):
            try:
                avatar = Image.open(avatar_path).convert("RGBA")
            except Exception:
                avatar = None
        if avatar is None:
            avatar = generate_cartoon_avatar(max(w, h), index=i)
        # 使用快速插值
        avatar = avatar.resize((w, h), Image.NEAREST)
        px = x + (w - avatar.width) // 2
        py = y + (h - avatar.height) // 2
        pil_img.paste(avatar, (px, py), avatar)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGR)


# ─── 功能 2：人脸模糊 ─────────────────────────────────────────
def apply_face_blur(image_bgr, faces, blur_strength=55):
    """纯净高斯模糊，只处理人脸区域，边缘平滑过渡（羽化），自然融合"""
    if not faces:
        return image_bgr
    
    result = image_bgr.copy()
    
    for (x, y, w, h) in faces:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(result.shape[1], x+w), min(result.shape[0], y+h)
        if x2 <= x1 or y2 <= y1:
            continue
            
        roi = result[y1:y2, x1:x2]
        
        # 动态计算核大小（基于人脸实际大小），让模糊更加轻微自然
        # blur_strength (15~99) 映射到相对脸部宽度的 2% ~ 12% 左右
        # 这样即使 99 也不会完全糊成马赛克，而是一种自然的景深/失焦模糊感
        ratio = (blur_strength / 100.0) * 0.12 
        ks = int(max(w, h) * ratio)
        ks = max(3, ks | 1)  # 保证奇数且至少为 3
        
        sigma = ks / 3.0
        blurred = cv2.GaussianBlur(roi, (ks, ks), sigma)
        
        # 创建羽化遮罩（椭圆形渐变）以实现自然边缘融合
        mask = np.zeros((y2-y1, x2-x1), dtype=np.float32)
        center = ((x2-x1)//2, (y2-y1)//2)
        axes = (int((x2-x1)*0.45), int((y2-y1)*0.45))
        
        if axes[0] > 0 and axes[1] > 0:
            cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
            # 羽化核大小也跟脸部尺寸相关，做到平滑过渡
            feather_ks = int(max(w, h) * 0.25) | 1
            feather_ks = max(3, feather_ks)
            mask = cv2.GaussianBlur(mask, (feather_ks, feather_ks), 0)
        else:
            mask += 1.0
            
        mask = mask[..., np.newaxis]  # 扩展为 3 通道以便与图像相乘
        
        # 将模糊后的图像与原图按照羽化遮罩进行平滑融合
        result[y1:y2, x1:x2] = (blurred * mask + roi * (1 - mask)).astype(np.uint8)
        
    return result


def _normalize_detect_mode(detect_mode):
    return detect_mode if detect_mode in {"single", "multi"} else "multi"


# ─── 主处理入口 ───────────────────────────────────────────────
def process_image(image_bytes, mode="blur", detect_mode=None, blur_strength=55, avatar_path=None, global_blur_strength=0, 
                  is_smart_batch=False, rule_single="blur", rule_multi="blur_faces", avatar_bytes=None, jpeg_quality_override=None):
    image_bgr = correct_image_orientation(image_bytes)
    if image_bgr is None:
        raise ValueError("无法读取图片")

    # 全量检测人脸，批处理时启用严格模式防止风景图误识别
    initial_detect_mode = "multi" if is_smart_batch else _normalize_detect_mode(detect_mode)
    faces = detect_faces(image_bgr, mode=mode, detect_mode=initial_detect_mode, strict=is_smart_batch)
    face_count = len(faces)

    output = image_bgr # 默认不复制，除非需要修改

    if is_smart_batch:
        # 智能批处理路由逻辑
        if face_count == 0:
            # 无脸照片：直接原图返回，不重编码
            return image_bytes, 0
        elif face_count == 1:
            if rule_single == "keep":
                return image_bytes, 1
            # 若不是 keep，再用 single 策略精准定位那一张脸
            faces = detect_faces(image_bgr, mode="blur", detect_mode="single", strict=is_smart_batch)
            if rule_single == "avatar":
                output = apply_cartoon_avatar(image_bgr, faces, avatar_path, avatar_bytes)
            else:
                output = apply_face_blur(image_bgr, faces, blur_strength)
        else: # face_count >= 2
            if rule_multi == "global_blur_only":
                output = image_bgr # 不模糊人脸，仅走下方全局模糊
            else: # rule_multi == "blur_faces"
                output = apply_face_blur(image_bgr, faces, blur_strength)
    else:
        # 单张精修逻辑：直接全量检测所有面部，统一处理
        if face_count == 0:
            output = image_bgr
        else:
            if mode == "avatar":
                output = apply_cartoon_avatar(image_bgr, faces, avatar_path, avatar_bytes)
            else:
                output = apply_face_blur(image_bgr, faces, blur_strength)

    # === 全局融合与老照片感处理 ===
    jpeg_quality = 95
    if global_blur_strength > 0:
        sigma = global_blur_strength / 20.0
        k_size = int(sigma * 2) | 1
        k_size = min(k_size, 7)
        if k_size >= 3:
            output = cv2.GaussianBlur(output, (k_size, k_size), sigma)
        jpeg_quality = max(35, 95 - int(global_blur_strength * 0.6))

    if jpeg_quality_override is not None:
        try:
            jpeg_quality = int(float(jpeg_quality_override))
        except Exception:
            jpeg_quality = 92
        jpeg_quality = max(10, min(jpeg_quality, 100))

    return _encode_jpeg(output, quality=jpeg_quality), face_count


def process_preview(image_bytes, strength=40, mode="blur", detect_mode=None, avatar_path=None, avatar_bytes=None, is_smart_batch=False,
                    rule_single="blur", rule_multi="blur_faces", safety_level=None):
    image_bgr = correct_image_orientation(image_bytes)
    if image_bgr is None:
        raise ValueError("无法读取图片")

    image_bgr = _resize_for_preview(image_bgr, max_side=900)
    blur_strength, global_blur_strength = _map_strength(strength)

    initial_detect_mode = "multi" if is_smart_batch else _normalize_detect_mode(detect_mode)
    faces = detect_faces(image_bgr, mode=mode, detect_mode=initial_detect_mode, strict=is_smart_batch)
    face_count = len(faces)
    output = image_bgr

    if is_smart_batch:
        if face_count == 0:
            output = image_bgr
        elif face_count == 1:
            if rule_single == "keep":
                output = image_bgr
            else:
                faces = detect_faces(image_bgr, mode="blur", detect_mode="single", strict=is_smart_batch)
                if rule_single == "avatar":
                    output = apply_cartoon_avatar(image_bgr, faces, avatar_path, avatar_bytes)
                else:
                    output = apply_face_blur(image_bgr, faces, blur_strength)
        else:
            if rule_multi == "global_blur_only":
                output = image_bgr
            else:
                output = apply_face_blur(image_bgr, faces, blur_strength)
    else:
        if face_count == 0:
            output = image_bgr
        else:
            if mode == "avatar":
                output = apply_cartoon_avatar(image_bgr, faces, avatar_path, avatar_bytes)
            else:
                output = apply_face_blur(image_bgr, faces, blur_strength)

    if global_blur_strength > 0:
        sigma = global_blur_strength / 20.0
        k_size = int(sigma * 2) | 1
        k_size = min(k_size, 7)
        if k_size >= 3:
            output = cv2.GaussianBlur(output, (k_size, k_size), sigma)

    return _encode_jpeg(output, quality=90), face_count
