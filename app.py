"""
Flask Web 应用 - 人脸隐私保护工具
"""
import os
import sys
import uuid
import json
import cv2
import numpy as np
import zipfile
import io
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
from face_processor import process_image, detect_faces, _detect_yunet, _detect_haar

def get_resource_path(relative_path):
    """获取资源绝对路径，兼容 PyInstaller 的 _MEIPASS"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), relative_path)

app = Flask(__name__,
            static_folder=get_resource_path("static"),
            template_folder=get_resource_path("templates"))
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB

USER_DATA_DIR = os.path.expanduser("~/.photo_privacy")
UPLOAD_FOLDER = os.path.join(USER_DATA_DIR, "uploads")
RESULT_FOLDER = os.path.join(USER_DATA_DIR, "results")
AVATAR_FOLDER = os.path.join(USER_DATA_DIR, "avatars")

for folder in [UPLOAD_FOLDER, RESULT_FOLDER, AVATAR_FOLDER]:
    os.makedirs(folder, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/user_data/<folder>/<filename>")
def user_data(folder, filename):
    if folder not in ["uploads", "results", "avatars"]:
        return "Forbidden", 403
    folder_path = os.path.join(USER_DATA_DIR, folder)
    return send_from_directory(folder_path, filename)

@app.route("/process", methods=["POST"])
def process():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "未上传图片"}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "不支持的文件格式"}), 400

    mode = request.form.get("mode", "blur")          # 处理效果: "avatar" 或 "blur"
    detect_mode = request.form.get("detect_mode")     # 检测策略: "single" 或 "multi"
    blur_strength = int(request.form.get("blur_strength", 55))
    global_blur_strength = int(request.form.get("global_blur_strength", 0))

    image_bytes = file.read()

    # 检查是否有自定义头像
    avatar_path = None
    if "avatar" in request.files and request.files["avatar"].filename:
        av_file = request.files["avatar"]
        av_ext = av_file.filename.rsplit(".", 1)[1].lower()
        av_filename = f"{uuid.uuid4().hex}.{av_ext}"
        avatar_path = os.path.join(AVATAR_FOLDER, av_filename)
        av_file.save(avatar_path)

    try:
        result_bytes, face_count = process_image(
            image_bytes,
            mode=mode,
            detect_mode=detect_mode,
            blur_strength=blur_strength,
            avatar_path=avatar_path,
            global_blur_strength=global_blur_strength
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    # 保存结果
    result_filename = f"{uuid.uuid4().hex}.jpg"
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    with open(result_path, "wb") as f:
        f.write(result_bytes)

    return jsonify({
        "success": True,
        "face_count": face_count,
        "result_url": f"/user_data/results/{result_filename}",
        "mode": mode,
        "detect_mode": detect_mode,
    })


@app.route("/batch_process", methods=["POST"])
def batch_process():
    if "images" not in request.files:
        return jsonify({"success": False, "error": "未上传图片"}), 400

    files = request.files.getlist("images")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"success": False, "error": "不支持的文件格式"}), 400

    mode = request.form.get("mode", "blur")
    detect_mode = request.form.get("detect_mode")
    blur_strength = int(request.form.get("blur_strength", 55))
    global_blur_strength = int(request.form.get("global_blur_strength", 0))

    # 新增批处理智能规则参数
    is_smart_batch = request.form.get("is_smart_batch") == "true"
    rule_single = request.form.get("rule_single", "blur")
    rule_multi = request.form.get("rule_multi", "blur_faces")

    avatar_path = None
    if "avatar" in request.files and request.files["avatar"].filename:
        av_file = request.files["avatar"]
        av_ext = av_file.filename.rsplit(".", 1)[1].lower()
        av_filename = f"{uuid.uuid4().hex}.{av_ext}"
        avatar_path = os.path.join(AVATAR_FOLDER, av_filename)
        av_file.save(avatar_path)

    zip_filename = f"batch_{uuid.uuid4().hex}.zip"
    zip_path = os.path.join(RESULT_FOLDER, zip_filename)
    
    total_faces = 0
    processed_count = 0

    try:
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in files:
                if file.filename and allowed_file(file.filename):
                    image_bytes = file.read()
                    result_bytes, face_count = process_image(
                        image_bytes,
                        mode=mode,
                        detect_mode=detect_mode,
                        blur_strength=blur_strength,
                        avatar_path=avatar_path,
                        global_blur_strength=global_blur_strength,
                        is_smart_batch=is_smart_batch,
                        rule_single=rule_single,
                        rule_multi=rule_multi
                    )
                    total_faces += face_count
                    processed_count += 1
                    # 避免同名文件覆盖，加上 uuid
                    safe_name = f"{uuid.uuid4().hex[:6]}_{file.filename}"
                    zipf.writestr(safe_name, result_bytes)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    return jsonify({
        "success": True,
        "processed_count": processed_count,
        "total_faces": total_faces,
        "result_url": f"/user_data/results/{zip_filename}",
        "mode": mode,
        "detect_mode": detect_mode,
    })

@app.route("/debug_boxes", methods=["POST"])
def debug_boxes():
    """返回叠加了人脸检测框的调试图（绿=YuNet, 橙=Haar, 红=最终）"""
    if "image" not in request.files:
        return jsonify({"success": False, "error": "未上传图片"}), 400
    file = request.files["image"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "不支持的格式"}), 400

    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"success": False, "error": "无法读取图片"}), 400

    mode = request.form.get("mode", "blur")
    detect_mode = request.form.get("detect_mode")
    yunet = _detect_yunet(img, conf_thr=0.5)
    haar = _detect_haar(img)
    final = detect_faces(img, mode=mode, detect_mode=detect_mode)

    vis = img.copy()
    for (x, y, w, h) in yunet:
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 200, 0), 2)
    for (x, y, w, h) in haar:
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 140, 255), 1)
    for i, (x, y, w, h) in enumerate(final):
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 220), 2)
        cv2.putText(vis, str(i+1), (x+4, y+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2)

    _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    dbg_filename = f"debug_{uuid.uuid4().hex}.jpg"
    dbg_path = os.path.join(RESULT_FOLDER, dbg_filename)
    with open(dbg_path, "wb") as f:
        f.write(buf.tobytes())

    return jsonify({
        "success": True,
        "yunet_count": len(yunet),
        "haar_count":  len(haar),
        "final_count": len(final),
        "debug_url":   f"/user_data/results/{dbg_filename}",
    })


@app.route("/download/<filename>")
def download(filename):
    path = os.path.join(RESULT_FOLDER, filename)
    if not os.path.exists(path):
        return "文件不存在", 404
    return send_file(path, as_attachment=True, download_name=f"processed_{filename}")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
