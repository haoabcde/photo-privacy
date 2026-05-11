"""
Flask Web 应用 - 人脸隐私保护工具
"""
import os
import sys
import uuid
import json
import zipfile
import io
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory, Response
from face_processor import process_image, process_preview

def get_resource_path(relative_path):
    """获取资源绝对路径，兼容 PyInstaller 的 _MEIPASS"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), relative_path)

app = Flask(__name__,
            static_folder=get_resource_path("static"),
            template_folder=get_resource_path("templates"))
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024 * 1024  # 10GB

_user_data_dir_override = os.environ.get("PHOTO_PRIVACY_DATA_DIR")
if _user_data_dir_override:
    USER_DATA_DIR = os.path.abspath(os.path.expanduser(_user_data_dir_override))
elif getattr(sys, 'frozen', False):
    USER_DATA_DIR = os.path.expanduser("~/Downloads/PhotoPrivacyData")
else:
    USER_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_data")
UPLOAD_FOLDER = os.path.join(USER_DATA_DIR, "uploads")
RESULT_FOLDER = os.path.join(USER_DATA_DIR, "results")
AVATAR_FOLDER = os.path.join(USER_DATA_DIR, "avatars")

for folder in [UPLOAD_FOLDER, RESULT_FOLDER, AVATAR_FOLDER]:
    os.makedirs(folder, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp", "zip"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def _sanitize_download_name(name, fallback, ext_hint=None):
    if not name:
        return fallback
    name = str(name)
    name = name.replace("\x00", "").replace("\r", "").replace("\n", "")
    name = name.replace("/", "_").replace("\\", "_").replace(":", "_")
    name = name.strip()
    if not name:
        return fallback
    if ext_hint and "." not in name:
        name = f"{name}.{ext_hint}"
    if len(name) > 180:
        name = name[-180:]
    return name

_BATCH_UPLOAD_INDEX_PATH = os.path.join(UPLOAD_FOLDER, "batch_upload_index.json")
_batch_upload_index_cache = None

def _load_batch_upload_index():
    global _batch_upload_index_cache
    if _batch_upload_index_cache is not None:
        return _batch_upload_index_cache
    if not os.path.exists(_BATCH_UPLOAD_INDEX_PATH):
        _batch_upload_index_cache = {}
        return _batch_upload_index_cache
    try:
        with open(_BATCH_UPLOAD_INDEX_PATH, "r", encoding="utf-8") as f:
            _batch_upload_index_cache = json.load(f)
        if not isinstance(_batch_upload_index_cache, dict):
            _batch_upload_index_cache = {}
    except Exception:
        _batch_upload_index_cache = {}
    return _batch_upload_index_cache

def _save_batch_upload_index(index):
    global _batch_upload_index_cache
    os.makedirs(os.path.dirname(_BATCH_UPLOAD_INDEX_PATH), exist_ok=True)
    tmp_path = _BATCH_UPLOAD_INDEX_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    os.replace(tmp_path, _BATCH_UPLOAD_INDEX_PATH)
    _batch_upload_index_cache = index

def _is_valid_zip_member_name(name):
    if not name:
        return False
    name = name.replace("\\", "/")
    parts = name.split("/")
    if any(p == "" for p in parts):
        return False
    if any(p.startswith(".") or p == "__MACOSX" for p in parts):
        return False
    return True

def _decode_zip_filename(info):
    if info.flag_bits & 0x800:
        return info.filename
    else:
        try:
            # 还原为字节流再尝试用 GBK（中文 Windows 默认）或 UTF-8 解码
            b = info.filename.encode('cp437')
            try:
                return b.decode('gbk')
            except UnicodeDecodeError:
                return b.decode('utf-8')
        except Exception:
            return info.filename

def _iter_zip_image_filenames(z):
    for info in z.infolist():
        if info.is_dir():
            continue
        filename = _decode_zip_filename(info).replace("\\", "/")
        if not _is_valid_zip_member_name(filename):
            continue
        if "." not in filename:
            continue
        ext = filename.rsplit(".", 1)[1].lower()
        if ext in {"jpg", "jpeg", "png", "webp", "bmp"}:
            yield filename, info


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
        
    builtin_avatar = request.form.get("builtin_avatar")
    if not avatar_path and builtin_avatar:
        builtin_path = os.path.join(get_resource_path("static"), "avatars", "built-in", builtin_avatar)
        if os.path.exists(builtin_path):
            avatar_path = builtin_path

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
        import traceback
        traceback.print_exc()
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


@app.route("/preview", methods=["POST"])
def preview():
    image_bytes = None
    if "image" in request.files and request.files["image"].filename:
        image_bytes = request.files["image"].read()
    else:
        sample_id = request.form.get("sample_id", "default")
        sample_path = os.path.join(get_resource_path("static"), "samples", f"{sample_id}.jpg")
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                image_bytes = f.read()

    if not image_bytes:
        return jsonify({"success": False, "error": "未提供图片"}), 400

    strength = request.form.get("strength", "40")
    mode = request.form.get("mode", "blur")
    detect_mode = request.form.get("detect_mode")
    is_smart_batch = request.form.get("is_smart_batch") == "true"
    rule_single = request.form.get("rule_single", "blur")
    rule_multi = request.form.get("rule_multi", "blur_faces")
    safety_level = request.form.get("safety_level")

    avatar_path = None
    avatar_bytes = None
    if "avatar" in request.files and request.files["avatar"].filename:
        avatar_bytes = request.files["avatar"].read()

    builtin_avatar = request.form.get("builtin_avatar")
    if not avatar_bytes and builtin_avatar:
        builtin_path = os.path.join(get_resource_path("static"), "avatars", "built-in", builtin_avatar)
        if os.path.exists(builtin_path):
            avatar_path = builtin_path

    try:
        result_bytes, _ = process_preview(
            image_bytes,
            strength=strength,
            mode=mode,
            detect_mode=detect_mode,
            avatar_path=avatar_path,
            avatar_bytes=avatar_bytes,
            is_smart_batch=is_smart_batch,
            rule_single=rule_single,
            rule_multi=rule_multi,
            safety_level=safety_level,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

    return Response(result_bytes, mimetype="image/jpeg")

@app.route("/batch_prepare", methods=["POST"])
def batch_prepare():
    zip_file = None
    if "zip" in request.files and request.files["zip"].filename:
        zip_file = request.files["zip"]
    elif "file" in request.files and request.files["file"].filename:
        zip_file = request.files["file"]
    elif "images" in request.files:
        for f in request.files.getlist("images"):
            if f and f.filename and f.filename.lower().endswith(".zip"):
                zip_file = f
                break

    if not zip_file or not zip_file.filename:
        return jsonify({"success": False, "error": "未上传 ZIP"}), 400
    if not allowed_file(zip_file.filename) or not zip_file.filename.lower().endswith(".zip"):
        return jsonify({"success": False, "error": "不支持的文件格式"}), 400

    upload_id = uuid.uuid4().hex
    zip_filename = f"upload_{upload_id}.zip"
    zip_path = os.path.join(UPLOAD_FOLDER, zip_filename)
    zip_file.save(zip_path)

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            samples = []
            for name, info in _iter_zip_image_filenames(z):
                # 仍然向前端发送内部真实路径，以便后续 getinfo() 能获取到
                samples.append(info.filename)
                if len(samples) >= 5:
                    break
    except Exception as e:
        try:
            if os.path.exists(zip_path):
                os.remove(zip_path)
        except Exception:
            pass
        return jsonify({"success": False, "error": f"ZIP 读取失败: {e}"}), 400

    index = _load_batch_upload_index()
    index[upload_id] = zip_filename
    _save_batch_upload_index(index)

    return jsonify({"success": True, "upload_id": upload_id, "samples": samples})


@app.route("/batch_preview_image", methods=["POST"])
def batch_preview_image():
    upload_id = request.form.get("upload_id")
    filename = request.form.get("filename")
    if not upload_id or not filename:
        return jsonify({"success": False, "error": "缺少 upload_id 或 filename"}), 400

    index = _load_batch_upload_index()
    zip_filename = index.get(upload_id)
    if not zip_filename:
        return jsonify({"success": False, "error": "upload_id 不存在或已过期"}), 404

    zip_path = os.path.join(UPLOAD_FOLDER, zip_filename)
    if not os.path.exists(zip_path):
        return jsonify({"success": False, "error": "ZIP 文件不存在"}), 404

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            try:
                info = z.getinfo(filename)
            except KeyError:
                return jsonify({"success": False, "error": "文件不存在"}), 404
            if info.is_dir():
                return jsonify({"success": False, "error": "目标是目录"}), 400
            image_bytes = z.read(info.filename)
    except Exception as e:
        return jsonify({"success": False, "error": f"ZIP 读取失败: {e}"}), 400

    strength = request.form.get("strength", "40")
    mode = request.form.get("mode", "blur")
    is_smart_batch = request.form.get("is_smart_batch") == "true"
    rule_single = request.form.get("rule_single", "blur")
    rule_multi = request.form.get("rule_multi", "blur_faces")
    safety_level = request.form.get("safety_level")

    avatar_path = None
    avatar_bytes = None
    if "avatar" in request.files and request.files["avatar"].filename:
        avatar_bytes = request.files["avatar"].read()

    builtin_avatar = request.form.get("builtin_avatar")
    if not avatar_bytes and builtin_avatar:
        builtin_path = os.path.join(get_resource_path("static"), "avatars", "built-in", builtin_avatar)
        if os.path.exists(builtin_path):
            avatar_path = builtin_path

    try:
        result_bytes, _ = process_preview(
            image_bytes,
            strength=strength,
            mode=mode,
            avatar_path=avatar_path,
            avatar_bytes=avatar_bytes,
            is_smart_batch=is_smart_batch,
            rule_single=rule_single,
            rule_multi=rule_multi,
            safety_level=safety_level,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

    return Response(result_bytes, mimetype="image/jpeg")


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
        
    builtin_avatar = request.form.get("builtin_avatar")
    if not avatar_path and builtin_avatar:
        builtin_path = os.path.join(get_resource_path("static"), "avatars", "built-in", builtin_avatar)
        if os.path.exists(builtin_path):
            avatar_path = builtin_path

    zip_filename = f"batch_{uuid.uuid4().hex}.zip"
    zip_path = os.path.join(RESULT_FOLDER, zip_filename)
    
    total_faces = 0
    processed_count = 0

    # 提取所有待处理的图片 (filename, image_bytes)
    images_to_process = []
    download_name = None
    
    try:
        for file in files:
            if not file.filename or not allowed_file(file.filename):
                continue
                
            ext = file.filename.rsplit(".", 1)[1].lower()
            if ext == "zip":
                if download_name is None:
                    download_name = file.filename
                # 处理上传的 zip 文件
                file_bytes = file.read()
                with zipfile.ZipFile(io.BytesIO(file_bytes), 'r') as z:
                    for zip_info in z.infolist():
                        # 忽略目录和隐藏文件（如 macOS 的 __MACOSX 目录）
                        if zip_info.is_dir():
                            continue
                        
                        # 检查路径中是否包含 __MACOSX 或任何隐藏文件/文件夹
                        decoded_name = _decode_zip_filename(zip_info)
                        path_parts = decoded_name.replace('\\', '/').split('/')
                        if any(p.startswith('.') or p == '__MACOSX' for p in path_parts):
                            continue
                        
                        # 检查扩展名是否为允许的图片
                        if "." in decoded_name:
                            inner_ext = decoded_name.rsplit(".", 1)[1].lower()
                            if inner_ext in {"jpg", "jpeg", "png", "webp", "bmp"}:
                                img_bytes = z.read(zip_info.filename)
                                # 完整保留在 ZIP 中的相对路径，使用解码后的名称以防乱码
                                images_to_process.append((decoded_name, img_bytes))
            else:
                # 处理普通图片文件
                images_to_process.append((file.filename, file.read()))
                
        if not images_to_process:
            return jsonify({"success": False, "error": "未在上传的文件或 ZIP 中找到有效的图片"}), 400

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filename, image_bytes in images_to_process:
                try:
                    result_bytes, face_count = process_image(
                        image_bytes,
                        mode=mode,
                        detect_mode=detect_mode,
                        blur_strength=blur_strength,
                        avatar_path=avatar_path,
                        global_blur_strength=global_blur_strength,
                        is_smart_batch=is_smart_batch,
                        rule_single=rule_single,
                        rule_multi=rule_multi,
                        avatar_bytes=None,
                        jpeg_quality_override=None
                    )
                    total_faces += face_count
                    processed_count += 1
                    # 拆分目录和文件名，只在文件名加上 uuid，以保留原有的目录结构
                    dir_name, base_name = os.path.split(filename)
                    safe_base_name = f"{uuid.uuid4().hex[:6]}_{base_name}"
                    safe_path = os.path.join(dir_name, safe_base_name).replace("\\", "/") if dir_name else safe_base_name
                    
                    import time
                    zinfo = zipfile.ZipInfo(safe_path, date_time=time.localtime()[:6])
                    zinfo.compress_type = zipfile.ZIP_DEFLATED
                    # 强制使用 UTF-8 编码，兼容所有系统解压
                    zinfo.flag_bits |= 0x800
                    
                    zipf.writestr(zinfo, result_bytes)
                except Exception as e:
                    print(f"Skipping {filename} due to error: {e}")
                    continue
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

    return jsonify({
        "success": True,
        "processed_count": processed_count,
        "total_faces": total_faces,
        "result_url": f"/user_data/results/{zip_filename}",
        "download_name": download_name or zip_filename,
        "mode": mode,
        "detect_mode": detect_mode,
    })

@app.route("/download/<filename>")
def download(filename):
    path = os.path.join(RESULT_FOLDER, filename)
    if not os.path.exists(path):
        return "文件不存在", 404
    desired_name = request.args.get("name")
    ext_hint = filename.rsplit(".", 1)[1].lower() if "." in filename else None
    download_name = _sanitize_download_name(desired_name, f"processed_{filename}", ext_hint=ext_hint)
    return send_file(path, as_attachment=True, download_name=download_name)


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG") == "1"
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5001"))
    app.run(debug=debug, host=host, port=port)
