# 人脸隐私保护工具（Photo Privacy）

一个本地/自部署的人脸隐私保护网页工具：支持对图片中的人脸进行卡通头像替换或局部模糊处理，并提供实时预览与批处理 ZIP 下载。

## 功能

- 单张精修：上传 1 张图片并实时预览处理效果
- 智能批处理：上传多张图片或 ZIP，批量处理并下载结果 ZIP
- 两种模式
  - 简单模式：一个“总强度”滑杆，自动联动人脸模糊与整体融合
  - 专业模式：可单独调节“局部人脸模糊强度”和“整体融合/包浆感”
- 完全自部署：你可以把它部署到自己的服务器上给别人直接使用

## 本地运行（开发）

```bash
uv sync
uv run python app.py
```

打开：

- http://127.0.0.1:5001/

如果你不使用 uv，也可以用 pip（兼容）：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

## 服务器运行（生产）

不要在公网用调试模式（`FLASK_DEBUG=1`），建议用 Gunicorn：

```bash
uv sync --frozen

export HOST=0.0.0.0
export PORT=5001
export PHOTO_PRIVACY_DATA_DIR=/tmp/photo-privacy-data

uv run gunicorn -b 0.0.0.0:${PORT} wsgi:app --workers 1 --threads 4 --timeout 120
```

## Docker 部署

```bash
docker build -t photo-privacy .
docker run --rm -p 5001:5001 -e PORT=5001 -e PHOTO_PRIVACY_DATA_DIR=/data photo-privacy
```

## 数据目录

服务会在 `PHOTO_PRIVACY_DATA_DIR` 下写入上传文件与处理结果：

- `uploads/`
- `results/`
- `avatars/`

如果不设置该变量，默认会写入项目目录下的 `./user_data/`（适合本地开发，不建议部署到只读文件系统）。
