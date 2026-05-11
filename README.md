# 🛡️ Photo Privacy (人脸隐私保护工具)

> **人脸隐私保护终极工具：让你的照片在分享时既安全又得体。**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/built%20with-uv-purple.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![OpenCV](https://img.shields.io/badge/OpenCV-YuNet-green.svg)](https://opencv.org/)

一个本地/自部署的人脸隐私保护网页工具：支持对图片中的人脸进行卡通头像替换或局部模糊处理，并提供实时预览与批处理 ZIP 下载。

## ✨ 核心特性

- **🧠 深度检测**：集成 **OpenCV YuNet** 深度学习模型，支持多尺度检测，哪怕是远景中的细小人脸也能精准锁定。
- **🎨 自然融合**：不同于死板的马赛克，我们采用**高斯模糊边缘羽化算法**，让处理区域与原图丝滑融合，保持画面美感。
- **🎭 趣味替换**：支持一键将人脸替换为风格多样的卡通头像，隐私保护也可以很个性。
- **⚡ 极速批处理**：完美支持 ZIP 压缩包上传，多线程并发处理，瞬间完成海量照片脱敏。
- **🔒 100% 本地运行**：所有计算均在本地或私有服务器完成，数据绝不上传云端，确保绝对私密。
- **🚀 智能路由**：根据人脸数量自动匹配最优处理策略（单人 vs 多人）。

## 🛠️ 技术亮点

- **YuNet AI 检测**：相较于传统 Haar 级联，提供更精准的定位和更高的召回率。
- **EXIF 自动校正**：自动识别并修正图片旋转方向，确保预览与输出一致。
- **缩略图优化**：智能缩放预览图，大幅提升网页加载与响应速度。
- **Modern 工具链**：基于 `uv` 构建，提供极速的依赖同步与开发体验。

## 🚀 快速开始

### 使用 uv (推荐)
仅需一步即可启动：
```bash
uv run python app.py
```
访问：[http://127.0.0.1:5001](http://127.0.0.1:5001)

### 使用传统方式 (pip)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

## ⚙️ 配置参数

| 环境变量 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `PHOTO_PRIVACY_DATA_DIR` | 上传与结果存储目录 | `./user_data/` |
| `HOST` | 服务监听地址 | `0.0.0.0` |
| `PORT` | 服务端口 | `5001` |
| `FLASK_DEBUG` | 是否开启调试模式 | `0` |

## 🐳 Docker 部署

```bash
docker build -t photo-privacy .
docker run --rm -p 5001:5001 -e PORT=5001 -e PHOTO_PRIVACY_DATA_DIR=/data photo-privacy
```

## 🏗️ 生产环境

建议使用 Gunicorn 配合多线程模式以获得更佳性能：
```bash
uv run gunicorn -b 0.0.0.0:5001 wsgi:app --workers 1 --threads 4 --timeout 120
```

## 🗺️ 开发路线图

- [ ] 视频人脸隐私脱敏支持
- [ ] 更多卡通头像预设库
- [ ] 基于 WebUI 的交互式人脸剔除
- [ ] GPU 加速推理支持 (CUDA)
