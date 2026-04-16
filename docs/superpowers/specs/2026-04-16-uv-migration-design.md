# 使用 uv 统一管理 Python 依赖（方案 A）设计稿

## Summary

将仓库的 Python 依赖管理从 `requirements.txt + pip` 迁移到 `pyproject.toml + uv.lock`，并保留 `requirements.txt` 作为兼容产物（由 uv 从 lock 导出）。同时新增 GitHub Actions，确保每次提交都能复现安装、跑通测试，并校验导出依赖文件与 lock 一致。

## Goals

- 统一依赖来源：以 `pyproject.toml + uv.lock` 作为唯一真相（single source of truth）。
- 可复现：任何人拉取仓库后，使用 uv 可以得到一致的依赖版本。
- 兼容部署：保留 `requirements.txt`，便于不使用 uv 的部署平台（或习惯 pip 的用户）。
- 提升开源专业度：提供清晰的本地运行/生产部署指引，并用 CI 防止依赖漂移。

## Non-Goals

- 不改变现有业务功能与接口（上传、预览、批处理、下载等）。
- 不引入新的模型/算法或额外的运行时依赖（除 uv 管理本身）。
- 不强制要求使用 Docker（仅优化 Dockerfile 依赖安装方式）。

## Current State Analysis

- 依赖文件：仅有 [requirements.txt](file:///Users/hao/Developer/photo/requirements.txt)。
- 文档与部署命令均基于 pip（见 [README.md](file:///Users/hao/Developer/photo/README.md)）。
- Dockerfile 通过 `pip install -r requirements.txt` 安装依赖（见 [Dockerfile](file:///Users/hao/Developer/photo/Dockerfile)）。
- 测试框架：已有 `unittest` 用例文件 [tests_test_face_processor.py](file:///Users/hao/Developer/photo/tests_test_face_processor.py)。

## Proposed Changes

### 1) 引入 uv 管理结构

- 新增 `pyproject.toml`
  - `project.name`：建议 `photo-privacy`
  - `requires-python`：固定 `>=3.11`
  - `dependencies`：把 requirements.txt 中的依赖迁移进去（flask, gunicorn, numpy, opencv-python-headless, pillow）
- 新增 `uv.lock`
  - 由 `uv lock` 生成并提交
- 新增 `.python-version`
  - 固定为 `3.11`，便于本地 pyenv/asdf 等工具一致化

### 2) requirements.txt 改为 uv 导出产物

- 保留 `requirements.txt`（仍然提交在仓库中）
- 新增脚本/约定：
  - 使用 `uv export --format requirements-txt --no-hashes -o requirements.txt`
  - CI 中每次生成后对比工作区 diff，确保与提交内容一致

### 3) README 改造：以 uv 为主，pip 为辅

- 本地开发路径：
  - `uv sync`
  - `uv run python app.py`
- 生产运行路径：
  - `uv run gunicorn ...`
- 兼容方式保留：
  - `pip install -r requirements.txt`

### 4) Dockerfile 改造：基于 uv.lock 安装依赖

- 在镜像中安装 uv
- `COPY pyproject.toml uv.lock` 后执行 `uv sync --frozen --no-dev`
- 保留现有 `gunicorn` 启动命令不变

### 5) GitHub Actions：测试 + 依赖一致性校验

- Workflow：
  - checkout
  - 安装 uv
  - `uv sync --frozen`
  - 运行 `python -m unittest -q tests_test_face_processor.py`
  - 导出 `requirements.txt` 并 `git diff --exit-code requirements.txt`

## Decisions

- 采用方案 A：以 `pyproject.toml + uv.lock` 为准，保留 `requirements.txt` 作为导出兼容产物。
- Python 版本固定为 3.11（与当前基础镜像一致）。
- CI 必须验证 requirements.txt 与 uv.lock 同步，避免出现“锁文件和 requirements 不一致”的部署风险。

## Verification

- 本地验证
  - `uv sync --frozen`
  - `uv run python -m py_compile app.py face_processor.py wsgi.py`
  - `uv run python -m unittest -q tests_test_face_processor.py`
- Docker 验证
  - `docker build .`
  - `docker run -p 5001:5001 ...` 后访问首页
- CI 验证
  - GitHub Actions 在 PR / push 上通过

