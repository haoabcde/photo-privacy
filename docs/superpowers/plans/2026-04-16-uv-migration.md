# uv 统一依赖管理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将项目 Python 依赖管理迁移到 `pyproject.toml + uv.lock`，保留 `requirements.txt` 作为从 lock 导出的兼容产物，并完善 Docker 与 GitHub Actions。

**Architecture:** 依赖以 `pyproject.toml` 声明，以 `uv.lock` 锁定版本；本地/CI/Docker 均使用 `uv sync --frozen` 复现安装；`requirements.txt` 通过 `uv export` 导出并在 CI 中校验不漂移。

**Tech Stack:** Python 3.11, uv, Flask, OpenCV, GitHub Actions, Docker, Gunicorn

---

## File Structure

**Create**
- `/Users/hao/Developer/photo/pyproject.toml`
- `/Users/hao/Developer/photo/uv.lock` (由 uv 生成)
- `/Users/hao/Developer/photo/.python-version`
- `/Users/hao/Developer/photo/.github/workflows/ci.yml`

**Modify**
- `/Users/hao/Developer/photo/requirements.txt`
- `/Users/hao/Developer/photo/README.md`
- `/Users/hao/Developer/photo/Dockerfile`

---

### Task 1: 初始化 uv 工程文件（pyproject / python-version）

**Files:**
- Create: `/Users/hao/Developer/photo/pyproject.toml`
- Create: `/Users/hao/Developer/photo/.python-version`

- [ ] **Step 1: 写入 `.python-version`**

内容：

```text
3.11
```

- [ ] **Step 2: 写入 `pyproject.toml`**

内容：

```toml
[project]
name = "photo-privacy"
version = "0.1.0"
description = "Self-hosted photo face privacy tool (blur or avatar) with realtime preview and batch ZIP processing."
readme = "README.md"
requires-python = ">=3.11"
license = { file = "LICENSE" }
authors = [{ name = "haoabcde" }]
dependencies = [
  "flask>=3.1.0",
  "gunicorn>=22.0.0",
  "numpy>=1.24.0",
  "opencv-python-headless>=4.8.0",
  "pillow>=10.0.0",
]

[tool.uv]
dev-dependencies = []
```

- [ ] **Step 3: 生成 lock**

Run:

```bash
uv lock
```

Expected: 生成 `uv.lock` 文件。

- [ ] **Step 4: 从 lock 导出 requirements.txt**

Run:

```bash
uv export --format requirements-txt --no-hashes -o requirements.txt
```

Expected: `requirements.txt` 被改写为 pinned 版本（便于 pip 兼容安装）。

- [ ] **Step 5: 快速验证可运行**

Run:

```bash
uv run python -m py_compile app.py face_processor.py wsgi.py
uv run python -m unittest -q tests_test_face_processor.py
```

Expected: 编译通过；单测通过。

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock requirements.txt .python-version
git commit -m "chore: migrate to uv dependency management"
```

---

### Task 2: 更新 README（以 uv 为主、pip 为辅）

**Files:**
- Modify: `/Users/hao/Developer/photo/README.md`

- [ ] **Step 1: 更新本地运行说明**

将“本地运行（开发）”改为 uv 版（保留 pip 兼容方式），示例命令：

```bash
uv sync
uv run python app.py
```

兼容方式：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

- [ ] **Step 2: 更新生产运行说明**

改为：

```bash
uv sync --frozen
export HOST=0.0.0.0
export PORT=5001
export PHOTO_PRIVACY_DATA_DIR=/tmp/photo-privacy-data
uv run gunicorn -b 0.0.0.0:${PORT} wsgi:app --workers 1 --threads 4 --timeout 120
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document uv workflow"
```

---

### Task 3: 更新 Dockerfile（使用 uv.lock 安装依赖）

**Files:**
- Modify: `/Users/hao/Developer/photo/Dockerfile`

- [ ] **Step 1: 修改 Dockerfile**

目标结构：
1) 安装 uv
2) 先拷贝 `pyproject.toml` 和 `uv.lock` 利用缓存
3) `uv sync --frozen --no-dev`
4) 再拷贝业务代码

期望 Dockerfile 关键片段（保持最小变动）：

```dockerfile
COPY pyproject.toml uv.lock /app/
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev

COPY . /app
CMD ["sh", "-c", "uv run gunicorn -b 0.0.0.0:${PORT:-5001} wsgi:app --workers 1 --threads 4 --timeout 120"]
```

- [ ] **Step 2: Docker build 验证**

Run:

```bash
docker build -t photo-privacy .
```

Expected: 构建成功。

- [ ] **Step 3: Commit**

```bash
git add Dockerfile
git commit -m "chore: use uv in docker build"
```

---

### Task 4: GitHub Actions（测试 + requirements.txt 同步校验）

**Files:**
- Create: `/Users/hao/Developer/photo/.github/workflows/ci.yml`

- [ ] **Step 1: 新建 workflow**

内容：

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up uv
        uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.11"

      - name: Install dependencies (locked)
        run: uv sync --frozen

      - name: Run unit tests
        run: uv run python -m unittest -q tests_test_face_processor.py

      - name: Verify exported requirements.txt is up-to-date
        run: |
          uv export --format requirements-txt --no-hashes -o requirements.txt
          git diff --exit-code requirements.txt
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add uv-based test workflow"
```

---

### Task 5: 全量验证与推送

**Files:**
- Modify: (none, verification only)

- [ ] **Step 1: 本地验证**

Run:

```bash
uv sync --frozen
uv run python -m py_compile app.py face_processor.py wsgi.py
uv run python -m unittest -q tests_test_face_processor.py
uv export --format requirements-txt --no-hashes -o requirements.txt
git diff --exit-code requirements.txt
```

Expected: 所有命令成功；requirements.txt 无 diff。

- [ ] **Step 2: Push**

Run:

```bash
git push origin main
```

