# DIffCAD Agent — 植物气孔智能检测与表型分析平台

基于 YOLO-OBB 深度学习模型的气孔（Stomata）与气孔开度（Aperture）检测工具。上传显微镜图像即可自动识别气孔、计算密度、开度尺寸、气孔导度等表型指标，支持交互式对话分析与批量处理。

## 功能概览

- **单张 / 批量 / 视频 / 相机实时检测** — 支持多种输入源
- **旋转边界框 (OBB)** — 精准贴合椭圆形气孔，比传统矩形框更准确
- **双模型自动切换** — 根据植物类型（双子叶/单子叶）和采样方式（破坏性/非破坏性）加载对应权重
- **表型指标自动计算** — 气孔数量、密度、长宽、长宽比、开度面积、气孔导度
- **AI 对话助手** — 内置植物学知识库，可询问气孔功能、检测原理、数据分析建议
- **CSV 导出** — 一键导出批量检测结果
- **Web 可视化界面** — 检测叠加图、指标面板、批量导航
- **轻量部署** — 支持 ONNX Runtime（无需 PyTorch，仅约 80MB 内存），适合免费云服务

## 快速开始

### 环境要求

- Python 3.10 ~ 3.11
- 内存：≥ 512 MB（ONNX 模式）/ ≥ 2 GB（PyTorch 模式）

### 1. 克隆仓库

```bash
git clone https://github.com/wlhu1/diffcad-agent.git
cd diffcad-agent
```

### 2. 下载模型权重

从 [Releases](https://github.com/wlhu1/diffcad-agent/releases) 页面下载对应的 `.onnx` 或 `.pt` 权重文件，放入 `weights/` 目录。

必需的权重文件：

```
weights/
├── dicotyledons_nondestructive.onnx   # 双子叶 + 非破坏性采样
├── dicotyledons_destructive.onnx      # 双子叶 + 破坏性采样
└── (可选) monocotyledons_*.onnx       # 单子叶植物模型
```

> 权重文件较大，未包含在仓库中，请单独下载。

### 3. 安装依赖

**轻量模式（仅 ONNX，适合低配服务器）：**

```bash
pip install -r requirements_render.txt
```

**完整模式（含 PyTorch，支持 GPU 加速）：**

```bash
pip install -r requirements.txt
```

### 4. 启动服务

```bash
python backend_api.py
```

浏览器访问 `http://localhost:5000` 进入首页，或访问 `http://localhost:5000/agent` 进入智能体对话界面。

## 使用指南

### Web 界面

| 页面 | 路径 | 说明 |
|------|------|------|
| 首页 | `/` | 项目介绍、API 文档、在线测试控制台 |
| 智能体 | `/agent` | 对话式检测界面：上传图片 → 自动检测 → 查看指标 → AI 问答 |

### 检测流程

1. **选择植物类型** — 双子叶（dicotyledons，如拟南芥、大豆）或单子叶（monocotyledons，如水稻、小麦）
2. **选择采样方式** — 非破坏性（nondestructive，如叶印迹）或破坏性（destructive，如叶片固定）
3. **上传图像** — 支持单张、文件夹批量、视频文件、相机实时流
4. **查看结果** — 自动生成叠加标注图、表型指标面板
5. **导出数据** — 批量结果可导出为 CSV

### API 接口

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/status` | 服务状态、已加载模型列表、内存占用 |
| POST | `/api/detect/single` | 单张图像检测 |
| POST | `/api/detect/batch` | 批量检测（多文件上传） |
| POST | `/api/detect/video` | 视频文件检测 |
| POST | `/api/detect/frame` | 单帧快速检测（适合相机流） |
| POST | `/api/export/csv` | 导出当前会话结果为 CSV |
| POST | `/api/agent/chat` | AI 对话助手 |
| GET | `/api/session/results` | 获取当前会话所有检测结果 |
| POST | `/api/session/reset` | 重置会话 |

#### 单张检测示例

```bash
curl -X POST http://localhost:5000/api/detect/single \
  -F "image=@your_stomata_image.jpg" \
  -F "plant_type=dicotyledons" \
  -F "sample_type=nondestructive" \
  -F "conf=0.5" \
  -F "scale_um=100.0"
```

返回示例：

```json
{
  "success": true,
  "metrics": {
    "stoma_count": 45,
    "aperture_count": 38,
    "stoma_avg_height_um": 28.5,
    "stoma_avg_width_um": 18.2,
    "stoma_density_mm2": 120.5,
    "conductance": 0.32
  },
  "overlay_b64": "...",
  "boxes": [[cx, cy, w, h, angle, cls, conf], ...]
}
```

## 部署

### Docker

```bash
docker build -t diffcad-agent .
docker run -p 7860:7860 diffcad-agent
```

### Render（免费套餐）

项目已配置 `render.yaml`，可直接连接 GitHub 仓库一键部署：

1. 在 [Render](https://render.com) 创建新 Web Service
2. 连接 GitHub 仓库
3. Render 自动读取 `render.yaml` 配置（新加坡节点、ONNX 模式、512MB 适配）

### Hugging Face Spaces

已配置 `Dockerfile`，兼容 Hugging Face Spaces 的 Docker SDK 模式。

## 项目结构

```
diffcad-agent/
├── backend_api.py                  # Flask 后端 API 主程序
├── onnx_worker.py                  # ONNX 推理子进程（轻量，无 PyTorch 依赖）
├── detection_worker.py             # PyTorch 推理子进程（完整功能回退）
├── app.py                          # PyQt5 桌面版应用
├── landing_page.html               # 首页（API 文档 + 测试控制台）
├── stomata_agent_frontend.html     # 智能体对话界面（/agent）
├── stomata_agent_frontend_v8.html  # 备用前端
├── icons/                          # 界面图标资源
├── weights/                        # 模型权重（需单独下载）
├── uploads/                        # 上传文件临时目录
├── results/                        # 批量结果 CSV 导出目录
├── requirements.txt                # 完整依赖（含 PyTorch）
├── requirements_render.txt         # 轻量依赖（仅 ONNX，适合云部署）
├── render.yaml                     # Render 部署配置
├── Dockerfile                      # Docker 构建文件
├── Procfile                        # 进程启动配置
└── runtime.txt                     # Python 版本指定
```

## 常见问题

**Q: 为什么检测结果有很多框？**

可提高 `conf`（置信度阈值，默认 0.5）或降低 `iou`（IoU 阈值，默认 0.7）来过滤低质量检测。在高级面板中调整这些参数。

**Q: 如何确定 scale (μm/pixel)？**

- 已知视野法：显微镜视野宽度 (μm) ÷ 图像宽度 (pixel)
- 比例尺法：比例尺实际长度 (μm) ÷ 比例尺在图像中的像素数

**Q: ONNX 和 PyTorch 模式有什么区别？**

- ONNX 模式：内存占用约 80MB，仅 CPU，适合 512MB 云服务器
- PyTorch 模式：内存占用约 300MB+，支持 GPU，检测结果与 ONNX 一致

后端会自动优先使用 ONNX worker，不可用时回退到 PyTorch。

**Q: 支持单子叶植物吗？**

支持。在检测时选择 `plant_type=monocotyledons`，并下载对应的权重文件到 `weights/` 目录。

## 技术栈

- **检测模型**：YOLOv8n-OBB (Oriented Bounding Box)
- **推理引擎**：ONNX Runtime / PyTorch
- **后端**：Flask + Flask-CORS
- **图像处理**：OpenCV, NumPy
- **部署**：Gunicorn, Docker, Render

## 致谢

DIffCAD 由河南大学作物逆境改良国家重点实验室开发。检测模型基于 Ultralytics YOLOv8-OBB 架构训练。
