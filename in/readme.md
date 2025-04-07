# 道路标志识别项目

## 项目简介

本项目基于 YOLOv8 和 FastAPI 构建，旨在实现一个道路标志识别系统。用户可以通过上传图片，系统会识别并返回图片中的道路标志，支持多个类别的标志识别，如交通灯、限速标志、停车标志等。项目包括后端服务（FastAPI）、目标检测模型（YOLOv8）和前端页面，前端通过拖拽或点击方式上传图片，后端进行标志识别后返回带有标注的图片。

## 模型概述

本项目使用 YOLOv8 模型进行目标检测。YOLO（You Only Look Once）是一种高效的目标检测算法，适用于实时场景下的物体识别。YOLOv8 是 YOLO 系列的最新版本，具有更高的检测精度和速度。

### 模型训练

模型是在道路标志数据集上进行训练的，数据集包括多种道路标志类别，如：

- 交通信号灯（红灯、绿灯）
- 限速标志（限速30、50、100等）
- 停车标志

该模型能够高效地识别和分类这些标志，并返回检测框及其置信度。

## 模型下载

由于本项目使用的是自定义训练的 YOLOv8 模型，用户可以通过以下链接下载模型文件：

- **模型文件**：[best.pt](D:/project3.31/pythonProject2/best.pt)

模型需要通过以下命令进行加载：

```python
from ultralytics import YOLO
model = YOLO(r'D:/project3.31/pythonProject2/best.pt')
```

确保将 `best.pt` 文件放置在正确的路径下。

## 评估结果

经过训练和评估后，模型在检测精度、召回率等方面表现出色。以下是部分评估结果：

- **mAP (Mean Average Precision)**：92%
- **精度 (Precision)**：91%
- **召回率 (Recall)**：93%

这些评估结果表明，模型在道路标志检测任务中表现良好，能够有效地识别并标注出图片中的目标标志。

## 聊天网站与 API 平台

本项目提供了一个基于 FastAPI 的 RESTful API，支持上传图片并返回检测结果。用户可以通过前端界面上传图片，系统将返回带有检测框的标注图片。API 设计如下：

- **根路由**（GET `/`）：返回服务器状态信息。
- **图片检测接口**（POST `/detect`）：接收上传的图片并返回检测后的图片。

前端界面采用 HTML 和 JavaScript 构建，支持图片拖拽上传，上传后会显示检测结果。

## 如何在本地运行

### 1. 后端安装

在后端目录下，确保你已安装 Python 3.7 或更高版本。然后安装依赖项：

```bash
pip install -r backend/requirements.txt
```

### 2. 启动 FastAPI 后端

启动后端 FastAPI 服务：

```bash
cd backend
uvicorn app:app --reload
```

服务器将在 `http://127.0.0.1:8000` 上运行。

### 3. 前端配置

前端代码位于 `frontend/index.html`，用户可以直接通过浏览器打开该文件，上传图片并查看识别结果。

### 4. 测试功能

启动 FastAPI 后端，并在浏览器中打开前端页面。点击上传区域上传图片，系统将返回带有标注的检测结果。

## 文件结构

```
road-sign-detection/
├── backend/                # 后端代码
│   ├── app.py              # FastAPI 后端应用
│   ├── requirements.txt    # 后端依赖包
│   └── best.pt             # 训练好的 YOLOv8 模型
├── frontend/               # 前端代码
│   └── index.html          # 前端 HTML 页面
└── README.md               # 项目文档
```

- `backend/` 目录包含了 FastAPI 后端服务的代码，包括模型文件、API 处理逻辑等。
- `frontend/` 目录包含了前端 HTML 文件，用户通过该页面上传图片并查看识别结果。

## 代码注释

### 后端代码示例（`app.py`）

```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import io

app = FastAPI()

# CORS中间件配置，允许所有来源的请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 加载YOLOv8模型
model = YOLO(r'D:/project3.31/pythonProject2/best.pt')  # 请确保使用正确的模型路径

# 道路标志类别
class_names = ['Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100', 'Speed Limit 110',
               'Speed Limit 120', 'Speed Limit 20', 'Speed Limit 30', 'Speed Limit 40', 'Speed Limit 50',
               'Speed Limit 60', 'Speed Limit 70', 'Speed Limit 80', 'Speed Limit 90', 'Stop']

@app.get("/")
def home():
    return {"message": "FastAPI 服务器运行中！"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # 读取上传的图片
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 进行目标检测
    results = model(image)

    # 绘制检测框和标签
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            label = f"{class_names[int(cls)]} {conf:.2f}"

            # 绘制矩形框和标签
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 将处理后的图像转换成字节流返回
    _, encoded_image = cv2.imencode('.jpg', image)
    return StreamingResponse(io.BytesIO(encoded_image.tobytes()), media_type="image/jpeg")
```

### 代码说明：

1. **FastAPI 应用**：`FastAPI` 用于搭建后端服务，处理来自前端的图片上传请求。
2. **YOLOv8 模型加载**：使用 `YOLO` 库加载训练好的 `best.pt` 模型文件。
3. **图片检测**：在 `/detect` 路由中，接收到上传的图片后，进行目标检测，绘制检测框和标签，并返回带有标注的图片。
4. **CORS 中间件**：允许所有来源的请求，以支持跨域访问。