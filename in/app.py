from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import io

app = FastAPI()

# CORS中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源的请求
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法（GET, POST等）
    allow_headers=["*"],  # 允许所有请求头
)

# 加载YOLOv8模型
model = YOLO(r'D:\project3.31\pythonProject2\best.pt')  # 请确保使用正确的模型路径

# 道路标志类别
class_names = ['Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100', 'Speed Limit 110',
               'Speed Limit 120', 'Speed Limit 20', 'Speed Limit 30', 'Speed Limit 40', 'Speed Limit 50',
               'Speed Limit 60', 'Speed Limit 70', 'Speed Limit 80', 'Speed Limit 90', 'Stop',
               'bicycle', 'bus', 'car', 'motorbike', 'person']

@app.get("/")
def home():
    return {"message": "FastAPI 服务器运行中！"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # 1. 校验上传类型是否为图片
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传正确格式的图片文件。")

    try:
        # 2. 读取上传内容并解码为图像
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="无法解析上传的图片，请确认格式正确。")

        # 3. 使用 YOLO 模型进行目标检测
        results = model(image)

        # 4. 绘制检测结果
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = map(int, box)
                label = f"{class_names[int(cls)]} {conf:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 5. 将图像编码为 JPEG 并返回
        _, encoded_image = cv2.imencode('.jpg', image)
        return StreamingResponse(io.BytesIO(encoded_image.tobytes()), media_type="image/jpeg")

    except Exception as e:
        # 捕获其他未预料到的异常
        raise HTTPException(status_code=500, detail=f"服务器处理出错：{str(e)}")