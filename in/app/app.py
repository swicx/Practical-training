from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import io
import speech_recognition as sr
import pyttsx3
from io import BytesIO
import os
from datetime import datetime, timedelta
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request


app = FastAPI()

# 允许前端跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 加载YOLO模型
model_yolo = YOLO(r'D:\project3.31\pythonProject2\models\best.pt')

# 加载AI问答模型
local_model_path = r'C:\Users\86159\.cache\huggingface\hub\models--cchongyun--DeepSeek-R1-Drive-COT\snapshots\49570bfba2c1791cadb7b73dfe09b4397210491c'
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model_ai = AutoModelForCausalLM.from_pretrained(local_model_path)

# 道路标志类别
class_names = ['Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100', 'Speed Limit 110',
               'Speed Limit 120', 'Speed Limit 20', 'Speed Limit 30', 'Speed Limit 40', 'Speed Limit 50',
               'Speed Limit 60', 'Speed Limit 70', 'Speed Limit 80', 'Speed Limit 90', 'Stop',
               'bicycle', 'bus', 'car', 'motorbike', 'person']

# 请求体模型
class Message(BaseModel):
    text: str


templates = Jinja2Templates(directory="templates")

@app.get("/index_v2", response_class=HTMLResponse)
def get_index_v2(request: Request):
    return templates.TemplateResponse("index_v2.html", {"request": request})

# 存储检测结果和AI问答记录
detection_results = []
ai_chat_records = []
total_detections = 0
daily_detections = 0
daily_detections_date = datetime.now().date()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return {"message": "FastAPI 服务器运行中！"}

# 图片检测接口
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    global total_detections, daily_detections, daily_detections_date

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传正确格式的图片文件。")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="无法解析上传的图片，请确认格式正确。")

        # 使用YOLO模型进行检测
        results = model_yolo(image)
        label = ""
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

        # 保存图像到本地
        save_dir = "static/detections"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        filename = f"{timestamp}_{file.filename}"
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, image)

        image_url = f"/static/detections/{filename}"

        # 添加记录
        detection_results.append({
            "timestamp": datetime.now().isoformat(),
            "imagePath": image_url,
            "detectionResult": label
        })

        # 更新检测统计
        total_detections += 1
        if datetime.now().date() == daily_detections_date:
            daily_detections += 1
        else:
            daily_detections = 1
            daily_detections_date = datetime.now().date()

        return {"message": "检测成功", "imagePath": image_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器处理出错：{str(e)}")


# AI 问答接口
@app.post("/generate/")
async def generate_text(message: Message):
    try:
        inputs = tokenizer(message.text, return_tensors="pt").to(model_ai.device)
        outputs = model_ai.generate(**inputs, max_new_tokens=100)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        ai_chat_records.append({
            "timestamp": datetime.now().isoformat(),
            "userQuestion": message.text,
            "aiResponse": response
        })
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

# 语音转文本接口
@app.post("/speech-to-text/")
async def speech_to_text(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="请上传正确格式的音频文件。")
    try:
        contents = await file.read()
        audio = BytesIO(contents)

        recognizer = sr.Recognizer()
        with sr.AudioFile(audio) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data, language="zh-CN")
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"语音识别出错：{str(e)}")

# 语音合成接口（修复版本）
@app.post("/text-to-speech/")
async def text_to_speech(message: Message):
    try:
        text = message.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="文本不能为空")

        engine = pyttsx3.init()
        temp_path = "output.mp3"
        engine.save_to_file(text, temp_path)
        engine.runAndWait()

        # 确保文件存在并读取为内存数据
        with open(temp_path, "rb") as f:
            audio_bytes = f.read()

        return StreamingResponse(BytesIO(audio_bytes), media_type="audio/mpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"语音合成出错：{str(e)}")

# 获取图片检测结果
@app.get("/image-results")
def get_image_results():
    return detection_results

# 获取检测统计
@app.get("/detection-stats")
def get_detection_stats():
    dates = []
    counts = []
    current_date = datetime.now().date()
    for i in range(7):  # 获取最近7天的检测数据
        date = current_date - timedelta(days=i)
        dates.append(date.isoformat())
        count = sum(1 for result in detection_results if datetime.fromisoformat(result["timestamp"]).date() == date)
        counts.append(count)
    return {
        "total": total_detections,
        "daily": daily_detections,
        "dates": dates,
        "counts": counts
    }

# 获取AI问答记录
@app.get("/ai-chat-records")
def get_ai_chat_records():
    return ai_chat_records