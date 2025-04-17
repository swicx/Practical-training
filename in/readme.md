# 道路标志识别系统 🚦

本项目是一个基于 YOLOv8 + 本地语言模型的道路标志识别系统，集成了前后端交互、图片上传识别、语音交互、AI 问答、历史记录与数据统计等功能。

## 项目特点

- **图像识别功能**：通过上传图片识别道路标志，识别结果叠加在原图中显示。

  <img src="G:\read\1.png" alt="1" style="zoom:67%;" />

  <img src="G:\read\2.png" alt="2" style="zoom:67%;" />

  <img src="G:\read\3.png" alt="3" style="zoom:67%;" />

- **语音交互**：支持语音输入提问和语音播报 AI 回答。

  <img src="G:\read\4.png" alt="4" style="zoom:67%;" />

- **AI 问答系统**：可基于用户提问生成交通法规相关回答（集成本地语言模型）。

- **检测记录展示**：
  
  - 图片识别历史记录
  
    <img src="G:\read\5.png" alt="5" style="zoom: 33%;" />
  
  - AI 问答历史记录
  
    <img src="G:\read\6.png" alt="6" style="zoom: 33%;" />
  
  - 底部数据看板显示总检测数、当日检测数与每日趋势图（Chart.js）
  
    ![7](G:\read\7.png)
  
- **前后端分离**：前端使用原生 HTML/CSS/JS，后端使用 FastAPI + YOLOv8 + 本地语言模型。

---

## 项目结构

```
pythonProject2
├── /static
│   └── /img
│       └── background.jpg             # 背景图
├── /templates
│   └── index.html                     # 主页面
│   └── index_v2.html                  # 新版功能扩展页
├── /app
│   └── app.py                         # FastAPI 后端
│   └── records.json                   # 存储检测和问答历史记录
├── requirements.txt                   # Python 依赖
└── README.md                          # 项目说明
```

---

##  使用说明

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动后端

```bash
uvicorn app.app:app --reload
```

默认运行在：`http://127.0.0.1:8000`

### 3. 启动前端

使用浏览器打开：

```
/templates/index.html 或 index_v2.html
```

---

## 功能说明

| 功能模块     | 说明                                             |
| ------------ | ------------------------------------------------ |
| 图片上传检测 | 支持点击上传或拖拽图片，后端调用 YOLOv8 模型处理 |
| AI 问答      | 用户可输入问题，AI 模型生成回答并支持语音播放    |
| 语音输入输出 | 集成 Web Speech API，实现语音识别和语音播报      |
| 历史记录查看 | 包括图片检测记录和 AI 问答记录，弹窗展示         |
| 数据统计     | 底部面板展示总检测次数、当日检测数、折线图统计   |

---

##  接口说明（FastAPI）

### 1. `/detect`  
- `POST`
- 上传图片，返回带有检测框的图片

### 2. `/generate`  
- `POST`
- 提交问题文本，返回 AI 回答

### 3. `/dashboard`
- `GET`
- 返回统计数据（总数、当日数量、折线图所需数据）

---

##  模型说明

- **目标检测模型**：YOLOv8（模型权重路径：`D:/project3.31/pythonProject2/best.pt`）
- **语言模型**：基于本地 HuggingFace 模型，如 `DeepSeek-R1-Drive-COT`
- **语音模块**：原生 Web Speech API（无需后端）

---

## 页面展示

- 首页（`index.html`）：支持图片识别 + AI 问答 + 语音交互  
- 扩展页（`index_v2.html`）：新增检测记录弹窗、数据统计面板等功能

---

##  TODO（可选扩展）

- ✅ 实现拖拽上传识别
- ✅ 添加语音输入与播报
- ✅ 历史记录弹窗（左侧为检测记录，右侧为问答记录）
- ✅ 页面底部统计看板（折线图 + 总数显示）
- 🔜 添加用户权限登录与日志管理+图片结果可视化（如需）

---

##  作者信息

-  开发者：完全对
- 时间：2025年4月
- 如需联系：1794887861@qq.com 492039184@qq.com 1462785464@qq.com
