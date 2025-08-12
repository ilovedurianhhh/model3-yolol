from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import io
from PIL import Image
import time
import uuid
import threading
from ultralytics import YOLO
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CloudPose API", version="1.0.0")

# 全局变量存储模型
model = None

class ImageRequest(BaseModel):
    id: str
    image: str

class KeyPoint(BaseModel):
    x: float
    y: float
    p: float

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    probability: float

class PoseResponse(BaseModel):
    id: str
    count: int
    boxes: list[BoundingBox]
    keypoints: list[list[KeyPoint]]
    speed_preprocess: float
    speed_inference: float
    speed_postprocess: float

class AnnotatedImageResponse(BaseModel):
    id: str
    annotated_image: str

def load_model():
    """加载YOLO pose detection模型"""
    global model
    try:
        logger.info("Loading YOLO pose detection model...")
        # 使用提供的模型文件
        model = YOLO('./yolo11l-pose.pt')
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

def decode_base64_image(base64_string: str) -> np.ndarray:
    """将base64编码的图像解码为OpenCV格式"""
    try:
        # 解码base64字符串
        image_data = base64.b64decode(base64_string)
        # 转换为PIL图像
        pil_image = Image.open(io.BytesIO(image_data))
        # 转换为OpenCV格式 (BGR)
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return opencv_image
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image data")

def encode_image_to_base64(image: np.ndarray) -> str:
    """将OpenCV图像编码为base64字符串"""
    try:
        # 编码图像为JPEG格式
        _, buffer = cv2.imencode('.jpg', image)
        # 转换为base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    except Exception as e:
        logger.error(f"Failed to encode image to base64: {e}")
        raise HTTPException(status_code=500, detail="Failed to encode image")

def process_pose_detection(image: np.ndarray):
    """处理姿态检测"""
    global model
    
    # 预处理计时
    start_preprocess = time.time()
    # 这里可以添加图像预处理步骤，如果需要的话
    # 目前YOLO模型会自动处理输入图像
    end_preprocess = time.time()
    
    # 推理计时
    start_inference = time.time()
    results = model(image)
    end_inference = time.time()
    
    # 后处理计时
    start_postprocess = time.time()
    
    boxes = []
    keypoints_list = []
    count = 0
    
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                
                boxes.append(BoundingBox(
                    x=float(x1),
                    y=float(y1),
                    width=float(x2 - x1),
                    height=float(y2 - y1),
                    probability=float(confidence)
                ))
                count += 1
        
        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            for person_keypoints in result.keypoints.xy:
                person_keypoints_conf = result.keypoints.conf[0] if result.keypoints.conf is not None else [1.0] * len(person_keypoints)
                
                keypoint_list = []
                for i, (x, y) in enumerate(person_keypoints):
                    conf = person_keypoints_conf[i] if i < len(person_keypoints_conf) else 1.0
                    keypoint_list.append([float(x), float(y), float(conf)])
                
                keypoints_list.append(keypoint_list)
    
    end_postprocess = time.time()
    
    # 计算处理时间（毫秒）
    speed_preprocess = (end_preprocess - start_preprocess) * 1000
    speed_inference = (end_inference - start_inference) * 1000
    speed_postprocess = (end_postprocess - start_postprocess) * 1000
    
    return boxes, keypoints_list, count, speed_preprocess, speed_inference, speed_postprocess

def annotate_image(image: np.ndarray, boxes: list[BoundingBox], keypoints_list: list) -> np.ndarray:
    """在图像上标注关键点和边界框"""
    annotated_image = image.copy()
    
    # 绘制边界框
    for box in boxes:
        x1 = int(box.x)
        y1 = int(box.y)
        x2 = int(box.x + box.width)
        y2 = int(box.y + box.height)
        
        # 绘制边界框
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 显示置信度
        cv2.putText(annotated_image, f'{box.probability:.2f}', 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 绘制关键点
    for person_keypoints in keypoints_list:
        for i, (x, y, conf) in enumerate(person_keypoints):
            if conf > 0.5:  # 只绘制置信度高的关键点
                cv2.circle(annotated_image, (int(x), int(y)), 3, (0, 0, 255), -1)
                cv2.putText(annotated_image, str(i), 
                           (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
        # 连接关键点（COCO格式的连接）
        connections = [
            [0, 1], [0, 2], [1, 3], [2, 4],  # 头部
            [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # 上肢
            [5, 11], [6, 12], [11, 12],  # 躯干
            [11, 13], [13, 15], [12, 14], [14, 16]  # 下肢
        ]
        
        for connection in connections:
            if len(person_keypoints) > max(connection):
                p1_x, p1_y, p1_conf = person_keypoints[connection[0]]
                p2_x, p2_y, p2_conf = person_keypoints[connection[1]]
                
                if p1_conf > 0.5 and p2_conf > 0.5:
                    cv2.line(annotated_image, (int(p1_x), int(p1_y)), 
                            (int(p2_x), int(p2_y)), (255, 0, 0), 2)
    
    return annotated_image

@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    load_model()

@app.get("/")
async def root():
    return {"message": "CloudPose API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

def process_request(request: ImageRequest, return_image: bool = False):
    """处理请求的线程函数"""
    try:
        # 解码图像
        image = decode_base64_image(request.image)
        
        # 处理姿态检测
        boxes, keypoints_list, count, speed_preprocess, speed_inference, speed_postprocess = process_pose_detection(image)
        
        if return_image:
            # 生成标注图像
            annotated_image = annotate_image(image, boxes, keypoints_list)
            encoded_image = encode_image_to_base64(annotated_image)
            return AnnotatedImageResponse(
                id=request.id,
                annotated_image=encoded_image
            )
        else:
            return PoseResponse(
                id=request.id,
                count=count,
                boxes=boxes,
                keypoints=keypoints_list,
                speed_preprocess=speed_preprocess,
                speed_inference=speed_inference,
                speed_postprocess=speed_postprocess
            )
    
    except Exception as e:
        logger.error(f"Error processing request {request.id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pose_estimation", response_model=PoseResponse)
async def pose_estimation(request: ImageRequest):
    """姿态检测JSON API"""
    logger.info(f"Processing pose estimation request: {request.id}")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # 使用线程处理请求以支持并发
    result = await asyncio.get_event_loop().run_in_executor(
        None, process_request, request, False
    )
    
    return result

@app.post("/api/pose_estimation_annotation")
async def pose_estimation_annotation(request: ImageRequest):
    """姿态检测图像标注API"""
    logger.info(f"Processing pose estimation annotation request: {request.id}")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # 使用线程处理请求以支持并发
    result = await asyncio.get_event_loop().run_in_executor(
        None, process_request, request, True
    )
    
    return result

if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    # 运行服务器
    uvicorn.run(app, host="0.0.0.0", port=60000)