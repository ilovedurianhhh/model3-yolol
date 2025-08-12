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
import asyncio
import uvicorn
import json
from typing import List
from typing import List, Dict, Optional, Union

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CloudPose Server API", version="1.0.0")

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
    boxes: List[BoundingBox]
    keypoints: List[List[KeyPoint]]
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
        model = YOLO('./yolo11l-pose.pt')  # 修改这一行
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
                    keypoint_list.append(KeyPoint(x=float(x), y=float(y), p=float(conf)))
                
                keypoints_list.append(keypoint_list)
    
    end_postprocess = time.time()
    
    # 计算处理时间（毫秒）
    speed_preprocess = (end_preprocess - start_preprocess) * 1000
    speed_inference = (end_inference - start_inference) * 1000
    speed_postprocess = (end_postprocess - start_postprocess) * 1000
    
    return boxes, keypoints_list, count, speed_preprocess, speed_inference, speed_postprocess

def annotate_image(image: np.ndarray, boxes: list, keypoints_list: list) -> np.ndarray:
    """在图像上标注检测结果"""
    annotated_image = image.copy()
    
    # 绘制边界框
    for box in boxes:
        x1, y1 = int(box.x), int(box.y)
        x2, y2 = int(box.x + box.width), int(box.y + box.height)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_image, f'{box.probability:.2f}', (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 绘制关键点
    for keypoints in keypoints_list:
        for i, kp in enumerate(keypoints):
            if kp.p > 0.5:  # 只绘制置信度高的关键点
                cv2.circle(annotated_image, (int(kp.x), int(kp.y)), 5, (0, 0, 255), -1)
                cv2.putText(annotated_image, str(i), (int(kp.x) + 5, int(kp.y) - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
        # 绘制连接线（简化版本）
        connections = [[5, 6], [5, 11], [6, 12], [11, 12]]  # 肩膀和髋部连接
        for connection in connections:
            if (connection[0] < len(keypoints) and connection[1] < len(keypoints) and 
                keypoints[connection[0]].p > 0.5 and keypoints[connection[1]].p > 0.5):
                p1 = (int(keypoints[connection[0]].x), int(keypoints[connection[0]].y))
                p2 = (int(keypoints[connection[1]].x), int(keypoints[connection[1]].y))
                cv2.line(annotated_image, p1, p2, (0, 255, 255), 2)
    
    return annotated_image

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

@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    load_model()

@app.get("/")
async def root():
    return {"message": "CloudPose Server API is running"}

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "model_loaded": model is not None}

# 兼容客户端的主要API端点
@app.post("/api/pose_estimation", response_model=PoseResponse)
async def pose_estimation(request_data: str):
    """姿态检测JSON API - 兼容客户端格式"""
    try:
        # 客户端发送的是json.dumps(data)，需要先解析
        data = json.loads(request_data)
        request = ImageRequest(**data)
        
        logger.info(f"Processing pose estimation request: {request.id}")
        
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # 使用线程处理请求以支持并发
        result = await asyncio.get_event_loop().run_in_executor(
            None, process_request, request, False
        )
        
        return result
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        logger.error(f"Error in pose estimation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 标准的FastAPI端点（用于其他客户端）
@app.post("/api/pose_estimation_standard", response_model=PoseResponse)
async def pose_estimation_standard(request: ImageRequest):
    """标准姿态检测JSON API"""
    logger.info(f"Processing standard pose estimation request: {request.id}")
    
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

@app.post("/api/pose_detection", response_model=PoseResponse)
async def pose_detection(request: ImageRequest):
    """姿态检测API - 匹配客户端调用的端点"""
    logger.info(f"Processing pose detection request: {request.id}")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # 使用线程处理请求以支持并发
    result = await asyncio.get_event_loop().run_in_executor(
        None, process_request, request, False
    )
    
    return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CloudPose Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=60000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    logger.info(f"Starting CloudPose Server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)