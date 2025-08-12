from locust import HttpUser, task, between
import base64
import json
import uuid
import os
import random
from io import BytesIO
from PIL import Image

class CloudPoseUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """在用户开始时准备测试图像"""
        self.images = self.load_test_images()
        print(f"Loaded {len(self.images)} test images")
    
    def load_test_images(self):
        """加载测试图像并转换为base64"""
        images = []
        
        # 查找images目录下的图像文件
        image_dir = "./images"  # 存放测试图像的目录
        
        if os.path.exists(image_dir):
            for filename in os.listdir(image_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(image_dir, filename)
                    try:
                        with open(image_path, 'rb') as f:
                            image_data = f.read()
                            base64_image = base64.b64encode(image_data).decode('utf-8')
                            images.append(base64_image)
                    except Exception as e:
                        print(f"Error loading image {filename}: {e}")
        
        # 如果没有找到图像文件，生成一些测试图像
        if not images:
            print("No test images found, generating synthetic images...")
            for i in range(10):
                # 生成简单的测试图像
                img = Image.new('RGB', (640, 480), color=(random.randint(0, 255), 
                                                         random.randint(0, 255), 
                                                         random.randint(0, 255)))
                
                # 转换为base64
                buffer = BytesIO()
                img.save(buffer, format='JPEG')
                image_data = buffer.getvalue()
                base64_image = base64.b64encode(image_data).decode('utf-8')
                images.append(base64_image)
        
        return images
    
    @task(weight=3)
    def pose_estimation_json(self):
        """测试JSON API端点"""
        if not self.images:
            return
        
        # 随机选择一个图像
        image_data = random.choice(self.images)
        
        # 创建请求负载
        payload = {
            "id": str(uuid.uuid4()),
            "image": image_data
        }
        
        # 发送POST请求
        with self.client.post(
            "/api/pose_estimation",
            json=payload,
            headers={"Content-Type": "application/json"},
            catch_response=True,
            name="pose_estimation_json"
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    # 验证响应格式
                    if "id" in result and "count" in result and "keypoints" in result:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
    
    @task(weight=1)
    def pose_estimation_annotation(self):
        """测试图像标注API端点"""
        if not self.images:
            return
        
        # 随机选择一个图像
        image_data = random.choice(self.images)
        
        # 创建请求负载
        payload = {
            "id": str(uuid.uuid4()),
            "image": image_data
        }
        
        # 发送POST请求
        with self.client.post(
            "/api/pose_estimation_annotation",
            json=payload,
            headers={"Content-Type": "application/json"},
            catch_response=True,
            name="pose_estimation_annotation"
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    # 验证响应格式
                    if "id" in result and "annotated_image" in result:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
    
    @task(weight=1)
    def health_check(self):
        """健康检查端点"""
        with self.client.get("/health", name="health_check") as response:
            if response.status_code != 200:
                response.failure(f"Health check failed: {response.status_code}")

# 如果需要自定义配置，可以使用命令行参数运行：
# locust -f locustfile.py --host=http://47.83.121.92:30000 -u 10 -r 2 -t 300s

class CloudPoseStressTest(HttpUser):
    """专门用于压力测试的用户类"""
    wait_time = between(0.1, 0.5)  # 更短的等待时间
    
    def on_start(self):
        self.images = self.load_test_images()
    
    def load_test_images(self):
        """简化的图像加载，只生成少量图像以减少内存使用"""
        images = []
        for i in range(3):  # 只生成3个测试图像
            img = Image.new('RGB', (320, 240), color=(128, 128, 128))  # 较小的图像
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            image_data = buffer.getvalue()
            base64_image = base64.b64encode(image_data).decode('utf-8')
            images.append(base64_image)
        return images
    
    @task
    def stress_test_pose_estimation(self):
        """压力测试主要API"""
        if not self.images:
            return
        
        payload = {
            "id": str(uuid.uuid4()),
            "image": random.choice(self.images)
        }
        
        self.client.post(
            "/api/pose_estimation",
            json=payload,
            headers={"Content-Type": "application/json"},
            name="stress_pose_estimation"
        )