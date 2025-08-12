# Generate the parallel requests based on the ThreadPool Executor
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
import sys
import time
import glob
import requests
import threading
import uuid
import base64
import json
import os

#send http request
def call_cloudpose_service(image):
    try:
        # 构建完整的API端点URL - 使用标准端点
        base_url = str(sys.argv[2])
        if not base_url.endswith('/'):
            base_url += '/'
        url = base_url + 'api/pose_estimation_standard'
        
        data = {}
        #generate uuid for image
        img_id = uuid.uuid5(uuid.NAMESPACE_OID, image)
        # Encode image into base64 string
        with open (image, 'rb') as image_file:
            data['image'] =  base64.b64encode(image_file.read()).decode('utf-8')

        data ['id'] = str(img_id)
        headers = {'Content-Type': 'application/json'}

        # 发送标准的JSON请求
        response = requests.post(url, json=data, headers=headers)

        if response.ok:
            output = "Thread : {},  input image: {},  output:{}".format(threading.current_thread().getName(),
                                                                        image,  response.text)
            print(output)
        else:
            print ("Error, response status:{}".format(response))
            # 添加详细错误信息
            try:
                print("Error details:", response.text)
            except:
                pass

    except Exception as e:
        print("Exception in webservice call: {}".format(e))

# gets list of all images path from the input folder
def get_images_to_be_processed(input_path):
    images = []
    
    # Check if input_path is a file
    if os.path.isfile(input_path):
        if input_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            images.append(input_path)
    # Check if input_path is a directory
    elif os.path.isdir(input_path):
        input_folder = os.path.join(input_path, "")
        for image_file in glob.iglob(input_folder + "*.jpg"):
            images.append(image_file)
        for image_file in glob.iglob(input_folder + "*.jpeg"):
            images.append(image_file)
        for image_file in glob.iglob(input_folder + "*.png"):
            images.append(image_file)
    
    return images

def main():
    ## provide arguments-> input folder/file, url, number of workers
    if len(sys.argv) != 4:
        raise ValueError("Arguments list is wrong. Please use the following format: {} {} {} {}".format("python cloudpose_client.py", "<input_folder_or_file>", "<URL>", "<number_of_workers>"))

    input_path = sys.argv[1]
    images = get_images_to_be_processed(input_path)
    num_images = len(images)
    
    # Add check for empty images list
    if num_images == 0:
        print("No valid image files found in: {}".format(input_path))
        print("Supported formats: .jpg, .jpeg, .png")
        return
    
    print("Found {} image(s) to process".format(num_images))
    
    num_workers = int(sys.argv[3])
    start_time = time.time()
    #create a worker thread to invoke the requests in parallel
    with PoolExecutor(max_workers=num_workers) as executor:
        for _ in executor.map(call_cloudpose_service, images):
            pass
    elapsed_time = time.time() - start_time
    print("Total time spent: {} average response time: {}".format(elapsed_time, elapsed_time/num_images))


if __name__ == "__main__":
    main()
