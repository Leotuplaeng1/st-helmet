from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import onnx
import onnxruntime as ort
import numpy as np
import base64
import re
import io
import requests
from PIL import Image
import paho.mqtt.client as mqtt
import os  # เพิ่มการใช้งาน os สำหรับสร้างไฟล์ชั่วคราว

app = Flask(__name__)
CORS(app)

# URLs ของโมเดล
vehicle_model_url = 'https://www.dropbox.com/scl/fi/zlmm6k6u96qgemddm4bzn/vehicle_classification.onnx?rlkey=pvcrm0bv2vxczmhou6bfqoa9h&st=4pplshvw&dl=1'
helmet_model_url = 'https://www.dropbox.com/scl/fi/gd8djpwcr9itx3nkxgjbr/helmet_detection_model.onnx?rlkey=f5p5ezg76wdicvcuzfw4kzuub&st=bpc3tiuu&dl=1'

# ฟังก์ชันสำหรับการโหลดโมเดลจาก URL
def load_model_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        model_data = io.BytesIO(response.content)  # โหลดโมเดลในหน่วยความจำ
        onnx_model = onnx.load(model_data)  # โหลดโมเดลด้วย onnx
        
        # ใช้ os เพื่อให้ได้ที่อยู่ไฟล์ที่เหมาะสมใน Windows
        model_path = os.path.join(os.getcwd(), "temp_model.onnx")
        
        onnx.save(onnx_model, model_path)  # บันทึกโมเดลลงในไฟล์ชั่วคราวในโฟลเดอร์ปัจจุบัน
        return ort.InferenceSession(model_path)  # โหลดโมเดลจากไฟล์ที่บันทึก
    else:
        raise Exception(f"Failed to download model from {url}")

# โหลดโมเดลจาก Dropbox
vehicle_session = load_model_from_url(vehicle_model_url)
helmet_session = load_model_from_url(helmet_model_url)

# รายละเอียดการเชื่อมต่อ MQTT
broker = "a2c612b479f9426a8cf0ce535dc46ef3.s1.eu.hivemq.cloud"
port = 8883
username = "python"
password = "0882501531Za"
topic = "esp8266/buzzer"

client = mqtt.Client()
client.tls_set()
client.username_pw_set(username, password)
client.connect(broker, port)

class_names = ['bike', 'car']
helmet_class_names = ['no helmet', 'helmet']
CONFIDENCE_THRESHOLD = 0.8  # กำหนดค่าความมั่นใจขั้นต่ำที่ต้องการ

# ฟังก์ชันสำหรับการเตรียมภาพ
def prepare_image(img, img_width=150, img_height=150):
    img = img.resize((img_width, img_height))  # ปรับขนาดภาพเป็น 150x150 ตามที่โมเดลคาดหวัง
    img_array = np.array(img).astype('float32')  # เปลี่ยนเป็น numpy array
    img_array = img_array / 255.0  # ทำให้ค่าพิกเซลอยู่ในช่วง [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # เพิ่มมิติที่ 0 สำหรับ batch size
    return img_array  # ผลลัพธ์คือ (1, 150, 150, 3)

# ฟังก์ชันสำหรับการทำนายประเภทพาหนะ
def classify_vehicle(img):
    try:
        img_array = prepare_image(img)
        input_name = vehicle_session.get_inputs()[0].name
        prediction = vehicle_session.run(None, {input_name: img_array})
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        return predicted_class, confidence
    except Exception as e:
        raise Exception(f"Error in vehicle classification: {str(e)}")

# ฟังก์ชันสำหรับการทำนายการใส่หมวกกันน็อค
def classify_helmet(img):
    try:
        img_array = prepare_image(img)
        input_name = helmet_session.get_inputs()[0].name
        outputs = helmet_session.run(None, {input_name: img_array})
        prediction = outputs[0]
        predicted_helmet_class = np.argmax(prediction)
        helmet_confidence = np.max(prediction)
        return predicted_helmet_class, helmet_confidence
    except Exception as e:
        raise Exception(f"Error in helmet classification: {str(e)}")

# Route สำหรับอัปโหลดภาพและประมวลผล
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' in request.form:
        try:
            # แปลงภาพจาก Base64
            image_data = request.form['image']
            image_data = re.sub('^data:image/.+;base64,', '', image_data)
            img = Image.open(io.BytesIO(base64.b64decode(image_data)))
            img = img.convert('RGB')
        except Exception as e:
            return jsonify({'error': 'Failed to decode image', 'details': str(e)}), 400

        # ตรวจจับประเภทพาหนะ
        predicted_vehicle_class, vehicle_confidence = classify_vehicle(img)

        # ตรวจสอบว่าความมั่นใจสูงพอ
        if vehicle_confidence < CONFIDENCE_THRESHOLD:
            return jsonify({'message': "No action required for low confidence prediction"}), 200

        # ตรวจสอบวัตถุที่ทำนายว่าเป็นประเภทที่เราสนใจเท่านั้น
        if predicted_vehicle_class == 0:  # มอเตอร์ไซค์
            client.publish(topic, "reset_camera")  # รีเซ็ตกล้องและถ่ายภาพใหม่
            predicted_helmet_class, helmet_confidence = classify_helmet(img)

            if predicted_helmet_class == 0:  # ใส่หมวก
                result = {
                    'vehicle_result': f"Prediction: Bike with confidence {vehicle_confidence * 100:.2f}%",
                    'helmet_result': f"Helmet detected with confidence {helmet_confidence * 100:.2f}%",
                    'action': "pass"
                }
                client.publish(topic, "pass")  # ส่งสัญญาณผ่าน
            else:  # ไม่ใส่หมวก
                result = {
                    'vehicle_result': f"Prediction: Bike with confidence {vehicle_confidence * 100:.2f}%",
                    'helmet_result': f"No helmet detected with confidence {helmet_confidence * 100:.2f}%",
                    'action': "no_pass"
                }
                client.publish(topic, "no helmet")  # ไม่ให้ผ่าน

        elif predicted_vehicle_class == 1:  # รถยนต์
            result = {
                'vehicle_result': f"Prediction: Car with confidence {vehicle_confidence * 100:.2f}%",
                'action': "pass"
            }
            client.publish(topic, "pass")  # ส่งสัญญาณผ่าน

        else:  # กรณีไม่ใช่รถยนต์หรือมอเตอร์ไซค์
            return jsonify({'message': "No action required for non-vehicle objects"}), 200

        return jsonify(result)
    
    else:
        return "No image found", 400

# หน้าเว็บหลัก
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

