import streamlit as st
import numpy as np
import tensorflow as tf
import requests
from tensorflow.keras.preprocessing import image
from PIL import Image
import io


helmet_model = 'https://www.dropbox.com/scl/fi/x58ezyimczi70qu836gsv/helmet_detection_model5.h5?rlkey=lylheycc8n0rr9pp0inzrlnce&st=lv9pjmy7&dl=1'
response = requests.get(helmet_model)
with open('helmet_detection_model5.h5', 'wb') as f:
    f.write(response.content)

# โหลดโมเดลสำหรับตรวจหมวกกันน็อค
# helmet_model = tf.keras.models.load_model('helmet_detection_model5.h5')

class_names_helmet = ['Without Helmet', 'With Helmet']

# ฟังก์ชันสำหรับแปลงภาพจากกล้องให้สามารถประมวลผลได้
def prepare_image(img, img_width=150, img_height=150):
    img = img.resize((img_width, img_height))  # ปรับขนาดภาพ
    img_array = np.array(img) / 255.0  # ปรับขนาดภาพให้มีค่าในช่วง 0-1
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ฟังก์ชันสำหรับตรวจสอบหมวกกันน็อค
def classify_helmet(img):
    prediction = helmet_model.predict(img)
    predicted_helmet_class = np.argmax(prediction)
    helmet_confidence = np.max(prediction)
    return predicted_helmet_class, helmet_confidence

def main():
    st.title("Helmet Detection")

    # ใช้กล้องเพื่อจับภาพ
    captured_image = st.camera_input("Capture Image")

    if captured_image:
        # โหลดภาพจากกล้อง
        img = Image.open(captured_image)
        st.image(img, caption="Captured Image", use_column_width=True)

        # เตรียมภาพเพื่อใช้กับโมเดล
        img_array = prepare_image(img)

        # ตรวจสอบหมวกกันน็อค
        predicted_helmet_class, helmet_confidence = classify_helmet(img_array)
        helmet_result_text = f"Helmet Prediction: {class_names_helmet[predicted_helmet_class]} with confidence {helmet_confidence * 100:.2f}%"
        st.write(helmet_result_text)

if __name__ == "__main__":
    main()