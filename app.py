import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Face Recognition App", layout="centered")
st.title("🧠 Face Recognition System")

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# Label mapping (same order as training)
labels = {
    0: "Zakir",
    1: "Ali"
}

st.markdown("📷 **Upload a clear face image**")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    img_np = np.array(image)

    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("❌ No face detected")
    else:
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face)

            name = labels.get(label, "Unknown")

            cv2.rectangle(img_np, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(
                img_np,
                f"{name} ({int(confidence)})",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

        st.image(img_np, caption="Recognition Result", channels="BGR")
