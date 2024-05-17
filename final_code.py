import cv2
import numpy as np
import streamlit as st
import tempfile
import os
import base64
import time

# Load Haar Cascade classifier
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Function to detect bodies in a frame using Haar Cascade
def detect_bodies_haar(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = body_classifier.detectMultiScale(gray, 1.1, 3)
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    return frame

# Function to detect bodies in a frame using HOG
def detect_bodies_hog(frame):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects, _ = hog.detectMultiScale(gray_frame)
    
    # Create a black mask
    mask = np.zeros_like(gray_frame)
    
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Create a mask for the area inside the rectangle
        roi = mask[y:y+h, x:x+w]
        roi[:] = 255  # Set the region inside the rectangle to white
    
    # Apply the mask to the frame to remove the background inside the detected rectangles
    frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    return frame

# Function to process video based on selected feature
def process_video(selected_feature, uploaded_file):
    # Initialize video capture
    cap = cv2.VideoCapture(uploaded_file.name)
    
    # Initialize background subtractor (MOG2)
    MOG2_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    
    # Initialize variables for processing
    processed_frames = []
    
    # Process each frame in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply background subtraction (MOG2)
        if selected_feature == "MOG2":
            foreground_mask = MOG2_subtractor.apply(frame)
            processed_frame = cv2.cvtColor(foreground_mask, cv2.COLOR_GRAY2BGR)  # Convert to BGR format
        
        # Detect bodies using HOG
        elif selected_feature == "HOG":
            processed_frame = detect_bodies_hog(frame)
        
        # Append processed frame to the list
        processed_frames.append(processed_frame)
    
    # Release video capture
    cap.release()
    
    return processed_frames

# Function to generate download link for processed video
def generate_download_link(processed_frames, selected_feature):
    video_name = f"processed_video_{selected_feature}.mp4"
    fps = 30
    height, width, _ = processed_frames[0].shape
    video_writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for frame in processed_frames:
        video_writer.write(frame)
    
    video_writer.release()

    with open(video_name, "rb") as file:
        video_bytes = file.read()
    
    os.remove(video_name)
    
    href = f"<a href='data:video/mp4;base64,{base64.b64encode(video_bytes).decode()}' download='{video_name}' style='text-decoration: none; color: #ffffff; background-color: #4CAF50; padding: 10px 20px; border-radius: 5px;'>Click here to download processed video</a>"
    return href

# Function for the landing page
def landing_page():
    # Add an image at the top of the app
    # st.image("bike.jpg")  # Replace with your image path

    st.markdown(
        """
        <div style="display: flex; flex-direction: column; align-items: flex-start; text-align: left;">
            <h1 style="color: orange;">Welcome to Body Detection App</h1>
            <p>
                Aplikasi ini mendeteksi tubuh dalam video yang diunggah menggunakan teknik Computer Vision.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Penjelasan MOG2 dan HOG
    st.markdown(
        """
        <div style="display: flex; flex-direction: column; align-items: flex-start; text-align: left;">
            <h2 style="color: white;">Feature Information</h2>
            <p>
                MOG2 (Mixture of Gaussians) adalah metode yang digunakan untuk mendeteksi objek bergerak dalam video. Metode ini bekerja dengan memodelkan setiap piksel sebagai campuran beberapa distribusi Gaussian, yang memungkinkan sistem untuk memisahkan latar belakang dari objek bergerak (foreground). Model ini diperbarui dengan setiap frame baru, sehingga dapat menyesuaikan dengan perubahan pencahayaan atau objek statis dalam jangka waktu lama. MOG2 berguna dalam pemantauan video dan analisis lalu lintas untuk mendeteksi dan melacak objek yang bergerak.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.image("mog2.jpg")  # Replace with your MOG2 image path

    st.markdown(
        """
        <div style="display: flex; flex-direction: column; align-items: flex-start; text-align: left;">
            <p>
                HOG (Histogram of Oriented Gradients) adalah teknik yang digunakan untuk mendeteksi objek dalam gambar, terutama manusia. Teknik ini bekerja dengan membagi gambar menjadi sel-sel kecil, menghitung arah gradien pada setiap piksel, dan membuat histogram dari arah gradien tersebut. Histogram-histogram ini kemudian dinormalisasi dalam blok-blok yang lebih besar untuk membuatnya lebih tahan terhadap perubahan pencahayaan. Seluruh fitur ini digabungkan untuk merepresentasikan gambar, sehingga memungkinkan deteksi objek yang akurat.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.image("HOG.jpg")  # Replace with your HOG image path

    # # Button to navigate to the video detection feature
    # if st.button("GET STARTED", type='primary'):
    #     st.session_state.page = 'upload'



    # Button to navigate to the video detection feature
    # if st.button("GET STARTED", type='primary'):
    #     st.session_state.page = 'upload'


# Function for the video upload and processing page
def video_upload_page():
    st.title("Body Detection App")
    st.write("This app detects bodies in uploaded videos using different methods.")
    
    # Information about the features
    st.markdown(
        """
        ### Feature Information:
        - MOG2 (Mixture of Gaussians) : adalah sebuah teknik dalam pengolahan citra yang digunakan untuk melakukan subtraksi latar belakang (background subtraction). Teknik ini bekerja dengan memodelkan setiap piksel dalam citra sebagai campuran dari beberapa distribusi Gaussian.
        - HOG (Histogram of Oriented Gradients) : adalah teknik dalam visi komputer yang digunakan untuk deteksi objek, terutama efektif dalam mendeteksi manusia. Teknik ini bekerja dengan menganalisis distribusi gradien orientasi dalam sebuah gambar.
        """
    )

    # Select feature
    selected_feature = st.radio("Select Feature:", ("MOG2", "HOG"))

    # Upload video
    uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

    # Process video and display download link
    if uploaded_file is not None:
        if st.button("Process Video", key="process_button"):
            st.write("Processing video... This may take a moment.")
            processed_frames = process_video(selected_feature, uploaded_file)

            # Download processed video
            download_link = generate_download_link(processed_frames, selected_feature)
            st.markdown(download_link, unsafe_allow_html=True)

# Main function
def main():
    st.set_page_config(page_title="Body Detection App", page_icon=":guardsman:", layout="wide")

    # Navigation sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Landing Page", "Video Upload"])

    if page == "Landing Page":
        landing_page()
    elif page == "Video Upload":
        video_upload_page()

if __name__ == "__main__":
    main()
