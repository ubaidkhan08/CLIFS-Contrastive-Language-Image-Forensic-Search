import streamlit as st
import cv2
from PIL import Image
import torch
import clip
import numpy as np
import torchvision

# Load the CLIP model and normalize images.
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device=device)
model.eval()

def CLIFS(video_path, text_prompt):
    # Load the video and extract frames.
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(frame))  # convert numpy array to PIL Image
    cap.release()

    # Process each frame and caption with CLIP.
    with torch.no_grad():
        
        #text_prompt = ["A truck with the text 'odwalla'"]
        text_prompt = clip.tokenize(text_prompt).to(device)
        images = [preprocess(frame).unsqueeze(0).to(device) for frame in frames]
        features = model.encode_image(torch.cat(images))

        # Compute the similarity between each caption and frame.
        similarities = features @ model.encode_text(text_prompt).T
        best_match_idx = torch.argmax(similarities)

        # Convert PIL image to NumPy array
        image_array = np.asarray(frames[best_match_idx])

        # Convert the color format of the image
        bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # Save the output frame
        cv2.imwrite('ouput.png', bgr_image)
        output_image = Image.fromarray(bgr_image)
        st.image(output_image, caption='Output Image', use_column_width=True)

# Streamlit app
st.title('CLIFS App')
st.write('This app uses the CLIFS algorithm to find the best match for a text prompt in a video.')

# File upload
video_file = st.file_uploader('Upload a video file', type=['mp4', 'avi', 'mkv'])

# Text prompt
text_prompt = st.text_input('Enter a text prompt')

# Process video and display output image
if video_file and text_prompt:
    CLIFS(video_file, text_prompt)
else:
    st.write('Please upload a video file and enter a text prompt to get started.')