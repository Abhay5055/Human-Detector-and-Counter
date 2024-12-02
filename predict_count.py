import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2
import os

model = load_model('human_count_cnn.h5')
print('Model loaded successfully.')

def preprocess_image(img_path, target_size=(128, 128)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_human_count(img_array, model, threshold=0.5):
    raw_prediction = model.predict(img_array)[0][0]
    predicted_count = 0 if raw_prediction < threshold else round(raw_prediction)
    return predicted_count, raw_prediction

def predict_folder(folder_path, model, threshold=0.5):
    total_human_count = 0
    results = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_array = preprocess_image(file_path)
        predicted_count, raw_prediction = predict_human_count(img_array, model, threshold)
        results.append((filename, predicted_count))
        total_human_count += predicted_count

    for filename, predicted_count in results:
        img = load_img(os.path.join(folder_path, filename), target_size=(128, 128))
        plt.imshow(img)
        plt.title(f'Predicted Human Count: {predicted_count}')
        plt.axis('off')
        plt.show()
        print(f'File: {filename} | Predicted Human Count: {predicted_count}')

    print(f"Total Humans Detected in Folder: {total_human_count}")
    return total_human_count

def predict_video(video_path, model, threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    total_human_count = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (128, 128))
        frame_array = img_to_array(frame_resized) / 255.0
        frame_array = np.expand_dims(frame_array, axis=0)
        
        predicted_count, raw_prediction = predict_human_count(frame_array, model, threshold)
        total_human_count += predicted_count
        frame_count += 1

        cv2.putText(frame, f'Predicted Human Count: {predicted_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Video Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Total Humans Detected in Video: {total_human_count} across {frame_count} frames.")
    return total_human_count

def predict_camera(model, threshold=0.5):
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("Error: Could not open the camera. Please check your webcam.")
        return

    print("Press 'q' to quit the camera feed.")

    while True:
        ret, frame = cap.read() 
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        frame_resized = cv2.resize(frame, (128, 128))
        frame_array = img_to_array(frame_resized) / 255.0
        frame_array = np.expand_dims(frame_array, axis=0)

        predicted_count, raw_prediction = predict_human_count(frame_array, model, threshold)

        cv2.putText(frame, f'Predicted Human Count: {predicted_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Camera Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Camera feed closed by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Humans Detected in the camera :",predicted_count)
    print("Camera detection session ended.")

def select_folder():
    folder_path = filedialog.askdirectory(title="Select Folder")
    if folder_path:
        print(f"Processing folder: {folder_path}")
        predict_folder(folder_path, model)
    else:
        messagebox.showwarning("No Folder Selected", "Please select a folder to process.")

def select_video():
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")]
    )
    if video_path:
        print(f"Processing video: {video_path}")
        predict_video(video_path, model)
    else:
        messagebox.showwarning("No Video Selected", "Please select a video file to process.")

def start_camera_detection():
    print("Starting camera detection...")
    try:
        predict_camera(model)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start camera detection: {str(e)}")

def center_window(window, width=400, height=300):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    window.geometry(f"{width}x{height}+{x}+{y}")

def main():
    root = tk.Tk()
    root.title("Human Detection System")
    root.geometry("800x600")

    center_window(root, width=800, height=500)
    root.configure(bg="cyan")

    title_label = tk.Label(root, text="Human Detection System", font=("Helvetica", 25, "bold"), bg='cyan', fg='black')
    title_label.pack(pady=20)

    folder_button = tk.Button(root, text="Process a Folder of Images", font=("Helvetica", 20, "bold"), bg='lime', fg='black',
                               command=select_folder, width=30)
    folder_button.pack(pady=10)

    video_button = tk.Button(root, text="Process a Video File", font=("Helvetica", 20, "bold"), bg='lime', fg='black',
                              command=select_video, width=30)
    video_button.pack(pady=10)

    camera_button = tk.Button(root, text="Use Camera for Live Detection", font=("Helvetica", 20, "bold"), bg='lime', fg='black',
                               command=start_camera_detection, width=30)
    camera_button.pack(pady=10)

    exit_button = tk.Button(root, text="EXIT", font=("Helvetica", 20, "bold"), fg='red', borderwidth=2, command=root.quit, width=30)
    exit_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()

