import os
import cv2
from tqdm import tqdm

# Set this to number of frames you want per video
FRAMES_PER_VIDEO = 10

# Input base path (where FaceForensics++ was downloaded)
BASE_INPUT_PATH = "data"
# Output directory where extracted frames will be saved
OUTPUT_DIR = "frames"

# Dataset names and subpaths
DATASETS = {
    'original_youtube_videos': 'misc/downloaded_youtube_videos.zip',
    'original_youtube_videos_info': 'misc/downloaded_youtube_videos_info.zip',
    'original': 'original_sequences/youtube',
    'DeepFakeDetection_original': 'original_sequences/actors',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'DeepFakeDetection': 'manipulated_sequences/DeepFakeDetection',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceShifter': 'manipulated_sequences/FaceShifter',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures'
}


def extract_frames_from_video(video_path, save_dir, num_frames=10):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        print(f"⚠️ Skipping {video_path}: no frames found.")
        return

    step = max(total_frames // num_frames, 1)
    frame_indices = [i * step for i in range(num_frames)]

    count = 0
    saved = 0
    success = True

    while success and saved < num_frames:
        success, frame = cap.read()
        if not success:
            break
        if count in frame_indices:
            frame_path = os.path.join(save_dir, f"frame_{count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
        count += 1

    cap.release()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for dataset_name, relative_path in DATASETS.items():
        # Skip zip datasets
        if relative_path.endswith('.zip'):
            continue

        print(f"\n🔍 Processing dataset: {dataset_name}")

        # Construct video directory path (usually ends with /c23/videos)
        video_dir = os.path.join(BASE_INPUT_PATH, relative_path, "c23", "videos")
        if not os.path.exists(video_dir):
            print(f"⚠️ Skipping {dataset_name}: No such directory {video_dir}")
            continue

        # Output path
        output_path = os.path.join(OUTPUT_DIR, dataset_name)
        os.makedirs(output_path, exist_ok=True)

        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        for video_file in tqdm(video_files, desc=f"[{dataset_name}]"):
            video_path = os.path.join(video_dir, video_file)
            video_name = os.path.splitext(video_file)[0]
            video_output_dir = os.path.join(output_path, video_name)
            extract_frames_from_video(video_path, video_output_dir, FRAMES_PER_VIDEO)


if __name__ == "__main__":
    main()