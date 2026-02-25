import os
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# ------------ Data Preparation Functions ------------
def clear_folder(folder):
    if os.path.exists(folder):
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(folder)

def move_frames(source_dirs, dest_dir, label):
    for src_dir in source_dirs:
        if not os.path.exists(src_dir):
            print(f"⚠️ Skipping missing folder: {src_dir}")
            continue
        for video_folder in os.listdir(src_dir):
            video_path = os.path.join(src_dir, video_folder)
            if not os.path.isdir(video_path):
                continue
            for frame_file in os.listdir(video_path):
                if frame_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src_frame = os.path.join(video_path, frame_file)
                    new_name = f"{label}_{video_folder}_{frame_file}"
                    dest_path = os.path.join(dest_dir, new_name)
                    shutil.copy2(src_frame, dest_path)

def prepare_data():
    real_sources = [
        'frames/original',
        'frames/DeepFakeDetection_original'
    ]
    fake_sources = [
        'frames/Deepfakes',
        'frames/DeepFakeDetection',
        'frames/Face2Face',
        'frames/FaceShifter',
        'frames/FaceSwap',
        'frames/NeuralTextures'
    ]

    dest_real = 'processed_data/real'
    dest_fake = 'processed_data/fake'

    # Clear destination folders to avoid duplicates
    clear_folder(dest_real)
    clear_folder(dest_fake)

    print("Processing real video frames...")
    move_frames(real_sources, dest_real, "real")

    print("Processing fake video frames...")
    move_frames(fake_sources, dest_fake, "fake")

    print("✅ All frames copied to 'processed_data/real' and 'processed_data/fake'")

# ------------ Data Loader Function ------------
def get_dataloaders(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# ------------ Run Preparation Only If Called Directly ------------
if __name__ == "__main__":
    prepare_data()