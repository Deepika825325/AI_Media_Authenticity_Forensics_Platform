import cv2
from pathlib import Path

INPUT_DIR = Path("datasets/samples")
OUTPUT_DIR = Path("datasets/processed/frames")

FRAME_RATE = 5

def extract_frames(video_path, output_folder):

    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_interval = int(fps / FRAME_RATE)

    count = 0
    saved = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if count % frame_interval == 0:

            frame_name = output_folder / f"frame_{saved:04d}.jpg"

            cv2.imwrite(str(frame_name), frame)

            saved += 1

        count += 1

    cap.release()

def process_videos(label):

    video_dir = INPUT_DIR / label

    output_dir = OUTPUT_DIR / label

    for video_file in video_dir.glob("*.mp4"):

        print(f"Processing: {video_file.name}")

        extract_frames(
            video_file,
            output_dir / video_file.stem
        )

process_videos("real")
process_videos("fake")