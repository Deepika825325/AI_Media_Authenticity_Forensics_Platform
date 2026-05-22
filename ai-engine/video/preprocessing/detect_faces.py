import cv2
import mediapipe as mp
from pathlib import Path

# =========================================
# ROOT PROJECT DIRECTORY
# =========================================

ROOT_DIR = Path(__file__).resolve().parents[3]

# =========================================
# INPUT / OUTPUT PATHS
# =========================================

INPUT_DIR = ROOT_DIR / "datasets/processed/frames"
OUTPUT_DIR = ROOT_DIR / "datasets/processed/faces"

# =========================================
# MEDIAPIPE SETUP
# =========================================

mp_face = mp.solutions.face_detection

detector = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

# =========================================
# FACE DETECTION FUNCTION
# =========================================

def detect_faces(label):

    frame_root = INPUT_DIR / label

    output_root = OUTPUT_DIR / label

    if not frame_root.exists():

        print(f"[ERROR] Path not found: {frame_root}")

        return

    for video_folder in frame_root.iterdir():

        if not video_folder.is_dir():
            continue

        print(f"\nProcessing video folder: {video_folder.name}")

        save_folder = output_root / video_folder.name

        save_folder.mkdir(parents=True, exist_ok=True)

        for frame_path in video_folder.glob("*.jpg"):

            image = cv2.imread(str(frame_path))

            if image is None:
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = detector.process(rgb)

            if results.detections:

                for idx, detection in enumerate(results.detections):

                    bbox = detection.location_data.relative_bounding_box

                    h, w, _ = image.shape

                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)

                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)

                    # Prevent negative coordinates

                    x = max(0, x)
                    y = max(0, y)

                    face = image[y:y+bh, x:x+bw]

                    if face.size == 0:
                        continue

                    save_path = (
                        save_folder /
                        f"{frame_path.stem}_{idx}.jpg"
                    )

                    cv2.imwrite(str(save_path), face)

        print(f"Saved faces to: {save_folder}")

# =========================================
# RUN PIPELINE
# =========================================

if __name__ == "__main__":

    detect_faces("real")
    detect_faces("fake")

    print("\nFace detection completed.")