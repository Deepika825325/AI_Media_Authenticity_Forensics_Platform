import cv2
from pathlib import Path

# =========================================
# ROOT PROJECT DIRECTORY
# =========================================

ROOT_DIR = Path(__file__).resolve().parents[3]

# =========================================
# INPUT / OUTPUT PATHS
# =========================================

INPUT_DIR = ROOT_DIR / "datasets/processed/faces"
OUTPUT_DIR = ROOT_DIR / "datasets/processed/aligned_faces"

IMAGE_SIZE = 224

# =========================================
# ALIGNMENT FUNCTION
# =========================================

def align_faces(label):

    input_root = INPUT_DIR / label

    output_root = OUTPUT_DIR / label

    if not input_root.exists():

        print(f"[ERROR] Input path not found: {input_root}")

        return

    for video_folder in input_root.iterdir():

        if not video_folder.is_dir():
            continue

        print(f"\nAligning faces from: {video_folder.name}")

        save_folder = output_root / video_folder.name

        save_folder.mkdir(parents=True, exist_ok=True)

        for face_path in video_folder.glob("*.jpg"):

            image = cv2.imread(str(face_path))

            if image is None:
                continue

            # Resize to standard model size

            aligned = cv2.resize(
                image,
                (IMAGE_SIZE, IMAGE_SIZE)
            )

            save_path = save_folder / face_path.name

            cv2.imwrite(str(save_path), aligned)

        print(f"Saved aligned faces to: {save_folder}")

# =========================================
# RUN
# =========================================

if __name__ == "__main__":

    align_faces("real")
    align_faces("fake")

    print("\nFace alignment completed.")