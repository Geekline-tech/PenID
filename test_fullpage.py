import numpy as np
import cv2
from pathlib import Path
from src.inference.gallery import Gallery
from src.utils.config import RAW_DIR

gallery = Gallery()
gallery.load_model()

# Test with original full-page images
test_files = [
    (RAW_DIR / "2820987" / "2820987_1.jpg", "2820987"),
    (RAW_DIR / "2820989" / "2820989_1.jpg", "2820989"),
    (RAW_DIR / "2820990" / "2820990_1.jpg", "2820990"),
    (RAW_DIR / "2820991" / "2820991_1.jpg", "2820991"),
    (RAW_DIR / "2820997" / "2820997_1.jpg", "2820997"),
    (RAW_DIR / "2821011" / "2821011_1.jpg", "2821011"),
    (RAW_DIR / "2821007" / "2821007_1.jpg", "2821007"),
    (RAW_DIR / "2821025" / "2821025_1.jpg", "2821025"),
]

correct = 0
for path, expected in test_files:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    result = gallery.identify_image(img)
    match = "OK" if result["name"] == expected else "FAIL"
    if result["name"] == expected:
        correct += 1
    print(f"[{match}] {expected} -> {result['name']} conf={result['confidence']:.3f} d={result['distance']:.4f}")

print(f"\nAccuracy: {correct}/{len(test_files)}")
