import os
import sys
import numpy as np
import pandas as pd
from sentinelhub import (
    SHConfig,
    SentinelHubRequest,
    DataCollection,
    MimeType,
    CRS,
    BBox,
    bbox_to_dimensions
)
from PIL import Image

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "train(1).csv")
IMG_DIR = os.path.join(BASE_DIR, "data", "images", "train")
os.makedirs(IMG_DIR, exist_ok=True)

# ---------------- SENTINEL CONFIG ----------------
config = SHConfig()
config.sh_client_id = "6a6030c2-d7fd-4e48-8c46-f0e36eead8f9"
config.sh_client_secret = "kNBRxkkFgc08WQVjfWLzkNAcl3NZ1mdL"
config.save()

# ---------------- IMAGE SETTINGS ----------------
BUFFER_DEG = 0.015
TARGET_RESOLUTION_M = 5
TIME_RANGE = ("2020-05-01", "2020-09-30")

# ---------------- EVALSCRIPT (FIXED TRUE COLOR - CLEAR GREENERY) ----------------
evalscript = """
//VERSION=3 (function version)
// Official Sentinel Hub true color - shows clear green vegetation [web:16]
function setup() {
  return {
    input: [{
      bands: ["B02", "B03", "B04"],
      units: "REFLECTANCE"
    }],
    output: { bands: 3, sampleType: "UINT8" }
  };
}

function evaluatePixel(sample) {
  // Simple multiplier - B04(red), B03(green), B02(blue) = true color
  return [
    Math.min(255, 2.5 * sample.B04 * 255),
    Math.min(255, 2.5 * sample.B03 * 255),
    Math.min(255, 2.5 * sample.B02 * 255)
  ];
}
"""

# ... rest of your functions stay EXACTLY the same ...
def build_bbox(lat: float, lon: float) -> BBox:
    return BBox(
        bbox=[lon - BUFFER_DEG, lat - BUFFER_DEG, lon + BUFFER_DEG, lat + BUFFER_DEG],
        crs=CRS.WGS84
    )

def fetch_and_save_image(row, config):
    img_id = str(int(row["id"]))
    lat = float(row["lat"])
    lon = float(row["long"])

    bbox = build_bbox(lat, lon)
    size = bbox_to_dimensions(bbox, resolution=TARGET_RESOLUTION_M)

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=TIME_RANGE
            )
        ],
        responses=[
            SentinelHubRequest.output_response("default", MimeType.PNG)
        ],
        bbox=bbox,
        size=size,
        config=config
    )

    data = request.get_data()
    if not data or data[0] is None:
        raise RuntimeError("Empty image data returned")

    img = data[0].astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    filename = f"{img_id}_{lat:.5f}_{lon:.5f}.png"
    out_path = os.path.join(IMG_DIR, filename)
    Image.fromarray(img).save(out_path)

    return out_path

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    if not {"id", "lat", "long"}.issubset(df.columns):
        raise ValueError("CSV must contain 'id', 'lat', and 'long' columns")

    total = len(df)
    print(f"Starting download for {total} properties...")

    failures = []
    for idx, row in df.iterrows():
        try:
            out_path = fetch_and_save_image(row, config)
            print(f"✅ [{idx+1}/{total}] Saved {os.path.basename(out_path)}")
        except Exception as e:
            err_msg = f"❌ [{idx+1}/{total}] Failed for id={row.get('id')}: {e}"
            print(err_msg, file=sys.stderr)
            failures.append({"id": row.get("id"), "error": str(e)})

    if failures:
        fail_df = pd.DataFrame(failures)
        fail_log_path = os.path.join(BASE_DIR, "data", "logs")
        os.makedirs(fail_log_path, exist_ok=True)
        fail_csv = os.path.join(fail_log_path, "image_download_failures.csv")
        fail_df.to_csv(fail_csv, index=False)
        print(f"\nFinished with {len(failures)} failures. Logged to {fail_csv}")
    else:
        print("\nAll images downloaded successfully.")

if __name__ == "__main__":
    main()
