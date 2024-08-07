import json
from io import BytesIO
import os
from google.cloud import storage
from PIL import Image

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/maximc/afeka/deep-learning/credentials/confident-trail-426114-e6-4eccbb1499cb.json"

with open(os.path.join(".", "config", "project.json")) as json_file:
  project_config = json.load(json_file)
BUCKETS = project_config["buckets"]
GIT = project_config["git"]
STD_OUTPUT_THRESHOLD = project_config["machine"]["std_output_threshold"]
TARGET_IMAGE_NUMBER = 1

def convert_bytes_to_image(data_bytes: bytes) -> Image:
  return Image.open(BytesIO(data_bytes))

def download_images_from_gcs(bucket_name, bucket_directory, target_folder):
  client = storage.Client()
  bucket = client.get_bucket(bucket_name)
  blobs = bucket.list_blobs(prefix=bucket_directory)
  count = 0
  for blob in blobs:
    image = convert_bytes_to_image(blob.download_as_bytes())
    image.save(os.path.join(target_folder, f"img_{count}.png"))
    count += 1
    if count % STD_OUTPUT_THRESHOLD == 0:
      print(f"Image #{count} was saved")
    if count == TARGET_IMAGE_NUMBER:
      print(f"{count} images were downloaded. Stopping")
      break

if __name__ == "__main__":
  download_images_from_gcs(BUCKETS["preprocessed"]["name"], BUCKETS["preprocessed"]["classic"]["test"]["name"], os.path.join(".", GIT["local_images_dir"]))
