import os
import json
from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/maximc/afeka/deep-learning/credentials/confident-trail-426114-e6-4eccbb1499cb.json"

with open(os.path.join(".", "config", "project.json")) as json_file:
  project_config = json.load(json_file)
BUCKETS = project_config["buckets"]
GIT = project_config["git"]
MODEL_INPUT = project_config["model_input"]["original_images_format"]
STD_OUTPUT_THRESHOLD = project_config["machine"]["std_output_threshold"]
SOURCE_FOLDER = '/Users/maximc/afeka/deep-learning/test2017'

def upload_images_to_gcs(bucket_name, source_folder):
  client = storage.Client()
  bucket = client.get_bucket(bucket_name)
  uploaded_count = 0
  std_threshold = STD_OUTPUT_THRESHOLD
  for root, _, files in os.walk(source_folder):
    for file_name in files:
      if file_name.lower().endswith(tuple(MODEL_INPUT)):
        local_file_path = os.path.join(root, file_name)
        blob = bucket.blob(file_name)
        blob.upload_from_filename(local_file_path)
        os.remove(local_file_path)
        uploaded_count += 1
        if uploaded_count > std_threshold:
          print(f"Image #{uploaded_count} was uploaded")
          std_threshold += STD_OUTPUT_THRESHOLD

if __name__ == "__main__":
  upload_images_to_gcs(BUCKETS["unpreprocessed"]["name"], os.path.join(".", GIT["local_images_dir"]))
