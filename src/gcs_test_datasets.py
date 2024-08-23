from io import BytesIO
import os
import json
from google.cloud import storage
from PIL import Image, ImageFilter

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/maximc/afeka/deep-learning/credentials/confident-trail-426114-e6-4eccbb1499cb.json"

with open(os.path.join(".", "config", "project.json")) as json_file:
  project_config = json.load(json_file)
BUCKETS = project_config["buckets"]
MODEL_INPUT = project_config["model_input"]
MODEL_TEST = project_config["model_test"]
STD_OUTPUT_THRESHOLD = project_config["machine"]["std_output_threshold"]
SOURCE_FOLDER = '/Users/maximc/afeka/deep-learning/image_test'

def convert_image_to_bytes(image: Image) -> bytes:
  result = BytesIO()
  image.save(result, format="PNG")
  result.seek(0)
  return result.getvalue()

def upload_images_to_gcs(bucket_name, bucket_directory, images):
  client = storage.Client()
  bucket = client.get_bucket(bucket_name)
  uploaded_count = 0
  std_threshold = STD_OUTPUT_THRESHOLD
  for image_data in images:
    blob = bucket.blob("%s/%s_%s" % (bucket_directory, image_data["name"], image_data["input"]["type"]))
    blob.upload_from_string(image_data["input"]["data"], content_type="image/%s" % MODEL_INPUT["type"])
    blob = bucket.blob("%s/%s_%s" % (bucket_directory, image_data["name"], image_data["expected_output"]["type"]))
    blob.upload_from_string(image_data["expected_output"]["data"], content_type="image/%s" % MODEL_INPUT["type"])
    uploaded_count += 1
    if uploaded_count > std_threshold:
      print(f"Image #{uploaded_count} was uploaded")
      std_threshold += STD_OUTPUT_THRESHOLD

def pair_original_distorted(source_folder):
  result = []
  count = 0
  for root, _, files in os.walk(source_folder):
    if count >= MODEL_TEST["distorted"]["number"]:
      break
    for file_name in files:
      if count >= MODEL_TEST["distorted"]["number"]:
        break
      if file_name.lower().endswith(tuple(MODEL_INPUT["original_images_format"])):
        image = Image.open(os.path.join(root, file_name))
        width, height = image.size
        if width < MODEL_INPUT["width"] or height < MODEL_INPUT["height"]:
          continue
        original = image.copy().resize((MODEL_INPUT["width"], MODEL_INPUT["height"]))
        distorted = image.copy().filter(ImageFilter.BLUR).resize((MODEL_INPUT["width"], MODEL_INPUT["height"]))
        result.append({
          "expected_output": { "type": "original", "data": convert_image_to_bytes(original) },
          "input": { "type": "distorted", "data": convert_image_to_bytes(distorted) },
          "name": f"test_{count}"
        })
        count += 1
  return result

def pair_original_minimized(source_folder):
  result = []
  count = 0
  for root, _, files in os.walk(source_folder):
    if count >= MODEL_TEST["minimized"]["number"]:
      break
    for file_name in files:
      if count >= MODEL_TEST["minimized"]["number"]:
        break
      if file_name.lower().endswith(tuple(MODEL_INPUT["original_images_format"])):
        image = Image.open(os.path.join(root, file_name))
        width, height = image.size
        if width < MODEL_INPUT["width"] or height < MODEL_INPUT["height"]:
          continue
        original = image.copy().resize((MODEL_INPUT["width"], MODEL_INPUT["height"]))
        minimized = image.copy().resize((MODEL_TEST["minimized"]["width"], MODEL_TEST["minimized"]["height"]))
        result.append({
          "expected_output": { "type": "original", "data": convert_image_to_bytes(original) },
          "name": f"test_{count}",
          "input": { "type": "minimized", "data": convert_image_to_bytes(minimized) }
        })
        count += 1
  return result

def pair_novel_minimized(source_folder):
  result = []
  count = 0
  for root, _, files in os.walk(source_folder):
    if count >= MODEL_TEST["novel"]["number"]:
      break
    for file_name in files:
      if count >= MODEL_TEST["novel"]["number"]:
        break
      if file_name.lower().endswith(tuple(MODEL_INPUT["original_images_format"])):
        image = Image.open(os.path.join(root, file_name))
        width, height = image.size
        if width < MODEL_TEST["novel"]["width"] or height < MODEL_TEST["novel"]["height"]:
          continue
        novel = image.copy().resize((MODEL_TEST["novel"]["width"], MODEL_TEST["novel"]["height"]))
        minimized = image.copy().resize((MODEL_TEST["minimized"]["width"], MODEL_TEST["minimized"]["height"]))
        result.append({
          "expected_output": { "type": "novel", "data": convert_image_to_bytes(novel) },
          "name": f"test_{count}",
          "input": { "type": "minimized", "data": convert_image_to_bytes(minimized) }
        })
        count += 1
  return result

if __name__ == "__main__":
  upload_images_to_gcs(BUCKETS["preprocessed"]["name"], BUCKETS["preprocessed"]["test"]["original"]["name"], pair_original_minimized(f"{SOURCE_FOLDER}_original"))
  upload_images_to_gcs(BUCKETS["preprocessed"]["name"], BUCKETS["preprocessed"]["test"]["novel"]["name"], pair_novel_minimized(f"{SOURCE_FOLDER}_novel"))
  upload_images_to_gcs(BUCKETS["preprocessed"]["name"], BUCKETS["preprocessed"]["test"]["distorted"]["name"], pair_original_distorted(f"{SOURCE_FOLDER}_distorted"))
