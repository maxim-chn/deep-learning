# Architecture overview
![image-architecture](assets/architecture_overview.png)

# Artifacts overview
Directory `assets` is expected to include any resource that is non-code related. For example, images that assist in project description.

Directory `config` is expected to include files with configurations for runtime environment, various source and target destinations,
machine specifications and more.

Directory `credentials` is expected to include files that grant access to resources such as `Google Cloud Services`.

Directory `image_abyss` is expected to include images that are collected from various sources. Its basically a large "basket" of items
that our `Preprocessor` reads and transforms into viable model input.  
Directory can also be used to download model tailored images and point at it during training.

Directory `src` contains mostly code and its related items. You can find there implementations such as image uploader to `GCS buckets`,
image downloader from `GCS buckets`, `Preprocessor` and more.

# Project Requirements
Project was developed in `Python3.X` environment.

We suggest to initialize `Python venv` prior to installing required modules and `PIP` for module management.  
The following modules are required for scripts, such as images uploader/downloader to and from `GCS buckets`:
* `google-cloud-storage`
* `pillow`
