# Pill Image Classification with MobileNetV3

A computer-vision prototype that classifies photographs of medication pills into **84 known product classes**. The project explores transfer learning with MobileNetV3-small to support visual quality-control workflows in which a medication image must be identified before manual review.

https://stock.adobe.com/images/cute-tablets-pills-and-capsules-cartoon-characters/636554169

> **Scope:** This is an educational image-classification experiment. It is **not** a medical device and must not be used for diagnosis, prescribing, dispensing, or autonomous clinical decisions.

## Result

The final two-stage fine-tuning pipeline achieved **76.79% accuracy** on an untouched external test set, exceeding the project target of 75%.

| Metric | Result |
|---|---:|
| Target classes | 84 |
| Source training images | 2,352 |
| Internal validation split | 20%, stratified |
| External test images | 504 |
| Best validation accuracy | 77.71% |
| Final external test accuracy | **76.79%** |
| Selected checkpoint | Fine-tuning phase, epoch 20 |
| Training environment | Google Colab, NVIDIA A100-SXM4-40GB |

## Methodology

The model is `MobileNetV3-small` initialized with `IMAGENET1K_V1` pretrained weights. The notebook follows the preprocessing associated with these weights: resize to 256 pixels, centre crop to 224×224, conversion to a tensor, and ImageNet RGB normalization. [1]

The data pipeline uses `ImageFolder`, where each medication class is represented by a separate directory of images. [2] The original `train/` folder is split stratified into training and validation subsets, while the provided `test/` folder is reserved for the final evaluation only.

Training consists of two stages. First, the feature extractor is frozen and the classifier head is trained for eight epochs. Second, the last three feature blocks and the classifier are fine-tuned with lower learning rates. The procedure uses `AdamW`, `ReduceLROnPlateau`, checkpointing based on validation loss, and early stopping. This is consistent with the standard distinction between a fixed feature extractor and partial fine-tuning in transfer learning. [3]

The notebook also includes a class-level precision/recall/F1 report and a full confusion matrix for analysis of final-test errors.

## Dataset

This project uses an 84-class subset of the publicly available [OGYEIv2 dataset on Kaggle](https://www.kaggle.com/datasets/richardradli/ogyeiv2), an image collection for pill recognition created by Richárd Rádli and collaborators. The full OGYEIv2 collection contains 112 pill classes and 4,480 images. [4]

The experiment uses 2,352 images in the provided training folder and 504 images in the external test folder. The notebook checks that class-to-index mappings are identical in the `train/` and `test/` directories before training.

> **Data notice:** Dataset files are not copied into this repository. Download OGYEIv2 directly from Kaggle and comply with the licence and terms specified in its data card. The dataset is listed there under the GNU Affero General Public License v3.0 at the time of access. [4]

The notebook expects an archive containing this structure:

```text
<dataset-root>/
├── train/
│   ├── class_name_1/
│   ├── class_name_2/
│   └── ...
└── test/
    ├── class_name_1/
    ├── class_name_2/
    └── ...
```

## Run in Google Colab

1. Download or clone this repository and upload `notebooks/pill_image_classification.ipynb` to [Google Colab](https://colab.research.google.com/).
2. In Colab, select **Runtime → Change runtime type → GPU**.
3. Download the [OGYEIv2 dataset from Kaggle](https://www.kaggle.com/datasets/richardradli/ogyeiv2) and place the authorized ZIP archive in Google Drive.
4. In the data-loading cell, set `ZIP_PATH` to that archive, for example:
   ```python
   ZIP_PATH = Path('/content/drive/MyDrive/path/to/tablets.zip')
   ```
5. Run the notebook from top to bottom. The MobileNetV3 pretrained weights are downloaded automatically by `torchvision` on the first run.

## Limitations

The external test set contains only six images per class, so its accuracy estimate has substantial uncertainty at the per-class level. Further work would require a larger independent test set, image acquisition metadata, robustness testing across lighting and background conditions, and expert validation before considering any operational use.
