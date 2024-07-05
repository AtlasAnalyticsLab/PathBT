# Enhanced Barlow Twins for Colorectal Polyp Screening
🚀 We propose an optimized Barlow Twins for colorectal polyp screening. The augmentation strategy was carefully studied to fit our dataset, a dataset from Kingston General Hospital made of 1037 WSIs from pathology (Hyperplatis Polyps, Sessile Serrated Lesions, Tubular Adenoma, Tubulovillous Adenoma) and histology colon. Using ResNet50 or Swin-Tiny, our porposed models perfom better than the basic Barlow Twins on the patch level and better than the basic Barlow Twins with limited data on the slide level.
<div align="center">
  <img width="35%" alt="Watercolor-colon" src="assets/MHIST-dallE.jpeg">
</div>

## 🔍 What is PathBT ?
<div align="center">
  <img width="80%" alt="General framework" src="assets/GitHub.png">
</div>

### Augmentation strategy
🎨 We adapt Barlow Twins augmentation strategy, including GrayScale, GaussianBlur, Solarization, ColorJitter, to the very specific pathology data. After an ablation study, we conclude that the best transformation strategy include a weak color jitter, a solarization with a high threshold, high posterization. The augmentation strategy is presented in `class TransformPath` in [factory.py](pathBT/factory.py)
### Encoder
As Swin Transformer maps the hierarchy of the pathology data, it acts as a local-global feature extractor and has shown remarkable results on diverse pathology dataset ([1](https://www.sciencedirect.com/science/article/pii/S1361841522002043),[2](https://pubmed.ncbi.nlm.nih.gov/36318158/)). We propose to compare [ResNet-50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) with [swin_tiny](https://pytorch.org/vision/main/models/generated/torchvision.models.swin_t.html?highlight=swin+t#torchvision.models.swin_t) within Barlow Twins frameworks.
### Datasets
The dataset used for the development of PathBT is a private dataset of 1037 WSIs from Kingston General Hospital, presenting with 5 classes of tissues: normal, hyperplastic polyps, sessile serrated lesions, tubular adenoma and tubulovillous adenoma. This dataset is annotated on the ROI level. We evaluated our frameworks on 4 different datasets from this private dataset, built on 4 different Fields of view. Barlow Twins was trained on all patches from the slides, and the linear evaluation is performed on the patches from the ROI. PathBT (with resnet50 or swin-t) was also evaluated on [PCam](https://github.com/basveeling/pcam) dataset. Downstream tasking was performed on [CRC](https://paperswithcode.com/dataset/crc) and [MHIST](https://bmirds.github.io/MHIST/) datasets.
## 🎯 Results
We evaluate the modes on the patch and slide level:
- on the patch level, we train a linear layer on top of the frozen backbone;
- on the slide level, we train [CLAM](https://github.com/mahmoodlab/CLAM) framework on the frozen backbone.
For downstream tasking, we train a linear layer on top of the frozen backbone (pretrained on KGH datasets)
Results are also compared with the pretrained weights of this following [work)(https://lunit-io.github.io/research/publications/pathology_ssl/) where a ResNet-50 encoder was trained in Barlow Twins setting on a very large cohort of 36K WSIs.
### Patch-level classification on KGH datasets
### Slide-level classification on KGH datasets
### Patch-level classification on PCam dataset
### Downstream tasking on CRC & MHIST

## 🖌️ Explainability of the results
### Confusion matrices
### GradCAM & SHAP
### CLAM

## 🦾 To train your own pathBT
### Environment
To install the environment used for these experiments, you can use `virtualenv`:
- first create the environment `virtualenv barlow-env`;
- activate it `source barlow-env/bin/activate`;
- install all necessary packages `pip install -r requirements-barlow.txt`;

NOTA BENE:
- for CLAM experiments, a conda environement is provided (see [config file](Clam/env.yml));
- for GradCAM, the authors provide their own environement as well or you can install it with `pip install grad-cam` (documentation [here](https://github.com/jacobgil/pytorch-grad-cam));
- for SHAP, the authors provide a library which you can install with `pip install shap` (documentation [here](https://pypi.org/project/shap/)).
### Datasets
### Scripts
To train a ResNet-50 encoder with the proposed augmentation strategy, you can run
```python
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc-per-node=1 /pathBT/main.py \
    --backbone resnet50 --dataset pkgh \
    --pretrained imagenet"--transform patho \
    --epochs 100  --batch-size 512 \
    --projector "8192-8192-8192"  \
    --checkpoint-dir your/model/directory \
```
To train a Swin Tiny encoder with the proposed augmentation strategy, you can run
```python
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc-per-node=1 /pathBT/main.py \
    --backbone swin_t --dataset pkgh \
    --pretrained imagenet"--transform patho \
    --epochs 100  --batch-size 512 \
    --projector "8192-8192-8192"  \
    --checkpoint-dir your/model/directory \
```
To train a ResNet encoder in the basic Barlow Twins settings, you can run
```python
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc-per-node=1 /pathBT/main.py \
    --backbone resnet50 --dataset pkgh \
    --epochs 100  --batch-size 512 \
    --projector "8192-8192-8192"  \
    --checkpoint-dir your/model/directory \
```
If you need Weights&Biases support, you can set `--wandb True` and precise your login key with `--wandb_login your-key`
## 🔽 Download the models
All models trained KGH dataset (trained with Barlow Twins or in a supervised manner) can be found in this [folder](https://drive.google.com/drive/folders/1Ut-Tsly1kSpRl6Jh_MgyVocQZG59dWcb?usp=sharing)
