# Cosmoformer Model

## Introduction

This repository "Cosmoformer Model" contains pipeline for training and saving `CosmoFormer` model, that was based on `CrossFormer` architecture. Both model and pipeline are the parts of my Bachelor thesis at Brno Technical University (BUT).

**Thesis title**: AI-Powered Web Application for Galaxy Morphology Classification on Red Hat OpenShift  
**Acad. year**: 2024/2025  
**Department**: Department of Intelligent Systems  
**Type of thesis**: Bachelor's Thesis  
**Language of thesis**: en  
**Thesis focus**: Artificial Intelligence  
**Supervisor**: Mgr. Kamil Malinka, Ph.D.  
**Reviewer**: Ing. Milan Šalko  
**Consultant**: Forde Kieran, Ph.D.  

**Electronic source citation (english):**

    SULTANOV, Artur. AI-Powered Web Application for Galaxy Morphology Classification on Red Hat OpenShift. Online, bachelor's Thesis. Kamil MALINKA (supervisor). Brno: Brno University of Technology, Faculty of Information Technology, 2025. Available at: https://www.vut.cz/en/students/final-thesis/detail/164309. [accessed 2025-04-15].

---

## Installation and Prerequisites

1. Install `Python 3.12` version.
2. Install `Python` packages from <a href="requirements.txt">requirements.txt</a> file.
3. Dataset should be placed at **cosmoformer-dataset** folder. Dataset can be pulled from <a href="https://github.com/ArturSultanov/cosmoformer-dataset"> ArturSultanov/cosmoformer-dataset</a> repository.
4. Open <a href="model_training.ipynb"> `model_training.ipynb` </a> file and follow steps inside the Jupyter notebook.

## Model overview

`CrossFormer` architecure has been used for model creation. The model code has been obtained via `vit-pytorch` Python package.

```python
model = CrossFormer(
    num_classes=len(le.classes_),
    dim=(32, 64, 128, 256),
    depth=(2, 2, 4, 2),
    global_window_size=(8, 4, 2, 1),
    local_window_size=7,
    attn_dropout=0.1,
    ff_dropout=0.1
).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
```

For training the `PyTorch` framework was used. This model was optimize to consume as less resources as possible while delivering dissent inference CPU performance.

Model was published on Hugging Face: <a href="https://huggingface.co/artursultanov/cosmoformer-model"> artursultanov/cosmoformer-model</a>. 

## Links:
1. Galaxy classification application: https://github.com/ArturSultanov/cosmoformer-application
2. CosmoFormer model: https://github.com/ArturSultanov/cosmoformer-model
3. Trained model on Hugging Face: https://huggingface.co/artursultanov/cosmoformer-model
4. CosmoFormer dataset: https://github.com/ArturSultanov/cosmoformer-dataset