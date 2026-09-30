# CoVer and CoVer+

Official code for **Unveiling Representation Expansion: A New Lens on Out-of-Distribution Detection** ([IJCV 2026](https://doi.org/10.1007/s11263-026-03036-2)) and the preceding [NeurIPS 2024 CoVer paper](https://arxiv.org/abs/2410.18472).

CoVer averages an OOD confidence score across an original image and its corrupted views. CoVer+ also averages across pretrained model backbones. The CLIP implementation here uses the maximum softmax probability (MCM) with temperature 1 by default. All image views and model backbones have equal weight; no model-dependent standardization is applied.

## Setup

Use Python 3.10, PyTorch and a matching torchvision build with CUDA support. Install the remaining packages with:

~~~bash
pip install -r requirements.txt
~~~

The repository includes the OpenCLIP implementation used by the original CoVer code. It downloads the OpenAI CLIP weights on first use. The selected corruptions use ImageMagick through wand; install ImageMagick on your system before generating or applying corruptions.

## Data

Dataset images and generated corruptions are not distributed with this repository. Download them separately into the ignored **datasets/** directory.

### Download original images

- **ImageNet-1K:** obtain the ILSVRC 2012 validation set from the [official ImageNet download page](https://image-net.org/challenges/LSVRC/2012/index.php#); registration may be required. Organize the validation images by synset under **datasets/ImageNet/val/**.
- **ImageNet-10/20/100 (optional):** follow the subset construction instructions in [MCM](https://github.com/deeplearning-wisc/MCM). These are needed only for the corresponding subset evaluations.
- **iNaturalist, SUN and Places:** use the ImageNet OOD subsets curated by [MOS](https://github.com/deeplearning-wisc/large_scale_ood#out-of-distribution-dataset): [iNaturalist.tar.gz](https://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz), [SUN.tar.gz](https://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz), and [Places.tar.gz](https://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz).
- **Textures (DTD):** download **dtd-r1.0.1.tar.gz** from the [official DTD page](https://www.robots.ox.ac.uk/~vgg/data/dtd/).

Extract the four OOD archives into **datasets/ImageNet_OOD_dataset/**. For example, if the archives are in your Downloads directory:

~~~bash
mkdir -p datasets/ImageNet_OOD_dataset
tar -xzf ~/Downloads/iNaturalist.tar.gz -C datasets/ImageNet_OOD_dataset
tar -xzf ~/Downloads/SUN.tar.gz -C datasets/ImageNet_OOD_dataset
tar -xzf ~/Downloads/Places.tar.gz -C datasets/ImageNet_OOD_dataset
tar -xzf ~/Downloads/dtd-r1.0.1.tar.gz -C datasets/ImageNet_OOD_dataset
~~~

The extracted paths expected by the ImageNet-1K CLIP evaluation are:

~~~text
datasets/
├── ImageNet/val/<synset>/<image>
└── ImageNet_OOD_dataset/
    ├── iNaturalist/images/<image>
    ├── SUN/images/<image>
    ├── Places/images/<image>
    └── dtd/images/<texture-class>/<image>
~~~

The small ImageNet class-name list used in zero-shot prompts is already included at **data/ImageNet/imagenet_class_clean.txt**.

### Generate corrupted views

The default CLIP view set follows Table 13 of the IJCV paper: the original image plus brightness, fog, saturate, motion blur, defocus blur and Gaussian blur, each at severity 1 and 2. Generate the required views from the downloaded original images:

~~~bash
python utils/imagenet_c/gen_corruptions.py \
  --dataset all --workers 8 \
  --corruptions brightness,fog,saturate,motion_blur,defocus_blur,gaussian_blur \
  --severities 1,2
~~~

This writes ImageNet views under **datasets/DistortedImageNet/val_224/** and OOD views under **datasets/ImageNet_OOD_dataset/Distorted_CLIP/**. These generated files are also ignored by Git.

Precomputation can require substantial disk space. For ImageNet-1K and the four OOD sets, **ON_THE_FLY_CORRUPT=1** applies corruptions while reading the original images without writing the corrupted directories. Use precomputed images for reproducible scores and identical corrupted views across backbones; the on-the-fly path resamples stochastic corruptions on each pass.

## Evaluation

Run from the repository root. CoVer+ uses CLIP ViT-B/16 and ViT-L/14 by default:

~~~bash
python eval_ood_detection.py --model CLIP --score 'CoVer+' \
  --in_dataset ImageNet --root-dir datasets --gpu 0 --batch-size 128 \
  --CLIP_ckpts ViT-B-16 ViT-L-14 --name ijcv_cover_plus
~~~

To use the on-the-fly data path, prefix the command with **ON_THE_FLY_CORRUPT=1**. To evaluate CoVer with one backbone:

~~~bash
python eval_ood_detection.py --model CLIP --score CoVer \
  --in_dataset ImageNet --CLIP_ckpt ViT-B-16 --gpu 0
~~~

**--CLIP_ckpts** accepts two or more distinct backbones for CoVer+. Both modes use the same view set defined by **imagenet_c** in **eval_ood_detection.py**. The script writes per-dataset scores, OOD metrics (FPR95, AUROC, AUPR), and distribution plots under **results/<ID>/<score>/.../**. Scores saved in the text files are negative confidence; the metrics invert their sign so larger confidence means more likely ID.

The original ResNet and ImageNet subset paths remain available, but need their respective pretrained checkpoints, DICE feature statistics, and subset data layout. The command above is the IJCV CoVer+ CLIP entry point.

## License

The CoVer code is released under the MIT license; see [LICENSE](LICENSE). Dataset and pretrained model licenses are separate.

## Citation

~~~bibtex
@article{zhang2026unveiling,
  title={Unveiling Representation Expansion: A New Lens on Out-of-Distribution Detection},
  author={Zhang, Boxuan and Zhu, Jianing and Wang, Zengmao and Liu, Tongliang and Du, Bo and Han, Bo},
  journal={International Journal of Computer Vision},
  volume={134},
  year={2026},
  doi={10.1007/s11263-026-03036-2}
}

@inproceedings{zhang2024what,
  title={What If the Input is Expanded in OOD Detection?},
  author={Zhang, Boxuan and Zhu, Jianing and Wang, Zengmao and Liu, Tongliang and Du, Bo and Han, Bo},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
~~~
