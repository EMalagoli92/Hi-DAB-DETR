# Hi-DAB-DETR: DETR with Hierarchical Prototype Refinement

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-ee4c2c.svg)](https://pytorch.org/)
[![AVSS 2026](https://img.shields.io/badge/AVSS-2026-lightgrey.svg)](https://avss2026.org/)

Official PyTorch implementation of **Hi-DAB-DETR**, accepted at **AVSS 2026** (Advanced Video and Signal-based Surveillance).

> **Hi-DAB-DETR: DETR with Hierarchical Prototype Refinement**
> Emanuele Malagoli, Nicola Conci, Luca Di Persio
> University of Trento, University of Verona, HPA (Terranova S.r.l.)
>
> *Paper / proceedings link will be added once available.*

## Abstract

We introduce Hi-DAB-DETR (Hierarchical DAB-DETR), an extension of DAB-DETR with a prototype-based refinement mechanism. The proposed approach injects a predefined class hierarchy into both the classification and localization sub-tasks of DETR-style object detectors. The hierarchical prototypes are obtained by fusing learnable prototypes with memory-based ones, represented as hyperbolic embeddings and updated online via a memory mechanism. They are used to initialize content queries and guide the iterative refinement of both content queries via attention and anchor boxes via prototype-based offsets. We also introduce a supervised hierarchical contrastive loss in hyperbolic space, with a dedicated sampling strategy, to align query embeddings with the class hierarchy.

## Architecture

<p align="center">
  <img src="assets/architecture.png" alt="Hi-DAB-DETR architecture" width="100%">
</p>

## Results

### Main Results (MS-COCO `val2017`)

Comparison between DAB-DETR and Hi-DAB-DETR across backbones (`*` denotes the variant with 3 pattern embeddings, as in Anchor DETR):

| Model                         | #epochs | AP   | AP50 | AP75 | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> | GFLOPs | Params |
|-------------------------------|:-------:|:----:|:----:|:----:|:----:|:----:|:----:|:------:|:------:|
| DAB-DETR-R50                  |   50    | 42.2 | 63.1 | 44.7 | 21.5 | 45.7 | 60.3 |   94   |  44M   |
| **Hi-DAB-DETR-R50**            |   50    | **42.7** | 63.6 | 45.4 | 23.5 | 46.3 | 61.8 |   98   |  47M   |
| DAB-DETR-R50*                 |   50    | 42.6 | 63.2 | 45.6 | 21.8 | 46.2 | 61.1 |  100   |  44M   |
| **Hi-DAB-DETR-R50***           |   50    | **43.1** | 63.9 | 45.9 | 23.0 | 46.6 | 62.6 |  110   |  47M   |
| DAB-DETR-DC5-R50              |   50    | 44.5 | 65.1 | 47.7 | 25.3 | 48.2 | 62.3 |  202   |  44M   |
| **Hi-DAB-DETR-DC5-R50**        |   50    | **45.1** | 65.7 | 48.1 | 25.7 | 48.7 | 63.0 |  206   |  47M   |
| DAB-DETR-DC5-R50*             |   50    | 45.7 | 66.2 | 49.0 | 26.1 | 49.4 | 63.1 |  216   |  44M   |
| **Hi-DAB-DETR-DC5-R50***       |   50    | **46.0** | 66.4 | 49.3 | 26.9 | 49.5 | 63.8 |  223   |  47M   |
| DAB-DETR-R101                 |   50    | 43.5 | 63.9 | 46.6 | 23.6 | 47.3 | 61.5 |  174   |  63M   |
| **Hi-DAB-DETR-R101**           |   50    | **44.0** | 65.0 | 47.3 | 24.3 | 48.4 | 62.7 |  178   |  66M   |
| DAB-DETR-R101*                |   50    | 44.1 | 64.7 | 47.2 | 24.1 | 48.2 | 62.9 |  179   |  63M   |
| **Hi-DAB-DETR-R101***          |   50    | **44.5** | 65.8 | 47.9 | 24.8 | 48.2 | 63.2 |  186   |  66M   |
| DAB-DETR-DC5-R101             |   50    | 45.8 | 65.9 | 49.3 | 27.0 | 49.8 | 63.8 |  282   |  63M   |
| **Hi-DAB-DETR-DC5-R101**       |   50    | **46.3** | 66.8 | 49.7 | 27.4 | 50.2 | 64.3 |  286   |  66M   |
| DAB-DETR-DC5-R101*            |   50    | 46.6 | 67.0 | 50.2 | 28.1 | 50.5 | 64.1 |  296   |  63M   |
| **Hi-DAB-DETR-DC5-R101***      |   50    | **46.9** | 67.5 | 50.7 | 28.3 | 50.6 | 65.1 |  303   |  66M   |

See the paper for the ablation study and results with fixed reference points `(x, y)`.

## Installation

```bash
git clone https://github.com/EMalagoli92/Hi-DAB-DETR.git
cd Hi-DAB-DETR
pip install -r requirements.txt
```

## Dataset Preparation

Hi-DAB-DETR is trained and evaluated on MS-COCO 2017. Download and extract the dataset with:

```bash
bash data/coco2017.sh
```

This downloads `train2017`, `val2017` and their annotations into `data/coco2017`, following the standard COCO directory layout expected by `datasets/coco.py`.

The class hierarchy used to build hierarchical prototypes on MS-COCO is derived from WordTree (introduced in YOLO9000, built from WordNet); the relevant files are available under [`datasets/coco_wordtree`](datasets/coco_wordtree).

## Training

Training is launched via [main.py](main.py), configured through a JSON/YAML config file (see [`configs/`](configs) for the configurations used in the paper, e.g. `R50.json`, `R50_pat3.json`, `R50DC5.json`, `R101.json`, ...):

```bash
python main.py --config configs/R50.json --output_dir output/R50
```

For multi-GPU / multi-node training, use `torch.distributed.run`:

```bash
python -m torch.distributed.run \
  --nnodes <nnodes> \
  --nproc_per_node <nproc_per_node> \
  --node_rank 0 \
  main.py \
  --config <config_path> \
  --output_dir <output_dir>
```

Configuration naming convention:
- `_pat3`: variant with 3 pattern embeddings (Anchor DETR style).
- `_fixxy`: variant with fixed reference points `(x, y)`.
- `DC5`: dilated backbone (DC5), following DAB-DETR.

## Evaluation

To evaluate a trained checkpoint, set `"eval": true` and `"resume": "<path/to/checkpoint.pth>"` in the config file, then run:

```bash
python main.py --config configs/R50.json --output_dir output/R50_eval
```

## Acknowledgement

This code is built upon [DAB-DETR](https://github.com/IDEA-Research/DAB-DETR) and [DETR](https://github.com/facebookresearch/detr). We thank the authors for their excellent work.

## Citation

The paper is currently in press for AVSS 2026; a BibTeX entry will be added here once the proceedings are published.

## License

This project is released under the [Apache License 2.0](LICENSE).