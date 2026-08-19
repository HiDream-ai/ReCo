# ReCo

<p align="center">
    <img src="assets/title_imgs.png" width="400"/>
<p>

<p align="center">
    🌐 <a href="https://zhw-zhang.github.io/ReCo-page/"><b>Project Page</b></a> &nbsp&nbsp  | &nbsp&nbsp🤗 <a href="https://huggingface.co/datasets/HiDream-ai/ReCo-Data">ReCo-Data</a>&nbsp&nbsp | &nbsp&nbsp 📈 <a href="https://huggingface.co/datasets/HiDream-ai/ReCo-Bench">ReCo-Bench</a>&nbsp&nbsp | &nbsp&nbsp 🤗 <a href="https://huggingface.co/HiDream-ai/ReCo">ReCo-Models  </a> &nbsp&nbsp | &nbsp&nbsp 🌟 <a href="https://zhw-zhang.github.io/ReCo-page/#reco-bench-leaderboard">Leaderboard(New!!)</a> &nbsp&nbsp 
<br>
 

[**ReCo: In-Context Generation with Regional Constraints for Instructional Video Editing**](https://zhw-zhang.github.io/ReCo-page/) <be>

🔆 If you find ReCo useful, please give a ⭐ for this repo, which is important to Open-Source projects. Thanks!


Here, we will gradually release the following resources, including:

* **ReCo training dataset:** ReCo-Data
* **Evaluation code:** ReCo-Bench
* **Model weights, inference code, and training code**

 
## Video Demos

<div align="center">
  <video controls autoplay loop muted playsinline src="https://github.com/user-attachments/assets/ba530e6f-e13b-4d04-ad60-b95277cc38ce"></video>
  <p><em>Examples of different video editing tasks by our ReCo.</em></p>
</div>

## 📢 News!!!

- **🌟 ReCo-Bench Leaderboard**: We have released a [ReCo-Bench Leaderboard](https://zhw-zhang.github.io/ReCo-page/#reco-bench-leaderboard) of models evaluated on ReCo-Bench. Welcome to check it out!

- **🌟 Dataset Usage**: We are excited to see our **ReCo-Data** being used for model training in [Mamoda2.5(ByteDance)](https://arxiv.org/pdf/2605.02641v1), [SAMA(Baidu)](https://arxiv.org/pdf/2603.19228), [LIVE(Kuaishou)](https://arxiv.org/pdf/2604.17021), [Kiwi-Edit(NUS)](https://arxiv.org/pdf/2603.02175), [ISA(HKUST)](https://arxiv.org/pdf/2605.04569), [Aurora(NVIDIA)](https://arxiv.org/pdf/2605.18748), etc. We sincerely thank these great works for building upon **ReCo-Data**.

- **2026.05.24**: As of now, **ReCo-Data** has reached **30.1K downloads**, with a peak monthly download count of **9.75K**. 🎉🚀 We greatly appreciate the community's interest and support.

- **2026.05.01**: ReCo has been accepted to **ICML 2026**. We will update the latest paper version soon.

- **2026.04.25**: Updates released today:
  - **ReCo_ref**: Our ReCo architecture naturally supports IP-reference-conditioned video editing. We further trained a multi-task IP-conditioned editing model with additional Kiwi-Edit data. Additional capabilities include： **IP+text- or text-conditioned object replacement, object insertion, and background changes.**
    - Released training code for `ReCo_ref`, including a mixed dataloader with Kiwi-Edit data.
    - Released inference code for `ReCo_ref`.
    - Released `ReCo_ref` inference videos, intermediate outputs, and final results on **ReCo-Bench**, **RefViE-Bench**, and **OpenVE-Bench**. See [Benchmark Results](#-benchmark-results) for RefViE / OpenVE summary tables and download links.

  - **ReCo_ori**: We released two variants, `2025_m12` with stronger overall performance and `2026_01_16_v1` with improved removal performance.
    - Released `ReCo_ori` inference videos, intermediate outputs, and final results on **ReCo-Bench**.
    - Released a Diffusers-based implementation script for `ReCo_ori` supporting 30-step sampling; note that it uses direct parameter conversion and may incur some quality loss. See `tools/run_reco_diffusers.sh`.
    - Added region-loss training code. See `tools\train_reco_add_region_loss_raw.py`.


- **2026.03.05**: We are excited to see that [Kiwi-Edit (NUS)](https://github.com/showlab/Kiwi-Edit/tree/main) has further refined our **HQ-ReCo dataset** and added reference image pairs. Check out their [DATASET.md](https://github.com/showlab/Kiwi-Edit/blob/main/DATASET.md) for further instructions.

## 🔥 Updates
- [x] **2026.04.25** Release ReCo_Ref-related code.
- [x] **2026.02.26** Release training code.
- [x] **2026.01.16** Release ReCo model weights and inference code.
- [x] **2026.01.16** Upload raw [video object masks](https://huggingface.co/datasets/HiDream-ai/ReCo-Data/tree/main/video_masks) to ReCo-Data.
- [x] **2025.12.23** Release ReCo-Data and usage code.
- [x] **2025.12.23** Release ReCo-Bench and evaluation code.
- [x] **2025.12.22** Upload our arXiv paper.




## 📊 Benchmark Results

Scores for `ReCo` / `ReCo_ref` on ReCo-Bench, RefViE-Bench, and OpenVE-Bench. Full packages are on Hugging Face; the live ReCo-Bench leaderboard (more methods) is on the [project page](https://zhw-zhang.github.io/ReCo-page/#reco-bench-leaderboard).

| Model | Release | Benchmarks | All results (download) |
| --- | --- | --- | --- |
| `ReCo_ori` | 2025-12 | ReCo-Bench | [Hugging Face](https://huggingface.co/datasets/HiDream-ai/ReCo-Bench/tree/main/reco_all_results_2025_m12) |
| `ReCo_ref` | 2026-04 | RefViE-bench, OpenVE-bench, ReCo-Bench | [Hugging Face](https://huggingface.co/datasets/HiDream-ai/ReCo-Bench/tree/main/reco_ref_all_results_2026_m4) |

### ReCo-Bench (`ReCo` / `ReCo_ref`)

Total scores under **Gemini-2.5-flash-thinking**, matching the [ReCo-Bench Leaderboard](https://zhw-zhang.github.io/ReCo-page/#reco-bench-leaderboard). Higher is better.

| Approach | Add | Remove | Replace | Style |
| --- | --- | --- | --- | --- |
| InsViE | 3.05 | 3.16 | 3.17 | 7.91 |
| Lucy-Edit | 6.31 | – | 6.72 | 4.83 |
| Ditto | 7.56 | – | 6.58 | 9.01 |
| VACE | – | 5.19 | – | – |
| ReCo | 8.23 | 7.00 | 8.74 | 9.17 |
| ReCo_Ref | 8.02 | 7.65 | 8.13 | 9.12 |
| VInO | 8.85 | 8.16 | 8.70 | 9.10 |
| Mamoda2.5 | 8.80 | 8.67 | 9.06 | 8.99 |

### RefViE-Bench (`ReCo_ref`)

| Model | Identity Consist. | Temporal Consist. | Physical Consist. | Reference Sim. | Matting Quality | Video Quality | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ReCo_Ref | 2.90 | 2.64 | 2.49 | 2.65 | 2.00 | 1.58 | 2.48 |
| Runway Aleph | 3.79 | 3.65 | 3.58 | 3.33 | 2.81 | 2.58 | 3.29 |
| Kling-O1 | 4.75 | 4.66 | 4.60 | 3.95 | 3.21 | 2.75 | 3.99 |
| Kiwi-Edit (All data) | 3.51 | 2.96 | 2.91 | 3.40 | 2.58 | 2.40 | 2.96 |
| Kiwi-Edit (Ref. data only) | 3.98 | 3.40 | 3.34 | 3.72 | 2.90 | 2.51 | 3.31 |

### OpenVE-Bench (`ReCo_ref`)

| Method | #Params | #Reso. | Overall ↑ | Global Style ↑ | Background Change ↑ | Local Change ↑ | Local Remove ↑ | Local Add ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VACE | 14B | 1280×720 | 1.57 | 1.49 | 1.55 | 2.07 | 1.46 | 1.26 |
| OmniVideo | 1.3B | 640×352 | 1.19 | 1.11 | 1.18 | 1.14 | 1.14 | 1.36 |
| InsViE | 2B | 720×480 | 1.45 | 2.20 | 1.06 | 1.48 | 1.36 | 1.17 |
| Lucy-Edit | 5B | 1280×704 | 2.22 | 2.27 | 1.57 | 3.20 | 1.75 | 2.30 |
| ICVE | 13B | 384×240 | 2.18 | 2.22 | 1.62 | 2.57 | 2.51 | 1.97 |
| DITTO | 14B | 832×480 | 2.13 | 4.01 | 1.68 | 2.03 | 1.53 | 1.41 |
| OpenVE-Edit | 5B | 1280×704 | 2.50 | 3.16 | 2.36 | 2.98 | 1.85 | 2.15 |
| ReCo_Ref | 2.1B | 832×480 | 2.80 | 3.96 | 1.92 | 3.70 | 2.24 | 2.17 |
| Kiwi-Edit (Stage-2 Instruct-Only) | 5B | 720×480 | 2.92 | 3.54 | 3.80 | 2.59 | 2.55 | 2.12 |
| Kiwi-Edit (Stage-2 Instruct-Only) | 5B | 1280×704 | 2.98 | 3.54 | 3.84 | 2.57 | 2.71 | 2.25 |
| Kiwi-Edit (Stage-3 Instruct-Reference) | 5B | 1280×704 | 3.02 | 3.64 | 2.64 | 3.83 | 2.63 | 2.36 |

## 📊 ReCo-Data Preparation

**ReCo-Data** is a large-scale, high-quality video editing dataset consisting of **500K+ instruction–video pairs**, covering four video editing tasks: **object addition (add)**, **object removal (remove)**, **object replacement (replace)**, and **video stylization (style)**.

<p align="center">
    <img src="assets/statistic.png" width="800"/>
<p>

### Downloading ReCo-Data

Please download each task of ReCo-Data into the `./ReCo-Data` directory by running:

```bash
bash ./tools/download_dataset.sh
````

Before downloading the full dataset, you may first browse the
**[visualization examples](https://huggingface.co/datasets/HiDream-ai/ReCo-Data/blob/main/examples.tar)**.

These examples are collected by **randomly sampling 50 instances from each task**
(add, remove, replace, and style), **without any manual curation or cherry-picking**,
and are intended to help users quickly inspect and assess the overall data quality.

Note: The examples are formatted for visualization convenience and do not strictly follow the dataset format.

### Directory Structure

After downloading, please ensure that the dataset follows the directory structure below:

<details open>
<summary>ReCo-Data directory structure</summary>

```text
ReCo-Data/
├── add/
│   ├── add_data_configs.json
│   ├── src_videos/
│   │   ├── video1.mp4
│   │   ├── video2.mp4
│   │   └── ...
│   └── tar_videos/
│       ├── video1.mp4
│       ├── video2.mp4
│       └── ...
├── remove/
│   ├── remove_data_configs.json
│   ├── src_videos/
│   └── tar_videos/
├── replace/
│   ├── replace_data_configs.json
│   ├── src_videos/
│   └── tar_videos/
└── style/
    ├── style_data_configs.json
    ├── src_videos/
    │   ├── video1.mp4
    │   └── ...
    └── tar_videos/
        ├── video1-a_Van_Gogh_style.mp4
        └── ...
```

</details>

### Testing and Visualization

After downloading the dataset, you can directly test and visualize samples from **any single task** using the following script
(taking the **replace** task as an example):

```bash
python scripts/reco_data_test_single.py \
  --json_path ./ReCo-Data/replace/replace_data_configs.json \
  --video_folder ./ReCo-Data \
  --debug
```

### Mixed Task Loading

You can also load a **mixed dataset** composed of the four tasks (**add**, **remove**, **replace**, and **style**) with arbitrary ratios by running:

```bash
python scripts/reco_data_test_mix_data.py \
  --json_folder ./ReCo-Data \
  --video_folder ./ReCo-Data \
  --debug
```

### Notes

* `src_videos/` contains the original source videos.
* `tar_videos/` contains the edited target videos corresponding to each instruction.
* `*_data_configs.json` stores the instruction–video mappings and metadata for each task.



## 📈 Evaluation

### VLLM-based Evaluation Benchmark
<details close>
<summary>ReCo-Bench details</summary>

Traditional video generation metrics often struggle to accurately assess the fidelity and quality of video editing results. Inspired by recent image editing evaluation protocols, we propose a **VLLM-based evaluation benchmark** to comprehensively and effectively evaluate video editing quality.


We collect **480 video–instruction pairs** as the evaluation set, evenly distributed across four tasks: **object addition**, **object removal**, **object replacement**, and **video stylization** (120 pairs per task). All source videos are collected from the **Pexels** video platform.


For local editing tasks (add, remove, and replace), we utilize **Gemini-2.5-Flash-Thinking** to automatically generate diverse editing instructions conditioned on video content. For video stylization, we randomly select **10 source videos** and apply **12 distinct styles** to each, resulting in **120 stylization evaluation pairs**.
</details>

---

### 1. Downloading ReCo-Bench
Please download **ReCo-Bench** into the `./ReCo-Bench` directory by running:
```bash
bash ./tools/download_ReCo-Bench.sh
````



---



### 2. Usage

After downloading the benchmark, you can directly start the evaluation using:
```bash
cd tools
bash run_eval_via_gemini.sh
```


<details close>
<summary>This script performs the evaluation in two stages:</summary>

#### Step 1: Per-dimension Evaluation with Gemini
In the first stage, **Gemini-2.5-Flash-Thinking** is used as a VLLM evaluator to score each edited video across multiple evaluation dimensions.

Key arguments used in this step include:
* `--edited_video_folder`: Path to the folder containing the edited (target) videos generated by the model.

* `--src_video_folder`: Path to the folder containing the original source videos.

* `--base_txt_folder`: Path to the folder containing task-specific instruction configuration files.

* `--task_name`: Name of the evaluation task, one of `{add, remove, replace, style}`.



This step outputs per-video, per-dimension evaluation results in JSON format.

#### Step 2: Final Score Aggregation

After all four tasks have been fully evaluated, the second stage aggregates the evaluation results and computes the final scores.
* `--json_folder`: Path to the JSON output folder generated in Step 1

  (default: `all_results/gemini_results`)

* `--base_txt_folder`: Path to the instruction configuration folder

This step produces the final benchmark scores for each task as well as the overall performance. 


</details>

### 3. Benchmark Results (Downloads and Summaries)

Summary tables for RefViE-Bench / OpenVE-Bench and download links are in [Benchmark Results](#-benchmark-results) above. ReCo-Bench summary markdown remains under [`ReCo-Bench/ReCo_Ref_results_md/`](https://github.com/HiDream-ai/ReCo/tree/main/ReCo-Bench/ReCo_Ref_results_md).

## 🏃 Inference

### 1. Environment Preparation

Create and activate the specialized Conda environment:

```bash
conda create -n reco python=3.11 -y
conda activate reco
pip install -r requirements.txt
```

### 2. Model Weights Setup

You need to prepare both the base model and our specific checkpoints.

<!-- | Model | Source | Description |
| --- | --- | --- |
| **VACE 1.3B** | [🤗 Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B) | Base VACE weights (Place in `./Wan-AI`) |
| **ReCo** | [🤗 Hugging Face](https://huggingface.co/HiDream-ai/ReCo) | Our ReCo checkpoint(Place in `all_ckpts/`). We will update better ckpts progressively afterward | -->

<table>
  <thead>
    <tr>
      <th width="25%" align="center">Model</th>
      <th width="25%" align="center">Source</th>
      <th align="left">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><b>Wan-2.1-VACE-1.3B</b></td>
      <td align="center"><a href="https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B">🤗 Hugging Face</a></td>
      <td>Base VACE weights. Place in <code>./Wan-AI</code></td>
    </tr>
    <tr>
      <td align="center"><b>ReCo_ori</b></td>
      <td align="center"><a href="https://huggingface.co/HiDream-ai/ReCo/blob/main/ReCo_ori_rank128-2025_m12_version.ckpt">🤗 Hugging Face</a></td>
      <td>Our original ReCo checkpoint trained on the four editing tasks. Place in <code>all_ckpts/</code>.</td>
    </tr>
    <tr>
      <td align="center"><b>ReCo_ref</b></td>
      <td align="center"><a href="https://huggingface.co/HiDream-ai/ReCo/blob/main/ReCo_ref_rank256-2026_m4_version.ckpt">🤗 Hugging Face</a></td>
      <td>Our multi-task editing checkpoint with IP-reference conditioning, additionally fine-tuned on Kiwi-Edit data. Supports IP-guided background replacement, object replacement, and object addition. Place in <code>all_ckpts/</code>.</td>
    </tr>
  </tbody>
</table>

**Organize the files as follows:**
```text
.
├── Wan-AI/                      
├── all_ckpts/                   
│   └── ReCo_ori_rank128-2025_m12_version.ckpt
|   |__ ReCo_ref_rank256-2026_m4_version.ckpt
├── assets/                      
└── scripts/
    └── inference_reco_single.py
```

### 3. Running Inference

We provide a bash script to automate the execution of different tasks (Replace, Remove, Style, Add and Propagation). Run the following command:

```bash
bash scripts/infer_server_single.sh
```

To run a specific task manually or customize the execution, use the python command directly:

```bash
python scripts/inference_reco_single.py \
    --task_name replace \
    --test_txt_file_name assets/replace_test.txt \
    --lora_ckpt ReCo_ori_rank128-2025_m12_version.ckpt
```

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `test_txt_file_name` | `str` | `assets/...` | Path to the `.txt` file containing test prompts/configs. |
| `task_name` | `str` | `replace` | Task type: `remove`, `replace`, `add`, `style`. Use the `_wf` suffix (e.g., `remove_wf`) for **Propagation tasks** given the first frame. |
| `base_video_folder` | `str` | `assets/test_videos` | Directory containing the source videos. |
| `base_wan_folder` | `str` | `./Wan-AI` | Path to the pre-trained Wan-AI model weights. |
| `lora_ckpt` | `str` | `all_ckpts/...` | Path to the specific LoRA checkpoint file. |

### 4. Running Inference with IP condition

Run the IP-conditioned inference script:

```bash
bash scripts/infer_server_single_ref_rank256.sh
```

This script calls `scripts/inference_reco_single_ref.py` and demonstrates different modes: prompt-only, IP-image-only, first-frame-only, or using both IP image and first-frame conditioning together.


## 🚀 Training

### 1) Basic Training

Run:

```bash
bash scripts/train.sh
```

Before launching training:

* Update the **pretrained model weight paths** in your script to match local paths.
* In `scripts/train.py`, update dataset paths in `LightningModelForTrain.train_dataloader`:
  * **JSON annotation directory**
  * **Video data directory**

### 2) Multi-task Training with IP Reference Data

We additionally provide multi-task training code with IP-image references, which additionally supports:

* background replacement with a given reference image
* object replacement with a given reference image
* object addition with a given reference image

To start this training pipeline, follow two steps:

**Step 1. Prepare data configs and local paths**

1. Download task config/data package from  
   [kiwidata.zip](https://huggingface.co/HiDream-ai/ReCo/blob/main/kiwidata/kiwidata.zip),  
   then place/extract it under the current project.
2. Update related configs in:
   * `scripts/train_multitask_add_kiwi_ref_data.py` (around `196-210`)
   * `kiwidata/test_dataset_mixdata.py` (around `32-54`)

**Step 2. Launch training**

```bash
bash scripts/train_multitask_add_kiwi_ref_data.sh
```

This pipeline includes mixed data loading from **ReCo-Data**, **DiTTO**, and **OpenVE-3M**, and also integrates kiwi-edit paired IP-reference data. The `kiwidata` folder provides our filtered and organized config files that better match the original dataset formats. Feel free to use and adapt them.


## 🌟 Star and Citation
If you find our work helpful for your research, please consider giving a star⭐ on this repository and citing our work.
```
@article{zhang2025region,
  title={Region-Constraint In-Context Generation for Instructional Video Editing},
  author={Zhang, Zhongwei and Long, Fuchen and Li, Wei and Qiu, Zhaofan and Liu, Wu and Yao, Ting and Mei, Tao},
  journal={arXiv preprint arXiv:2512.17650},
  year={2025}
}
```


## 💖 Acknowledgement
<span id="acknowledgement"></span>

Our code is inspired by several works, including [WAN](https://github.com/Wan-Video/Wan2.1), [ObjectClear](https://github.com/zjx0101/ObjectClear)--a strong object remover, [VACE](https://github.com/ali-vilab/VACE), [Flux-Kontext-dev](https://github.com/black-forest-labs/flux). Thanks to all the contributors! 


