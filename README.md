# ReCo

<p align="center">
    <img src="assets/title_imgs.png" width="400"/>
<p>

<p align="center">
    🖥️ <a href="https://github.com/HiDream-ai/ReCo">GitHub</a> &nbsp&nbsp ｜ &nbsp&nbsp  🌐 <a href="https://zhw-zhang.github.io/ReCo-page/"><b>Project Page</b></a> &nbsp&nbsp  | &nbsp&nbsp🤗 <a href="https://huggingface.co/datasets/HiDream-ai/ReCo-Data">ReCo-Data</a>&nbsp&nbsp | &nbsp&nbsp 📈 <a href="https://huggingface.co/datasets/HiDream-ai/ReCo-Bench">ReCo-Bench</a>&nbsp&nbsp | &nbsp&nbsp 🤗 <a href="https://huggingface.co/HiDream-ai/ReCo">ReCo-Models  </a> &nbsp&nbsp | &nbsp&nbsp 📖 <a href="https://arxiv.org/abs/2512.17650">Paper</a> &nbsp&nbsp 
<br>
 

[**ReCo: Region-Constraint In-Context Generation for Instructional Video Editing**](https://zhw-zhang.github.io/ReCo-page/) <be>

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

- **Dataset Usage**: We are excited to see our **ReCo-Data** being used for model training in [Mamoda2.5](https://arxiv.org/pdf/2605.02641v1), [SAMA](https://arxiv.org/pdf/2603.19228), [LIVE](https://arxiv.org/pdf/2604.17021), [Kiwi-Edit](https://arxiv.org/pdf/2603.02175), [ISA](https://arxiv.org/pdf/2605.04569), etc. Thanks for these great works!

- **2026.05.01**: ReCo has been accepted to **ICML 2026**. We will update the latest paper version soon.

- **2026.04.25**: Updates released today:
  - **ReCo_ref**: Our ReCo architecture naturally supports IP-reference-conditioned video editing. We further trained a multi-task IP-conditioned editing model with additional Kiwi-Edit data.
    - Released training code for `ReCo_ref`, including a mixed dataloader with Kiwi-Edit data.
    - Released inference code for `ReCo_ref`.
    - Released `ReCo_ref` inference videos, intermediate outputs, and final results on **ReCo-Bench**, **RefViE-Bench**, and **OpenVE-Bench**. See [ReCo-Bench/ReCo_Ref_results_md/](https://github.com/HiDream-ai/ReCo/tree/main/ReCo-Bench/ReCo_Ref_results_md)

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
python reco_data_test_single.py \
  --json_path ./ReCo-Data/replace/replace_data_configs.json \
  --video_folder ./ReCo-Data \
  --debug
```

### Mixed Task Loading

You can also load a **mixed dataset** composed of the four tasks (**add**, **remove**, **replace**, and **style**) with arbitrary ratios by running:

```bash
python reco_data_test_mix_data.py \
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

We provide downloadable evaluation outputs for different model releases. Summary tables/markdown files are stored in this repository, while full result packages are hosted on Hugging Face.

| Model | Release | Benchmarks | All results (download) | Summary tables |
| --- | --- | --- | --- | --- |
| `ReCo_ori` | 2025-12 | ReCo-Bench | [Hugging Face](https://huggingface.co/datasets/HiDream-ai/ReCo-Bench/tree/main/reco_all_results_2025_m12) | The ReCo paper |
| `ReCo_ref` | 2026-04 | RefViE-bench, OpenVE-bench, ReCo-Bench | [Hugging Face](https://huggingface.co/datasets/HiDream-ai/ReCo-Bench/tree/main/reco_ref_all_results_2026_m4) | [ReCo-Bench/ReCo_Ref_results_md/](https://github.com/HiDream-ai/ReCo/tree/main/ReCo-Bench/ReCo_Ref_results_md) |

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
└── inference_reco_single.py
```

### 3. Running Inference

We provide a bash script to automate the execution of different tasks (Replace, Remove, Style, Add and Propagation). Run the following command:

```bash
bash infer_server_single.sh
```

To run a specific task manually or customize the execution, use the python command directly:

```bash
python inference_reco_single.py \
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
bash infer_server_single_ref_rank256.sh
```

This script calls `inference_reco_single_ref.py` and demonstrates different modes: prompt-only, IP-image-only, first-frame-only, or using both IP image and first-frame conditioning together.


## 🚀 Training

### 1) Basic Training

Run:

```bash
bash train.sh
```

Before launching training:

* Update the **pretrained model weight paths** in your script to match local paths.
* In `train.py`, update dataset paths in `LightningModelForTrain.train_dataloader`:
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
   * `train_multitask_add_kiwi_ref_data.py` (around `196-210`)
   * `kiwidata/test_dataset_mixdata.py` (around `32-54`)

**Step 2. Launch training**

```bash
bash train_multitask_add_kiwi_ref_data_2node.sh
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


