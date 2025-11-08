![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)

# MG-Gen: Single Image to Motion Graphics Generation with Layer Decopmosition
This is an official repository for MG-Gen.  
MG-Gen is a novel method to generate motion graphics from a single raster image preserving input content consistency with dynamic text motion.

- Paper: https://arxiv.org/abs/2504.02361
- Project Page: https://cyberagentailab.github.io/mggen/

# Setup Experimental Environment
Create and activate a Python venv. (Requirements: python >= 3.10)
```
python -m venv .venv
source .venv/bin/activate
```

Install dependencies.
```
pip install -r requirements.txt
playwright install
```

Download anime.min.js
```
mkdir src/libs/
curl -o src/libs/anime.min.js https://raw.githubusercontent.com/juliangarnier/anime/v4.2.2/lib/anime.min.js
```

Download the weights from GCS.
If you have not installed gsutil, see the [installation instructions](https://cloud.google.com/storage/docs/gsutil_install?hl=en).
```
gsutil -m cp gs://ailab-public/image-to-video/mggen/weights.zip .
unzip weights.zip
```

# Generate motion graphics from images
Set your OpenRouter API_KEY in `.env`. You can get an API key from [OpenRouter](https://openrouter.ai/). The default model is `openrouter/polaris-alpha`, but you can customize it using environment variables.

Required:
```
echo "OPENROUTER_API_KEY=\"your-api-key\"" > .env
```

Optional (customize endpoint and model):
```
echo "OPENROUTER_BASE_URL=\"https://openrouter.ai/api/v1\"" >> .env
echo "OPENROUTER_MODEL=\"openrouter/polaris-alpha\"" >> .env
```

Optional (enable alternative providers and retrieval):
```
# OpenRouter models (text + vision)
echo "OPENROUTER_MODEL_TEXT=\"openrouter/polaris-alpha\"" >> .env
echo "OPENROUTER_MODEL_VISION=\"google/gemini-2.5-flash-image\"" >> .env

# Local DeepSeek-OCR (Hugging Face transformers)
echo "DEEPSEEK_OCR_MODEL=\"deepseek-ai/DeepSeek-OCR\"" >> .env
# Optional overrides
echo "HF_HOME=\"/path/to/hf-cache\"" >> .env
echo "DEEPSEEK_DEVICE=\"cuda\"" >> .env   # or "cpu"
echo "DEEPSEEK_BASE_SIZE=\"1024\"" >> .env
echo "DEEPSEEK_IMAGE_SIZE=\"640\"" >> .env
echo "DEEPSEEK_CROP_MODE=\"1\"" >> .env   # 1=true, 0=false
```

Start a gradio demo sever.
```
cd src
python demo.py
```

Run a generation batch script.
```
cd src
python inference_batch.py --testset_dir "../example_inputs" 
```

Demo videos
![demo videos](gradio_demo.gif)

# License
This project is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](https://www.gnu.org/licenses/agpl-3.0.en.html).

# Citation
```bibtex
@article{shirakawa2025mg,
  title={MG-Gen: Single Image to Motion Graphics Generation with Layer Decomposition},
  author={Shirakawa, Takahiro and Suzuki, Tomoyuki and Haraguchi, Daichi},
  journal={arXiv preprint arXiv:2504.02361},
  year={2025}
}
```

## Notes on Provider and OCR
- The animation script generator uses OpenRouter by default; set `OPENROUTER_MODEL` (default: `google/gemini-2.5-flash-image`).
- Optional DeepSeek-OCR refinement affects text content only (not geometry). Enable via `AI_CONF_default1.ocr.deepseek_refine: true`. This repository now uses the local Hugging Face transformers model (no remote endpoint required). GPU is strongly recommended; CPU works but will be slow.
