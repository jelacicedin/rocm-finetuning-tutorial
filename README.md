# LLM Fine-Tuning on ROCm (AMD 7900XTX)

This repository reproduces and extends the tutorial by Venelin Valkov on fine-tuning large language models, adapted to run on AMD GPUs using ROCm. It demonstrates:

- Data loading from SQLite databases
- Custom preprocessing for text classification (subject, sentiment, subjectivity)
- Fine-tuning HuggingFace LLMs using `transformers` and `trl` libraries
- ROCm compatibility and performance testing on the AMD 7900XTX

## Tutorial Reference

Original tutorial by Venelin Valkov:  
[https://www.youtube.com/watch?v=_KPEoCSKHcU](https://www.youtube.com/watch?v=_KPEoCSKHcU)

## AMD Reference
Refer to the AMD Docs website:
[https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/fine-tuning/single-gpu-fine-tuning-and-inference.html](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/fine-tuning/single-gpu-fine-tuning-and-inference.html)
## Hardware Setup

- GPU: AMD Radeon RX 7900 XTX
- ROCm: Verified on ROCm 6.3.4
- PyTorch: Built with ROCm backend

## Dependencies

Install dependencies:

```bash
pip install -r requirements.txt
```

Example requirements.txt:

makefile
Copy
Edit
transformers
trl
datasets
scikit-learn
seaborn
matplotlib
pandas
sqlite3
torch==<rocm-compatible-version>

## Notebook Overview

llm-fine-tuning.ipynb: Main Jupyter notebook for preprocessing, visualization, and fine-tuning.

Uses SFTTrainer from trl for supervised fine-tuning.

Visualizes class distributions and performs initial exploratory data analysis.

## Data
The dataset is stored in SQLite format:

```bash
crypto-news-db/
    ├── crypto-news.db
        ├── train
        ├── validation
        ├── test
```

Each table contains:

text: the article text

subject, sentiment, subjectivity: the target labels

Notes on ROCm Usage
Ensure your PyTorch version is compiled with ROCm and that your environment supports AMD GPU acceleration.

```python
import torch
print(torch.version.hip)
print(torch.cuda.is_available())  # False on ROCm
print(torch.backends.mps.is_available())  # Also False; not applicable
```
