# Curriculum-Aligned Math Question Generation for Teacher Professional Development
🏫 **University of Adelaide (UoA)** | 📘 **COMP SCI 7104A & 7104B: Research Project Master Computer Science**

---

This repository contains the research code used to fine-tune and evaluate language models for curriculum-aligned mathematics question generation.

The main workflow is in `main.ipynb`. It covers dataset preparation, AS-ES labelling, model-selection experiments, generation-parameter tuning, LoRA fine-tuning, curriculum-similarity evaluation, and second-stage fine-tuning for year-specific question generation.

## Repository Structure

```text
.
├── main.ipynb                  # Main research notebook
├── trainer.py                  # Hugging Face Trainer wrapper with optional LoRA
├── math_question_generator.py  # Prompting and generation utilities
├── feedback_fine_tuning.py     # Feedback/preference-data fine-tuning workflow
├── PHI2_Utils.py               # Phi-2 embedding and similarity helpers
├── as_es_learning.py           # AS-ES labelling utilities
├── custom_data_collator.py     # Causal-LM data collator
├── evaluation_matrix.py        # BLEU and proficiency matching metrics
├── optimizer.py                # Optuna hyperparameter optimization
├── aqua_dataset.py             # AQuA dataset utilities
├── mwp_dataset.py              # Math word-problem dataset utilities
├── utils.py                    # Device, equation parsing, and validation helpers
├── DPO_data/                   # Curated preference/question data
└── curriculum_Desc/            # Curriculum descriptions and generated/evaluated data
```

## Important Notes Before Running

- `main.ipynb` and `as_es_learning.py` currently import `gsm8k_dataset`, but `gsm8k_dataset.py` is not present in this workspace. Add that source file or update the imports before publishing/running from a clean clone.
- Large trained models, checkpoints, logs, zip archives, virtual environments, and frontend dependency folders are excluded by `.gitignore`.
- Do not commit access tokens. Use environment variables or notebook secrets, for example `HF_TOKEN`.
- Some cells require substantial GPU memory and download models from Hugging Face.

## Setup

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For gated Hugging Face models, authenticate without hard-coding tokens:

```bash
export HF_TOKEN="your_huggingface_token"
```

Then open the notebook:

```bash
jupyter notebook main.ipynb
```

## Workflow Summary

1. Prepare math-problem prompts and train/test splits.
2. Generate AS-ES labels and evaluation metrics.
3. Compare candidate models including LLaMA, DeepSeek, Gemma, GPT-NeoX, and Phi-2.
4. Tune generation parameters with Optuna.
5. Fine-tune the selected Phi-2 model with LoRA.
6. Generate year-specific mathematics questions.
7. Evaluate generated questions against curriculum descriptions using embedding similarity.
8. Use filtered generations for second-stage fine-tuning.

## Artifacts

Model outputs are intentionally not versioned in Git. Store large artifacts such as trained adapters, checkpoints, and experiment logs in a release, external storage, or a model hub.

Recommended artifact locations:

- Hugging Face Hub for trained adapters/tokenizers.
- Zenodo, Figshare, or institutional storage for research datasets and experiment outputs.
- GitHub Releases only for small reproducibility artifacts.

## Reproducibility Checklist

Before making the repository public:

- Add the missing `gsm8k_dataset.py` source file or remove that dependency.
- Re-run or clear notebook outputs to avoid committing local paths, usernames, and large outputs.
- Confirm that all datasets can be redistributed, or replace them with download/preparation instructions.
- Add a license suitable for the code and compatible with any included data.
- Revoke any Hugging Face token that was previously stored in notebook history.
