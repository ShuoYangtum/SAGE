# SAGE: Sparse Adaptive Guidance for\\Dependency-Aware Tabular Data Generation

====

Data sets
----
All of the datasets we used are open-soursed.<br>
Adult Income dataset: [https://www.kaggle.com/datasets/wenruliu/adult-income-dataset](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset)<br>
HELOC dataset: [https://www.kaggle.com/datasets/averkiyoliabev/home-equity-line-of-creditheloc](https://www.kaggle.com/datasets/averkiyoliabev/home-equity-line-of-creditheloc)<br>
Iris dataset: [https://archive.ics.uci.edu/dataset/53/iris](https://archive.ics.uci.edu/dataset/53/iris)<br>
California Housing dataset: [https://www.kaggle.com/datasets/camnugent/california-housing-prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices)<br>
The CDC dataset: [https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)<br>
The MIC dataset: [https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications](https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications)<br>


Setup
----
To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Quickstart

```python
from sage import SAGE
import torch

generator = SAGE(model_name="meta-llama/Llama-3.2-1B", device=torch.device("cuda:1"))
mi_threshold = None  # None -> use median MI from training set (paper default)

# Training
generator.fit('../house/train.csv', max_sample_num=None,\
                batch_size=32, epochs=100, lr=1e-4, max_length=300, shuffle=True,\
                early_stopping_rounds=5, mi_threshold=mi_threshold, mi_n_bins=5,\
                mi_strategy='fd', constrain_string_values=True, val_ratio=0.001, num_workers=8, gradient_accumulation_steps=1, drop=[])

# Sampling 1000 new samples
sampled_df=generator.sample(1000, 
                                 temperature=1.0, 
                                 p=0.7, 
                                 mi_threshold=mi_threshold, 
                                 apply_final_constraints=True,
                                 copy_factor=1, 
                                )
```

## CLI usage

You can run the full train + imputation pipeline with command-line arguments:

```bash
python run_sage.py \
  --target-column <TARGET_COLUMN> \
  --train-path train.csv \
  --test-path test.csv \
  --output-path ../output.csv \
  --model-name gpt2 \
  --device cuda:0 \
  --seed 42 \
  --deterministic \
  --checkpoint-path best_generator_model.pt \
  --epochs 100 \
  --batch-size 32 \
  --lr 1e-4 \
  --max-length 50 \
  --val-ratio 0.001 \
  --early-stopping-rounds 5 \
  --mi-n-bins 10 \
  --mi-strategy fd \
  --mi-threshold 0.0001 \
  --constrain-string-values \
  --num-workers 8 \
  --gradient-accumulation-steps 1 \
  --shuffle
```

If you omit `--mi-threshold`, SAGE uses the median MI computed on the training set (paper default).

## Important arguments

- `--mi-strategy`: MI preprocessing discretization strategy. `fd` follows the paper (Freedman-Diaconis bin rule with cap at 16 bins for numerical features).
- `--mi-threshold`: Threshold used by Feature Selector. Leave unset to use dataset-specific median MI.
- `--constrain-string-values`: Enables constrained decoding for value tokens (reduces invalid generations).
- `--disable-final-constraints`: Skip post-generation value correction (enabled by default).
- `--drop-columns`: Comma-separated columns to drop before training (e.g., `--drop-columns id,timestamp`).
- `--seed`: Global random seed for Python/NumPy/PyTorch.
- `--deterministic`: Enforce deterministic backend behavior for better reproducibility.
- `--checkpoint-path`: Save/load path for the best validation checkpoint.

## Notes on paper-aligned implementation

- Numerical pseudo-feature discretization now supports `fd` rule.
- Logit correction follows the paper form: scaling logits with `1 + lambda * (mu_sample / mu_train - 1)`.
- Training computes loss on value tokens (template/separator tokens are masked).
