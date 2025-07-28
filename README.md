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
mi_threshold=0.004 

# Training
generator.fit('../house/train.csv', max_sample_num=None,\
                batch_size=32, epochs=100, lr=1e-4, max_length=300, shuffle=True,\
                early_stopping_rounds=5, mi_threshold=mi_threshold, mi_n_bins=5,\
                constrain_string_values=True, val_ratio=0.001, num_workers=8, gradient_accumulation_steps=1, drop=[])

# Sampling 1000 new samples
sampled_df=generator.sample(1000, 
                                 temperature=1.0, 
                                 p=0.7, 
                                 mi_threshold=mi_threshold, 
                                 apply_final_constraints=True,
                                 copy_factor=1, 
                                )
```
