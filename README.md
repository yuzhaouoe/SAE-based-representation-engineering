

# Steering Knowledge Selection Behaviours in LLMs via SAE-Based Representation Engineering

---

This repository hosts the data and code of the paper: [Steering Knowledge Selection Behaviours in LLMs via SAE-Based Representation Engineering](https://arxiv.org/pdf/2410.15999)

---


## Project Setup
```bash
conda create -n spare python=3.9 -y
conda activate spare
bash ./scripts/install.sh
```

## Run SpARE


```bash
python ./demo.py
```

Test your cases by replacing the `test_examples` list in `demo.py` with your own dictionaries containing `context` and `question` keys.

SpARE currently only supports short-form ODQA task, and we plan to add support for more tasks in the next version. 

## Run Experiments

Use the cached intermediate data to run experiments.

The cached data is in the `cache_data` folder, including mutual information, expectation, and the values of functional SAE activations.  

```bash
bash ./scripts/run_all_experiments.sh
```

## Run SpARE Step by Step

Observe the outputs of prompts and group them based on the knowledge selection behaviours:
```bash
bash ./scripts/run_group_prompts.sh
```

Save the activations of grouped prompts:
```bash
bash ./scripts/run_save_grouped_activations.sh
```

Estimate the mutual information and expectations for each SAE activation: 

```bash
bash ./scripts/run_mutual_information_and_expectations.sh
```

Evaluate SpARE
```bash
python ./scripts/run_spare.py \
  --model_path="meta-llama/Llama-2-7b-hf" \
  --data_name="nqswap" \
  --layer_ids 12 13 14 15 \
  --edit_degree=2.0 \
  --select_topk_proportion=0.07 \
  --seed=42 \
  --hiddens_name="grouped_activations" \
  --mutual_information_save_name="mutual_information" \
  --run_use_parameter \
  --run_use_context
```

## Acknowledgement

The implementation of the sparse auto-encoder is adapted from EleutherAI/sae https://github.com/EleutherAI/sae and jbloomAus/SAELens https://github.com/jbloomAus/SAELens.
We appreciate their open-source contributions! 

## Citing

Steering Knowledge Selection Behaviours in LLMs via SAE-Based Representation Engineering
```text
@misc{zhao2024steeringknowledgeselectionbehaviours,
      title={Steering Knowledge Selection Behaviours in LLMs via SAE-Based Representation Engineering}, 
      author={Yu Zhao and Alessio Devoto and Giwon Hong and Xiaotang Du and Aryo Pradipta Gema and Hongru Wang and Xuanli He and Kam-Fai Wong and Pasquale Minervini},
      year={2024},
      eprint={2410.15999},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.15999}, 
}
```

The preliminary study: Analysing the Residual Stream of Language Models Under Knowledge Conflicts

```text
@misc{zhao2024analysingresidualstreamlanguage,
      title={Analysing the Residual Stream of Language Models Under Knowledge Conflicts}, 
      author={Yu Zhao and Xiaotang Du and Giwon Hong and Aryo Pradipta Gema and Alessio Devoto and Hongru Wang and Xuanli He and Kam-Fai Wong and Pasquale Minervini},
      year={2024},
      eprint={2410.16090},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.16090}, 
}
```
