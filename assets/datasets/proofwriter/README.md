---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
  - split: validation
    path: data/validation-*
dataset_info:
  features:
  - name: id
    dtype: string
  - name: maxD
    dtype: int64
  - name: NFact
    dtype: int64
  - name: NRule
    dtype: int64
  - name: theory
    dtype: string
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: QDep
    dtype: int64
  - name: QLen
    dtype: float64
  - name: allProofs
    dtype: string
  - name: config
    dtype: string
  splits:
  - name: train
    num_bytes: 849283641
    num_examples: 585552
  - name: test
    num_bytes: 262359174
    num_examples: 174476
  - name: validation
    num_bytes: 121834168
    num_examples: 85468
  download_size: 42944158
  dataset_size: 1233476983
---
# Dataset Card for "proofwriter"

[More Information needed](https://github.com/huggingface/datasets/blob/main/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)