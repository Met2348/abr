---
annotations_creators:
- crowdsourced
language_creators:
- crowdsourced
language:
- en
license:
- cc-by-sa-4.0
multilinguality:
- monolingual
size_categories:
- 10K<n<100K
source_datasets:
- original
task_categories:
- question-answering
- text2text-generation
task_ids:
- extractive-qa
- abstractive-qa
paperswithcode_id: drop
pretty_name: DROP
dataset_info:
  features:
  - name: section_id
    dtype: string
  - name: query_id
    dtype: string
  - name: passage
    dtype: string
  - name: question
    dtype: string
  - name: answers_spans
    sequence:
    - name: spans
      dtype: string
    - name: types
      dtype: string
  splits:
  - name: train
    num_bytes: 105572506
    num_examples: 77400
  - name: validation
    num_bytes: 11737755
    num_examples: 9535
  download_size: 11538387
  dataset_size: 117310261
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
---

# Dataset Card for "drop"

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:** https://allenai.org/data/drop
- **Repository:** [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)
- **Paper:** https://aclanthology.org/N19-1246/
- **Paper:** https://arxiv.org/abs/1903.00161
- **Point of Contact:** [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)
- **Size of downloaded dataset files:** 8.30 MB
- **Size of the generated dataset:** 110.91 MB
- **Total amount of disk used:** 119.21 MB

### Dataset Summary

DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs.
. DROP is a crowdsourced, adversarially-created, 96k-question benchmark, in which a system must resolve references in a
question, perhaps to multiple input positions, and perform discrete operations over them (such as addition, counting, or
 sorting). These operations require a much more comprehensive understanding of the content of paragraphs than what was
 necessary for prior datasets.

### Supported Tasks and Leaderboards

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Languages

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

## Dataset Structure

### Data Instances

#### default

- **Size of downloaded dataset files:** 8.30 MB
- **Size of the generated dataset:** 110.91 MB
- **Total amount of disk used:** 119.21 MB

An example of 'validation' looks as follows.
```
This example was too long and was cropped:

{
    "answers_spans": {
        "spans": ["Chaz Schilens"]
    },
    "passage": "\" Hoping to rebound from their loss to the Patriots, the Raiders stayed at home for a Week 16 duel with the Houston Texans.  Oak...",
    "question": "Who scored the first touchdown of the game?"
}
```

### Data Fields

The data fields are the same among all splits.

#### default
- `passage`: a `string` feature.
- `question`: a `string` feature.
- `answers_spans`: a dictionary feature containing:
  - `spans`: a `string` feature.

### Data Splits

| name  |train|validation|
|-------|----:|---------:|
|default|77409|      9536|

## Dataset Creation

### Curation Rationale

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Source Data

#### Initial Data Collection and Normalization

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

#### Who are the source language producers?

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Annotations

#### Annotation process

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

#### Who are the annotators?

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Personal and Sensitive Information

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Discussion of Biases

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Other Known Limitations

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

## Additional Information

### Dataset Curators

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Licensing Information

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Citation Information

```
@inproceedings{Dua2019DROP,
  author={Dheeru Dua and Yizhong Wang and Pradeep Dasigi and Gabriel Stanovsky and Sameer Singh and Matt Gardner},
  title={  {DROP}: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs},
  booktitle={Proc. of NAACL},
  year={2019}
}

```


### Contributions

Thanks to [@patrickvonplaten](https://github.com/patrickvonplaten), [@thomwolf](https://github.com/thomwolf), [@mariamabarham](https://github.com/mariamabarham), [@lewtun](https://github.com/lewtun) for adding this dataset.