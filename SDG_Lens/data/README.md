---
dataset_info:
  features:
  - name: text
    dtype: string
  - name: embedding
    sequence: float64
  - name: labels
    sequence: int64
  - name: metadata
    struct:
    - name: country
      dtype: string
    - name: file_id
      dtype: string
    - name: language
      dtype: string
    - name: locality
      dtype: string
    - name: size
      dtype: string
    - name: type
      dtype: string
    - name: year
      dtype: int64
  splits:
  - name: train
    num_bytes: 124052504
    num_examples: 5880
  - name: test
    num_bytes: 36948683
    num_examples: 1470
  download_size: 129951175
  dataset_size: 161001187
configs:
- config_name: default
  data_files:
  - split: train
    path: train-*
  - split: test
    path: test-*
license: cc-by-nc-sa-4.0
task_categories:
- text-classification
language:
- en
- es
- fr
tags:
- sustainable-development-goals
- sdgs
pretty_name: SDGi Corpus
size_categories:
- 1K<n<10K
---
# Dataset Card for SDGi Corpus

Standalone SDG Lens stores the copied SDGi parquet files directly in this
`data/` directory.

<!-- Provide a quick summary of the dataset. -->

SDGi Corpus is a curated dataset for text classification by the [United Nations Sustainable Development Goals (SDGs)](https://www.un.org/sustainabledevelopment/sustainable-development-goals/).

## Dataset Details

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->

SDG Integration Corpus (SDGi Corpus) is the most comprehensive multilingual collection of texts labelled by Sustainable 
Development Goals (SDGs) to date. Designed for multi-label multilingual classification, SDGi Corpus contains over 7,000 
examples in English, French and Spanish. Leveraging years of international SDG reporting on the national and subnational 
levels, we hand-picked texts from Voluntary National Reviews (VNRs) and Voluntary Local Reviews (VLRs) from more than 180 
countries to create an inclusive dataset that provides both focused and broad perspectives on the SDGs. The dataset comes 
with a predefined train/test split.

- **Curated by:** United Nations Development Programme
- **Language(s):** English, French and Spanish
- **License:** CC BY-NC-SA 4.0

### Dataset Sources [optional]

<!-- Provide the basic links for the dataset. -->

- **Repository:** https://github.com/UNDP-Data/dsc-sdgi-corpus (benchmarks)
- **Paper:** https://ceur-ws.org/Vol-3764/paper3.pdf
- **Demo:** TBA.

## Uses

<!-- Address questions around how the dataset is intended to be used. -->

The dataset is designed primarily for text classification tasks – including binary, multiclass and multi-label classification – 
in one or more of the three supported languages. The dataset includes rich metadata with provenance information and can be used for 
other text mining tasks like topic modelling or quantitative text analysis with a focus on the 2030 Agenda for Sustainable Development.

### Direct Use

<!-- This section describes suitable use cases for the dataset. -->

The dataset can be directly used for training machine learning models for text classification tasks. It can also be used for topic modelling to 
identify the main themes that occur in the corpus or a specific subset of it. The rich metadata provided makes it possible to conduct both a trageted or comparative
analyses along linguistic, geographic (country and/or locality) and temporal dimensions.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the dataset will not work well for. -->

The dataset is not suitable for tasks that require information not included in the dataset, such as image analysis or audio processing. 
It cannot be used for predicting future trends or patterns in the SDGs and is not linked to SDG indicator data directly.

## Dataset Structure

The dataset consists of `7350` examples, with `5880` in the training set and `1470` in the test set. Each example includes the following fields:

- `text`: `str` – the text of the example in the original language.
- `embedding`: `list[float]` – 1536-dimensional embedding from OpenAI's `text-embedding-ada-002` model.
- `labels`: `list[int]` – one or more integer labels corresponding to SDGs. About 89% of the examples have just one label.
- `metadata`: `dict` – a dictionary containing metadata information, including:
  - `country`: `str` – ISO 3166-1 alpha-3 code.
  - `file_id`: `str` – internal ID of the original file. Used for provenance and troubleshooting only.
  - `language`: `str` – one of the three supported languages, i.e., `en` (English), `fr` (French), `es` (Spanish).
  - `locality`: `str` – name of the locality within `country` for examples from VLRs, e.g., city, province or region name.
  - `size`: `str` – the size group of the example in terms of tokens, i.e., `s` (small, approx. < 512 tokens), `m` (medium, approx. 512-2048 tokens), `l` (large, approx. > 2048 tokens).
  - `type`: `str` – one of the two document types, i.e., `vnr` (Voluntary National Review) or `vlr` (Voluntary Local Review).
  - `year`: `int` – year of the publication.

<aside class="note">
  <b>Note:</b>
  the embeddings were produced from texts after removing digits. Embedding raw `text` will not produce the same result.
  After applying the following replacements, you should be able to obtain similar emebedding vectors:
</aside>

```python
re.sub(r'(\b\d+[\.\,]?\d*\b)', 'NUM', text)
```

The dataset comes with a predefined train/test split. The examples for the test set were not sampled at random. Instead, they were 
sampled in a stratified fashion using weights proportional to the cross-entropy loss of a simple classifier fitted on the full dataset. 
For details on the sampling process, refer to the paper.

## Dataset Creation

### Curation Rationale

<!-- Motivation for the creation of this dataset. -->

The dataset was created to facilitate automated analysis of large corpora with respect to the 2030 Agenda for Sustainable Development. 
The dataset comprises texts from Voluntary National Reviews (VNRs) and Voluntary Local Reviews (VLRs) which are arguably the most 
authoritative sources of SDG-related texts. The dataset is a collection of texts labelled by the source data producets, the curators 
have not labelled any data themselves.

### Source Data

<!-- This section describes the source data (e.g. news text and headlines, social media posts, translated sentences, ...). -->

All examples were collected from one of the two sources:

- [Voluntary National Reviews (VNRs)](https://hlpf.un.org/vnrs)
- [Voluntary Local Reviews (VLRs)](https://sdgs.un.org/topics/voluntary-local-reviews)

Only Reviews in English, French and Spanish published between January 2016 and December 2023 were included.

#### Data Collection and Processing

<!-- This section describes the data collection and processing process such as data selection criteria, filtering and normalization methods, tools and libraries used, etc. -->

To create SDGi Corpus, we manually analysed each document, searching and extracting specific parts clearly linked to SDGs.
Our curation process can be summarised in 4 steps as follows:

1. Manually examine a given document to identify SDG-labelled content.
2. Extract pages containing relevant content to SDG-specific folders.
3. Edit extracted pages to redact (mask) irrelevant content before and after the relevant content.
4. For content linked to multiple SDGs, fill out a metadata sheet.

For details on the curation process, refer to the paper.

#### Who are the source data producers?

<!-- This section describes the people or systems who originally created the data. It should also include self-reported demographic or identity information for the source data creators if this information is available. -->

Voluntary National Reviews (VNRs) and Voluntary Local Reviews (VLRs) are typically produced by government agencies, national 
statistical offices, and other relevant national and subnational institutions within each country. These entities are responsible 
for collecting, analysing, and reporting on the progress of their respective countries towards the SDGs. In addition, international 
organisations, civil society organisations, academia, and other stakeholders may also contribute to the data collection and reporting 
process for VNRs and VLRs.

### Annotations

<!-- If the dataset contains annotations which are not part of the initial data collection, use this section to describe them. -->

The labels in the dataset come directly from the source documents. No label annotation has been performed to produce SDGi Corpus.

#### Annotation process

<!-- This section describes the annotation process such as annotation tools used in the process, the amount of data annotated, annotation guidelines provided to the annotators, interannotator statistics, annotation validation, etc. -->

Not applicable.

#### Who are the annotators?

<!-- This section describes the people or systems who created the annotations. -->

Not applicable.

#### Personal and Sensitive Information

<!-- State whether the dataset contains data that might be considered personal, sensitive, or private (e.g., data that reveals addresses, uniquely identifiable names or aliases, racial or ethnic origins, sexual orientations, religious beliefs, political opinions, financial or health data, etc.). If efforts were made to anonymize the data, describe the anonymization process. -->

While VNR and VLR texts are unlikely to contain any sensitive Personally Identifiable Information (PII) due to their public nature 
and intented use, users should adhere to ethical standards and best practices when handling the dataset. Should sensitive PII 
information be found in the dataset, you are strongly encouraged to notify the curators.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

- **Language Bias**: The dataset includes texts in three languages, with English (71.9%) examples dominating the dataset, followed by examples in Spanish (15.9%) and French (12.2%). The performance of models trained on this dataset may be biased towards these languages and may not generalise well to texts in other languages. Multilingual classifiers should ensure consistent performance across the languages of interest.

- **Geographical Bias**: The dataset includes data from various countries. However, because VNRs and VLRs are self-reported documents, some countries have produced more reports than others and are therfore overrepresented while some others are underrepresented in the dataset. This could lead to geographical bias in the models trained on this dataset.

- **Temporal Limitations**: The dataset includes data from reports published between 2016 and 2023. Some earlier reports did not have the right structure to derive SDG labels and were not included in the dataset. As a text corpus, the dataset does not lend itself for predictive modelling to determine future trends or patterns in the SDGs.

- **Labelling Bias**: While the labels in the dataset come from the source documents directly, they may not be entirely bias-free. The biases of the authors of the source documents might be reflected in the content of the section or the labels they assigned to it.

- **Domain Bias**: VNRs and VLRs are formal public documents. Models trained on the data form these sources may not generalise well to other types of documents or contexts.

- **Sociotechnical Risks**: The use of this dataset for decision-making in policy or other areas related to the SDGs should be done with caution, considering all the potential biases and limitations of the dataset. Misinterpretation or misuse of the data could lead to unfair or ineffective decisions.

- **Corrupted texts**: A small fraction of texts in the dataset were not properly extracted from source PDFs and is corrupted. Affected examples will be removed from the dataset in the next version.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users should be made aware of the risks, biases and limitations of the dataset.

Concerning the existence of corrupted texts, users are advised to remove them early on in the processing/training pipeline.
To identify such examples, one can look for a large share of non-alphanumeric or special characters as well as the number of
single character tokens.

## Citation

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

```
@inproceedings{skrynnyk2024sdgi,
  author    = {Mykola Skrynnyk and Gedion Disassa and Andrey Krachkov and Janine DeVera},
  title     = {SDGi Corpus: A Comprehensive Multilingual Dataset for Text Classification by Sustainable Development Goals},
  booktitle = {Proceedings of the 2nd Symposium on NLP for Social Good},
  year      = {2024},
  editor    = {Procheta Sen and Tulika Saha and Danushka Bollegala},
  volume    = {3764},
  series    = {CEUR Workshop Proceedings},
  pages     = {32--42},
  publisher = {CEUR-WS.org},
  series    = {CEUR Workshop Proceedings},
  address   = {Aachen},
  venue     = {Liverpool, United Kingdom},
  issn      = {1613-0073},
  url       = {https://ceur-ws.org/Vol-3764/paper3.pdf},
  eventdate = {2024-04-25},
}
```

**APA:**

Skrynnyk, M., Disassa, G., Krachkov, A., & DeVera, J. (2024). SDGi Corpus: A Comprehensive Multilingual Dataset for Text Classification by Sustainable Development Goals. In P. Sen, T. Saha, & D. Bollegala (Eds.), Proceedings of the 2nd Symposium on NLP for Social Good (Vol. 3764, pp. 32–42). CEUR-WS.org. https://ceur-ws.org/Vol-3764/paper3.pdf

## Glossary

<!-- If relevant, include terms and calculations in this section that can help readers understand the dataset or dataset card. -->

- **SDGs (Sustainable Development Goals)** : A collection of 17 global goals set by the United Nations General Assembly in 2015 for the year 2030. They cover social and economic development issues including poverty, hunger, health, education, climate change, gender equality, water, sanitation, energy, urbanization, environment and social justice.
- **VLR (Voluntary Local Review)**: A process undertaken by local and regional governments to evaluate their progress towards the 2030 Agenda. Note that unlike VNRs, VLRs were not originally envisioned in the 2030 Agenda but emerged as a popular means of communication about SDG localisation.
- **VNR (Voluntary National Review)**: A process undertaken by national governments to evaluate their progress towards the 2030 Agenda.

## More Information

The dataset is a product of the DFx. [Data Futures Platform (DFx)](https://data.undp.org) is an open-source, central hub for data innovation for development impact. 
Guided by UNDP’s thematic focus areas, we use a systems approach and advanced analytics to identify actions to 
accelerate sustainable development around the world.

## Dataset Card Contact

For inquiries regarding data sources, technical assistance, or general information, please feel free to reach out to us at data@undp.org.
