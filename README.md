This repository contains an end-to-end experiment for research on multilingual truthfulness of LLMs as part of a MSc thesis at the Pattern Recognition and Bioinformatics (PRB) dpt. at TU Delft. The research focuses on both practical, quantitative truthfulness for many of the most-spoken languages in the world, as well as on the qualitative relationships between languages and their truthfulness, for better understanding of LLM multilinguality. We mostly aim to answer the question "How truthful are Large Language Models in non-English languages?", supplemented by three additional questions: 1) "What factors correlate strongest to truthfulness in non-English languages?", 2) "Do the findings respect the notion that Large Language Models greatly rely on translation to English for non-English tasks?" and 3) "Are the findings generalizable over other alignment criteria?" We obtain these numerical results with the experiments in this repository by querying a diverse range of models (GPT-4o, Gemini, Bloomz, mT5, Llama- 2-7B, -13B, -70B) in a range of languages, much-spoken but diverse in several aspects such as script, family, similarity to English and abundance in web-data. We distinguish between internal knowledge (benchmarked by a 1000-sample subset of Google’s ’Natural Questions’ dataset translated to respective langauges) and external knowledge (benchmarked by a 1000-sample random subset, 250 each, of the scientific datasets Climate-FEVER, SciFact, COVID-Fact and HealthVER). Results are regressed with language similarity to English (lang2vec Python library, cosine similarity) and web-data abundance (using Common Crawl fractions as estimates), and results are softly observed in light of the other, categorical factors. We also relate GPT-4o translation quality to observed truthfulness and potentially ablate with another criterion to further answer the posed sub-questions

# Overview of target languages
```
target_languages = [
    'arb', 'fra', 'spa', 'hin',
    'zho', 'eng', 'cym', 'fin',
    'hun', 'zul', 'nld', 'ita',
    'vie', 'swh', 'jpn', 'deu',
    'ind', 'urd', 'rus', 'por',
    'ben'
]
```

# Folder Structure
.
├── benchmarks/
│   ├── clean/
│   │   └── // pre-processed benchmarks that fit a universal .json formatting
│   ├── results/
│   │   └── // acquired results in .json, formatted as '{benchmark}_{model}_{language}.json'
│   └── translated/
│       └── // translated benchmarks, formatted as '{benchmark}_{language}.json' (e.g. 'NQ_it')
├── features/
│   └── # TO ADD, POST-ANALYSIS
├── models/
│   └── // Place to load local models (may be temporary)
├── config.json
├── evaluate.py
├── experiment.py
├── model.py
├── preprocess.py
├── query.py
├── translation.py
├── utils.py
└── README.md

# Repository lay-out
The repository contains all used benchmarks (preprocessed) and functions to . The main experiment function is ```experiment.py```, which runs experiments based on configurations in ```config.json```. All translation, querying, evaluation, model loading will be automated.

# TODO: Add config.json overview of options

# Available experiments
| Metric       | GPT-4o | Gemini | Bloomz | mT0 | Llama2-7B | Llama2-13B | Llama2-70B |
|--------------|--------|--------|--------|-----|------------|-------------|-------------|
| External knowledge (scientific)  |        |        |        |     |            |             |             |
| Internal knowledge (QA)     |   ✔      |        |   ✔      |     |            |             |             |
