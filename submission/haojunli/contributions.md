## Team Contributions Summary

Qingyang Liu (qliu3)

Chenglin Zhang (chengliz)

Haojun Liu (haojunli)

### Methodology Brainstorming

* Together proposed the method using the few-shot setting. Proposed the baseline model with a default RAG pipeline. Listing the data collection and annotation process. Proposing alternative pipeline component and corresponding models.

### Data Collection

- Chenglin Zhang: Collected ~50 URLs w.r.t. General Info & History, developed the pipeline for data scraping, data processing, data quality checks.
- Qingyang Liu: Collected ~50 URLs for w.r.t. music and future events, data quality checks.
- Haojun Liu: Collected ~50 URLs w.r.t. culture and sports, developed draft scripts for scraping, data quality checks.

### Data Annotation

* Qingyang Liu: Built the QA annotation pipeline, conducted prompt engineering.
* Chenglin Zhang: Built the question type annotion pipeline based on Qingyang's pipeline, prompt engineering, output the QA set.

### RAG Pipeline

* Qingyang Liu: Implemented the Rerank and Hypocritical Document Generator components. Help implement the dafault RAG pipeline and scripts.
* Haojun Liu: Implemented the default RAG pipeline and scripts.

### Experiments

* Qingyang Liu: Conducted experiments for text splitter, the rerank model and the hypo-doc-gen model (related hyperparameters).
* Chenglin Zhang: Conducted experiments for the default pipeline and provided technical support.
* Haojun Liu: Conducted experiments for embedding models, Retriever (related Hyperparameters), and sublink-file cases.

### Graphing

* Qingyang Liu: Generated graphs for retrieval performance.
* Chenglin Zhang: Created visualizations for generation and reranking metrics.
* Haojun Liu: Graphs related to is_future_event attribute.

### Report Writing

* Qingyang Liu: Abstract, data annotation, model details (rerank and HyDE part), results.
* Chenglin Zhang: Overview, data collection, data annotation, model details (hyper-parameter tuning part), results.
* Haojun Liu: Wrote the analysis section and draft of RAG section.
