# LLM Evaluation Tools Comparison:

This repository provides the experimental evidence, datasets, and implementation scripts used for the paper:

**"An LLM Evaluation Framework Applied to Q&A and Translation Tasks of User Manuals in Consumer Electronics"**

![Pipeline Overview](llm-eval-tools-comparison/results_paper/bigpicture_methodology.png)

## Contents

- **/data**: Evaluation datasets in `.json` format for Q&A and translation tasks. The raw .pdf files used are available as well.

- **/experiments**: Raw results and exports from each evaluation tool:
  - `ragas/`: Metrics by task/model from the RAGAS pipeline.
  - `galileo/`: Self-hosted and online evaluation from Galileo platform.
  - `deepeval/`: Answer relevancy metrics results from DeepEval.
  - `trulens/`: 
  
- **/notebooks**: Interactive notebooks used to run and analyze evaluation tools.

- **/scripts**: Auxiliary Python scripts and environment setup files.

- **/results**: Final formatted tables, radar charts, and plots used in the paper.

---

## Reproducing Experiments

### 1. Environment Setup

```bash
conda env create -f environment.yml
conda activate llm_eval_env
```

### 2. Model Setup and Prompt Templates

The responses from different LLMs were obtained using a custom script in models_setup.py
, which supports both OpenAI API (e.g., GPT-3.5-turbo) and local models (e.g., LLaMA, Gemma) via Ollama.

Prompt templates were defined for each task:

```python
PROMPT_TEMPLATES = {
    "qna": (
        "Considering the {section} and page(s) {pages} of the user manual of "
        "{market_name} {product_type} by {manufacture}, answer the following question:\n"
        "{question}"
    ),
    "translation": (
        "Translate the following {source_language} sentence into {target_language}:\n"
        "{source}"
    )
}
```

Prompt formatting and inference functions are defined as follows:

- `generate_response_openai(prompt, model)` – executes completion via OpenAI API.

- `generate_response_ollama(model_name, prompt)` – sends request to local Ollama instance.

### 3. Running Inference on All Samples

Use the notebook notebooks/running_task.ipynb to iterate through tasks and models. Each sample from the dataset is loaded from .json, the prompt is dynamically generated, and responses are saved for further evaluation.

Supported tasks: `qna`, `translation`
Supported models: `gpt-3.5-turbo`, `llama2:7b`, `gemma:2b`, `mistral`

```python
MODELS = ["gpt-3.5-turbo", "llama2:7b", "gemma:2b", "mistral"]
TASKS_TO_RUN = ["qna", "translation"]
```

The inference pipeline performs the following steps:

1. Load samples from a custom JSON file.

2. Build the prompt based on metadata.

3. Call the appropriate model depending on the model name.

4. Append results incrementally to:

- outputs_run.json (one record per line)

- outputs_run.csv (tabular format)

Sample output record:
```json
{
  "timestamp": "2025-09-16T12:45:00Z",
  "task": "qna",
  "id": "q1234",
  "model": "mistral",
  "prompt": "...",
  "reference": "This device has a 5000mAh battery.",
  "output": "The smartphone includes a 5000 milliamp-hour battery.",
  "error": null
}
```

### 4. Datasets and Prompt Metadata
Evaluation datasets are available in:

```bash
data/qna_dataset_with_metadata.json
data/translation_dataset_with_metadata.json
```

Each JSON object includes:

- input fields (question, text_source)

- reference answer

- metadata used to construct prompts (manual section, pages, product model, etc.)

```json
{
  "task_type": "qna",
  "input": "What is the battery capacity of this device?",
  "reference": "The device has a 5000mAh battery.",
  "prediction": "This phone includes a 5000 milliamp-hour battery."
}
```

### 5. Evaluation Setup: Open notebooks in /notebooks for each tool:

- RAGAS: mu_ragas_eval_setup.ipynb

- TruLens: TruLens_eval_setup_mu.ipynb

- DeepEval: deepeval_quickstart.ipynb

- Galileo: galileo_eval_trial.ipynb

### Metrics included

| Tool     | Metrics                                                                                  |
| -------- | ---------------------------------------------------------------------------------------- |
| RAGAS    | BLEU, answer\_similarity, answer\_correctness, helpfulness, coherence, adequacy, fluency |
| DeepEval | answer\_relevancy                                                                        |
| TruLens  | answer\_relevancy                                                                        |
| Galileo  | ROUGE, BLEU, Answer Relevance (manual)                                                   |


## References

Below are the official repositories and documentation pages for the evaluation tools and libraries used in this project:

### Evaluation Tools

- RAGAS (Retrieval-Augmented Generation Assessment)

    - GitHub: https://github.com/explodinggradients/ragas

    - Docs: https://docs.ragas.io

- TruLens

    - GitHub: https://github.com/truera/trulens

    - Docs: https://www.trulens.org

- DeepEval

    - GitHub: https://github.com/confident-ai/deepeval

    - Docs: https://deepeval.com/docs/getting-started

- Galileo

    - Website: https://galileo.ai/

    - Docs: https://v2docs.galileo.ai/what-is-galileo?_gl=1*15052v4*_gcl_au*MTEyOTYxMzUzNy4xNzU2MzYyMzk2 

    - Terms of Service (data usage, limitations): https://galileo.ai/terms-of-service | https://galileo.ai/privacy-policy

## Citation
Please cite this work as: