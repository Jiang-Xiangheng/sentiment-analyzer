# Sentiment Analyzer

[![CI](https://img.shields.io/github/actions/workflow/status/Jiang-Xiangheng/sentiment-analyzer/ci.yml?branch=main&label=CI&style=for-the-badge&logo=github)](https://github.com/Jiang-Xiangheng/sentiment-analyzer/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs-526CFE?style=for-the-badge&logo=materialformkdocs&logoColor=white)](https://Jiang-Xiangheng.github.io/sentiment-analyzer/)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Jiang-Xiangheng/sentiment-analyzer/badges/coverage.json&style=for-the-badge&logo=pytest&logoColor=white)](https://github.com/Jiang-Xiangheng/sentiment-analyzer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![HF Hub](https://img.shields.io/badge/🤗%20HF%20Hub-model-FFD21E?style=for-the-badge)](https://huggingface.co/Jiang-Xiangheng/sentiment-analyzer)

A production-grade ML project.

**Russian version:** [README.ru.md](README.ru.md)

## Task

Task type: `nlp` · Framework: `pytorch`.

## Dataset

Document dataset source, size, splits. Link to Kaggle / HF dataset page.

## Results

Fill in after training. Include metrics table with main model vs baseline.

| Model | Metric 1 | Metric 2 |
|---|---|---|
| Main | - | - |
| Baseline | - | - |

## Quick Start

```bash
uv sync --all-groups
make data
make train
make evaluate
make serve
docker compose up
```

## Project Structure

```
src/sentiment_analyzer/
├── data/
├── models/
├── training/
├── evaluation/
├── inference/
├── serving/
└── utils/
```

## Acknowledgements

This project was bootstrapped from the **ml-project-template** by Daniil Kiselyov
([kiselyovd](https://github.com/kiselyovd)), which is distributed under the MIT License.

- Original repository: <https://github.com/kiselyovd/ml-project-template>
- Original author: Daniil Kiselyov (kiselyovd)
- Original license: MIT

The original MIT copyright notice is retained in [LICENSE](LICENSE).

## License

MIT - see [LICENSE](LICENSE).
