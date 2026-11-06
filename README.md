# Development of an AI-Based Student Academic Support System

Honours project, Heriot-Watt University (MACS, Aktobe campus)
Student: Darmen Orynbasar · Supervisor: Inkar Zhumay

## Project in one paragraph

The project builds and evaluates machine-learning models that identify students at risk of failing or withdrawing (Fail/Withdrawn vs Pass/Distinction) from the first *N* weeks of activity in the Open University Learning Analytics Dataset (OULAD). Logistic regression, random forest and XGBoost are compared using F1, ROC-AUC and recall on the at-risk class, with SHAP explanations and a fairness check across demographic groups. A prototype support layer turns each prediction into a human-reviewed advisor letter and meeting proposal.

## Repository contents

| Path | Contents |
|---|---|
| `docs/D1_Project_Proposal_Darmen_Orynbasar.docx` | Deliverable 1: project proposal in the official HWU template (draft) |
| `docs/D1_project_proposal_draft.md` | Same proposal as plain text, for quick reading on GitHub |
| `docs/gantt.png`, `docs/architecture.png` | Figures used in the proposal |
| `literature.md` | Reading list with APA references and verification status |
| `data/` | Local only, **not committed** (see below) |

## Data

The project uses only OULAD (Kuzilek, Hlosta & Zdrahal, 2017), a public, anonymised dataset under CC BY 4.0.

1. Download the ZIP (44.6 MB) from the authors' official Figshare record: https://figshare.com/articles/OULAD_Open_University_Learning_Analytics_Dataset/5081998 (the old analyse.kmi.open.ac.uk page no longer works)
2. Put them in `data/raw/`

Data files are excluded by `.gitignore`, in line with the ethics application: no dataset files are uploaded to this public repository.

> Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data, 4*, 170171. https://doi.org/10.1038/sdata.2017.171

## Running the pipeline

```bash
pip install pandas pyarrow scikit-learn xgboost shap matplotlib streamlit ollama pytest
python src/features.py       # leakage-safe features for N = 4, 8, 12, 16 weeks -> data/processed/
python src/train.py          # LR / RF / XGBoost vs baseline -> results/, models/
python src/evaluate.py       # threshold for recall 0.75 + fairness by group -> results/
python src/build_support.py  # risk scores + SHAP factors for unseen 2014 students
```

## Running the advisor chatbot

Needs [Ollama](https://ollama.com) with a local Qwen model. No data leaves the machine.

```bash
ollama pull qwen3:4b-instruct
streamlit run src/app.py --server.address localhost
```

The chatbot (`src/chatbot.py`) lets Qwen choose tools from `src/support.py` and phrase their results. Every number comes from the tools. It has no tool that approves or sends anything: a human saves each decision in the app, and that only writes a local simulated log.

Tests: `pytest tests/test_support.py`, and `python tests/chatbot_scenarios.py` for the chatbot scenarios (needs Ollama).

`features.py` looks for the CSVs in `data/raw/`, then in `../anonymisedData/`.

## Status

- [x] Topic and dataset agreed with supervisor (OULAD)
- [x] Ethics application revised (modelling methods + dataset citation)
- [ ] D1 proposal: draft in progress (`docs/`)
- [ ] Data exploration
- [x] Leakage-safe feature pipeline (`src/features.py`)
- [x] Baseline models and first results (`src/train.py`, `results/`)
- [x] Threshold selection and fairness analysis (`src/evaluate.py`)
- [ ] Hyperparameter tuning, SHAP stability, support-layer prototype
