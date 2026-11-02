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

1. Download the CSV files from https://analyse.kmi.open.ac.uk/open-dataset
2. Put them in `data/raw/`

Data files are excluded by `.gitignore`, in line with the ethics application: no dataset files are uploaded to this public repository.

> Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data, 4*, 170171. https://doi.org/10.1038/sdata.2017.171

## Status

- [x] Topic and dataset agreed with supervisor (OULAD)
- [x] Ethics application revised (modelling methods + dataset citation)
- [ ] D1 proposal: draft in progress (`docs/`)
- [ ] Data exploration and leakage-safe feature pipeline
- [ ] Baseline models and first results (feasibility)
