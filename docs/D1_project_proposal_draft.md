# D1 Project Proposal: Working Draft

**Title:** Development of an AI-Based Student Academic Support System: Early Identification of At-Risk Students Using the Open University Learning Analytics Dataset
**Student:** Darmen Orynbasar (do3000@hw.ac.uk) · **Supervisor:** Inkar Zhumay
**Status:** draft for supervisor feedback · last updated 2 November 2026

> **Notes for the reader.** This is a working draft in Markdown. The final version will be moved into the official Word/LaTeX template and submitted as PDF. Main body limit: 14 pages.
> Sections marked **[TODO]** still need the student's own work (results, Gantt dates, personal reflection).
> References marked ⚠️ in `literature.md` are still being verified.

---

## Front matter (to complete in the template)

- Title page: university template **[TODO]**
- Declaration: university template **[TODO]**
- Table of contents: generated in the template; main body starts at page 1

### Abstract (draft, ~180 words)

Many students who fail or withdraw from a module show warning signs in the first weeks, but staff rarely have the time or tools to spot them early. This project develops an AI-based academic support system whose core is a machine-learning model that identifies students at risk of failing or withdrawing using only the first *N* weeks of activity. Using the publicly available, anonymised Open University Learning Analytics Dataset (OULAD), logistic regression, random forest and XGBoost will be trained to separate Fail/Withdrawn from Pass/Distinction outcomes, and compared against a baseline using F1, ROC-AUC and recall on the at-risk class. A leakage-safe pipeline evaluates several cut-off weeks to measure how early risk can be detected. SHAP explanations show which factors drive each prediction, and performance is compared across demographic groups to check for bias. A prototype support layer turns each prediction into an explained, human-reviewed action: a drafted letter to an advisor and proposed meeting slots. The project contributes an evaluated, explainable and fairness-aware early-warning pipeline linked to a concrete support workflow.

---

## 1. Introduction

### 1.1 Context and motivation

Student failure and withdrawal are costly for students and institutions. In online and blended learning, staff see students less often, so early signs of disengagement (missed logins, late or missing assessments, falling activity) are easy to miss. Virtual learning environments (VLEs) record exactly these signals, which makes data-driven early warning possible (Siemens & Long, 2011; Arnold & Pistilli, 2012). However, many predictive models are evaluated only for accuracy, are hard for educators to interpret, and stop at the prediction without linking it to an action (Viberg et al., 2018; Jin et al., 2024). This project aims to close that gap.

**[TODO: add 2–3 sentences of personal motivation, e.g. why this matters at your campus or from your own experience as a student.]**

### 1.2 Aim

To design, implement and evaluate an AI-based academic support system whose core model identifies students at risk of failing or withdrawing from the first *N* weeks of activity, and whose support layer turns each prediction into an explained, human-reviewed action (advisor letter and meeting proposal).

### 1.3 Research questions

- **RQ1.** How accurately can logistic regression, random forest and XGBoost identify at-risk students (Fail/Withdrawn vs Pass/Distinction) from VLE activity, assessment and registration data available in the first *N* weeks of a module presentation?
- **RQ2.** How early can at-risk students be identified reliably, i.e. how do recall, F1 and AUC on the at-risk class change as *N* increases?
- **RQ3.** Which features drive the predictions, and do the models perform consistently across demographic groups (gender, age band, disability, deprivation band)?
- **RQ4.** Can model outputs and explanations be translated into a support workflow that meets defined functional requirements and works on realistic scenarios?

### 1.4 Objectives

| ID | Objective | Linked RQ |
|---|---|---|
| O1 | Review literature on AI in academic support, early-warning systems, OULAD-based prediction, explainability, fairness and ethics | all |
| O2 | Prepare OULAD: clean data, define the at-risk label, build leakage-safe features for several cut-off weeks | RQ1, RQ2 |
| O3 | Train and tune logistic regression, random forest and XGBoost against a majority-class baseline | RQ1 |
| O4 | Evaluate with recall, precision, F1, ROC-AUC and PR-AUC on the at-risk class (preliminary targets: recall ≥ 0.75, ROC-AUC ≥ 0.80 by week 8–12; revised after baseline results) | RQ1, RQ2 |
| O5 | Explain predictions with SHAP and compare performance across demographic groups | RQ3 |
| O6 | Build a support-layer prototype (risk list, explanation, letter, meeting proposal) and test it on OULAD-derived scenarios | RQ4 |
| O7 | Reflect on professional, legal, ethical and social issues and on limitations | all |

### 1.5 Problems to solve

1. **Label and timing:** defining "at risk" so that only information available before the cut-off week is used.
2. **Data leakage:** preventing future data, or the same student, from leaking between training and test sets.
3. **Class imbalance:** at-risk students may be a minority in some modules and cut-offs, so accuracy is misleading.
4. **Interpretability:** making predictions understandable to staff.
5. **Fairness:** checking that the model does not perform worse for particular groups.
6. **From prediction to action:** designing a workflow where humans stay in control.

### 1.6 Organisation of the document

Section 2 reviews related work and identifies the research gap. Section 3 describes the methodology, requirements and evaluation plan. Section 4 presents preliminary work showing feasibility. Section 5 concludes. Project management (plan, risks), PLES and the GenAI summary follow the references.

---

## 2. Background Research

### 2.1 AI in academic support: evidence and limits

Artificial intelligence is widely applied in higher education. In a systematic review of 146 articles, Zawacki-Richter et al. (2019) grouped applications into profiling and prediction, assessment and evaluation, adaptive systems and personalisation, and intelligent tutoring systems, and noted that studies rarely involved educators or reflected critically on ethics.

The strongest evidence of effectiveness concerns intelligent tutoring systems (ITS). A meta-analysis of 107 effect sizes found that ITS outperformed teacher-led large-group instruction, non-ITS computer-based instruction and textbooks, but did not differ significantly from individual human tutoring (Ma et al., 2014). For college students, ITS had a moderate positive effect and were less effective than human tutoring (Steenbergen-Hu & Cooper, 2014); VanLehn (2011) similarly found that step-based tutoring can approach human tutors. These findings suggest AI support helps but does not replace people, which motivates a design where the system flags students and hands over to staff.

### 2.2 Learning analytics and early-warning systems

Learning analytics and educational data mining use learner data to understand and improve learning (Baker & Yacef, 2009; Siemens & Long, 2011). Course Signals at Purdue is a well-known early-alert system that flagged risk and supported communication with students (Arnold & Pistilli, 2012). More recent systems automate the workflow: LANSE trains models on learning-management-system data and gives teachers weekly dropout and failure predictions with visualisations, while reporting challenges in privacy compliance, data management and security (Cechinel et al., 2024). Hlosta et al. (2017) address early identification in new modules without historical data. However, Viberg et al. (2018) found limited evidence that learning analytics improves learning outcomes, suggesting that how a prediction leads to support matters as much as its accuracy.

### 2.3 Machine learning for predicting performance, at-risk status and dropout

Albreiki et al. (2021) reviewed educational data mining studies from 2009 to 2021 on identifying dropouts and at-risk students and found most use institutional databases or online-platform data. Sekeroglu et al. (2021) highlighted wide variation in models, datasets, validation and evaluation, and called for standardised evaluation and validation. Logistic regression and tree ensembles such as random forest and gradient boosting are common for tabular behavioural data (Jin et al., 2024).

Two methodological issues recur. First, class balance: when at-risk students are a minority, accuracy is misleading and recall, F1 and precision–recall analysis are more informative (Saito & Rehmsmeier, 2015). Second, data leakage: using information from after the prediction point, or the same student in training and test sets, inflates results.

### 2.4 Predictive modelling with OULAD

OULAD contains anonymised demographic, assessment and daily VLE click data for 32,593 students across 22 module presentations, released under CC BY 4.0 (Kuzilek et al., 2017). A systematic review of 17 OULAD studies (2017–2024) grouped them into predicting performance, identifying at-risk students and predicting engagement, with frequent use of decision trees, random forest, SVM, logistic regression, gradient boosting and k-nearest neighbours, and growing use of deep learning (Jin et al., 2024). Examples include deep learning on VLE clicks to predict withdrawal (Hassan et al., 2019) and deep neural networks for performance prediction (Waheed et al., 2020). Jin et al. (2024) call for more attention to interpretability and to linking models with educational theory.

### 2.5 Interpretability, fairness and ethics

SHAP assigns each feature an additive contribution to a prediction (Lundberg & Lee, 2017), supporting both global analysis and student-level explanations. Fairness matters because models may perform differently across groups; equality of opportunity requires similar true-positive rates across groups (Hardt et al., 2016). Cabral et al. (2024) compare traditional machine-learning methods in terms of fairness for student performance prediction in CS1 courses. Prinsloo and Slade (2017) argue that ethics and privacy enable rather than obstruct learning analytics, and Rubel and Jones (2016) raise concerns about intrusive data use and student autonomy. This project uses only public, anonymised data but treats these concerns as design requirements.

### 2.6 Research gap and contribution

OULAD studies concentrate mostly on predictive accuracy and algorithm comparison. Fewer studies combine, in one design: (i) how early risk can be detected, (ii) explanations educators can read, (iii) performance consistency across demographic groups, and (iv) a concrete, human-in-the-loop action. This project addresses all four.

**[TODO: after reading the primary OULAD studies, add one comparison table: study, models, features, cut-off, best metric.]**

---

## 3. Methodology / Work Plan

### 3.1 Research approach

A quantitative, experimental study (RQ1–RQ3) using secondary analysis of OULAD only, plus a design-and-demonstration component (RQ4) evaluated by requirements and scenario testing. No human participants are involved, consistent with the ethics application.

### 3.2 Data

OULAD is downloaded from the official source (Kuzilek et al., 2017). Tables: `courses`, `assessments`, `vle`, `studentInfo`, `studentRegistration`, `studentAssessment`, `studentVle`. The outcome `final_result` takes Pass, Distinction, Fail or Withdrawn. Daily clicks are stored relative to module start, so features can be limited to the first *N* weeks.

### 3.3 Problem definition and preprocessing

- **Target:** at-risk = 1 for Fail/Withdrawn; 0 for Pass/Distinction.
- **Cut-offs:** *N* ∈ {4, 8, 12, 16} weeks; only data up to the cut-off is used.
- **Exclusions:** students who unregistered before the cut-off are excluded at that cut-off (their outcome is already known); counts are reported.
- **Features:** (a) VLE engagement: total clicks, active days, clicks per activity type, days since last activity, weekly trend; (b) assessment: submissions, mean score, lateness, missed assessments; (c) registration: registration timing, previous attempts, studied credits; (d) demographics: gender, age band, disability, region, highest education, deprivation band.
- **Two feature sets:** Experiment A uses (a)–(c); Experiment B adds (d), to measure the contribution of demographics and support the fairness analysis.
- **Missing values:** imputed with statistics learned from training data only.
- **Class balance:** checked per module and cut-off; class weighting if needed; SMOTE (Chawla et al., 2002) tested only inside training folds.

### 3.4 Models and training

Majority-class baseline; logistic regression (regularised, standardised inputs); random forest (Breiman, 2001); XGBoost (Chen & Guestrin, 2016). Hyperparameters tuned by randomised search within cross-validation. Python with pandas, scikit-learn (Pedregosa et al., 2011), XGBoost and SHAP; code under version control on GitHub (data excluded).

**Leakage control:** group-aware splits by student; all preprocessing fitted inside training folds; an additional out-of-time check trains on 2013 presentations and tests on 2014 presentations, removing test students who also appear in training.

### 3.5 Evaluation

- **Primary:** recall on the at-risk class, plus precision and F1.
- **Ranking:** ROC-AUC and PR-AUC; confusion matrices at the chosen threshold.
- **Threshold:** chosen on validation data to reach a target recall; resulting precision (alert volume) reported.
- **Uncertainty:** mean ± SD across folds; bootstrap 95% confidence intervals on test results.
- **Earliness (RQ2):** metrics plotted against *N*.
- **Explainability (RQ3):** global SHAP importance, local explanations, stability of top features.
- **Fairness (RQ3):** recall, false-positive rate and precision by gender, age band, disability and deprivation band; true-positive-rate gap (Hardt et al., 2016); Experiment A vs B.

### 3.6 Support-layer prototype

For each flagged student the prototype shows: (1) risk level at the current week; (2) top three factors in plain language from SHAP; (3) an editable advisor letter generated from a template; (4) proposed meeting slots from a mock calendar. Pseudonymised OULAD IDs only; a human approves every action; nothing is sent automatically. Python backend (e.g. FastAPI) and simple web interface (e.g. Streamlit).

**Use cases:** UC1 view at-risk list for a module and week · UC2 inspect a student's explanation · UC3 generate and edit an escalation letter · UC4 select a meeting slot · UC5 record the outcome (simulated).

**[TODO: add a use-case diagram and an architecture diagram (draw.io).]**

### 3.7 Requirements (MoSCoW)

| Priority | Requirements |
|---|---|
| **Must** | M1 leakage-safe feature pipeline per cut-off · M2 train and compare LR, RF, XGBoost vs baseline · M3 report recall, precision, F1, ROC-AUC, PR-AUC per cut-off · M4 global and individual SHAP explanations · M5 performance comparison across demographic groups · M6 prototype with risk list and explanations |
| **Should** | S1 editable advisor letter from template · S2 meeting slots from mock calendar · S3 threshold tool (recall vs alert volume) · S4 out-of-time validation (2013 → 2014) |
| **Could** | C1 LLM rephrasing of letters with fictional text only and human review · C2 weekly risk-trend chart · C3 fairness mitigation (e.g. reweighing) |
| **Won't** | W1 data collection from real students or staff · W2 automatic messages or decisions · W3 integration with live university systems |

### 3.8 Prototype evaluation

Requirements coverage (traceability to MoSCoW), 10–20 scenario-based functional tests built from OULAD cases (e.g. early withdrawal pattern, late engagement pattern), unit tests for the data pipeline, and response-time checks. User studies are out of scope (they would need an ethics amendment) and are listed as future work.

---

## 4. Feasibility

**[TODO: this section must show real preliminary work. Fill it after the first experiments; do not invent numbers.]**

### 4.1 Dataset access and exploration (planned evidence)

- OULAD downloaded from the official page; licence (CC BY 4.0) confirmed.
- Table sizes and join keys checked: **[TODO: rows per table]**
- Outcome distribution by module: **[TODO: % Pass / Distinction / Fail / Withdrawn]**
- Share of at-risk students per cut-off week after exclusions: **[TODO]**

### 4.2 Baseline experiment (planned evidence)

- One cut-off (e.g. *N* = 8), Experiment A features, logistic regression vs majority baseline with group-aware 5-fold CV.
- Report recall, F1 and ROC-AUC on the at-risk class: **[TODO]**
- One SHAP summary plot for the baseline: **[TODO: figure]**

### 4.3 Tools evaluated

| Need | Option chosen | Alternatives considered | Reason |
|---|---|---|---|
| Data processing | pandas | Polars | Familiar, sufficient for OULAD size **[TODO: confirm memory use on studentVle]** |
| Models | scikit-learn + XGBoost | LightGBM, PyTorch | Supervisor-recommended models; standard, well documented |
| Explanations | SHAP | LIME | Additive, consistent; works with tree models |
| Prototype UI | Streamlit | Flask + HTML | Fast to build; enough for a demonstration |
| Version control | GitHub (code only) | — | Data excluded per ethics approval |

### 4.4 Design evidence

**[TODO: first architecture sketch and one prototype screen mock-up.]**

### 4.5 Ethics

Ethics application submitted for secondary analysis of public, anonymised data, with dataset citation in the Methods section, as requested by the supervisor. **[TODO: add approval status/reference number when available.]**

---

## 5. Conclusions

This proposal sets out a focused project: an early-warning model for at-risk students built on OULAD, evaluated rigorously for accuracy, earliness, explainability and fairness, and connected to a human-in-the-loop support workflow. The methods follow directly from the research questions, the data is public and ethically approved, and preliminary work (Section 4) will show that the pipeline is feasible within the second semester.

---

## References

Albreiki, B., Zaki, N., & Alashwal, H. (2021). A systematic literature review of student' performance prediction using machine learning techniques. *Education Sciences, 11*(9), 552. https://doi.org/10.3390/educsci11090552

Arnold, K. E., & Pistilli, M. D. (2012). Course Signals at Purdue: Using learning analytics to increase student success. In *Proceedings of the 2nd International Conference on Learning Analytics and Knowledge* (pp. 267–270). ACM.

Baker, R. S., & Yacef, K. (2009). The state of educational data mining in 2009: A review and future visions. *Journal of Educational Data Mining, 1*(1), 3–17.

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.

Cabral, L. de S., Pereira, F. D., & Ferreira Mello, R. (2024). Enhancing algorithmic fairness in student performance prediction through unbiased and equitable machine learning models. In A. M. Olney et al. (Eds.), *Artificial Intelligence in Education: AIED 2024 Workshops* (CCIS 2150, pp. 418–426). Springer. https://doi.org/10.1007/978-3-031-64315-6_39

Cechinel, C., Queiroga, E. M., Primo, T. T., Lemos dos Santos, H., Ramos, V. F. C., Munoz, R., Ferreira Mello, R., & Machado, M. F. B. (2024). LANSE: A cloud-powered learning analytics platform for the automated identification of students at risk in learning management systems. In A. M. Olney et al. (Eds.), *Artificial Intelligence in Education: AIED 2024 Workshops* (CCIS 2150, pp. 127–138). Springer. https://doi.org/10.1007/978-3-031-64315-6_10

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*, 321–357.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). ACM.

Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. In *Advances in Neural Information Processing Systems, 29* (pp. 3315–3323).

Hassan, S.-U., Waheed, H., Aljohani, N. R., Ali, M., Ventura, S., & Herrera, F. (2019). Virtual learning environment to predict withdrawal by leveraging deep learning. *International Journal of Intelligent Systems, 34*(8), 1935–1952.

Hlosta, M., Zdrahal, Z., & Zendulka, J. (2017). Ouroboros: Early identification of at-risk students without models based on legacy data. In *Proceedings of the 7th International Learning Analytics & Knowledge Conference* (pp. 6–15). ACM.

Jin, L., Wang, Y., Song, H., & So, H.-J. (2024). Predictive modelling with the Open University Learning Analytics Dataset (OULAD): A systematic literature review. In A. M. Olney et al. (Eds.), *Artificial Intelligence in Education: AIED 2024 Workshops* (CCIS 2150, pp. 477–484). Springer. https://doi.org/10.1007/978-3-031-64315-6_46

Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data, 4*, 170171. https://doi.org/10.1038/sdata.2017.171

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems, 30* (pp. 4765–4774).

Ma, W., Adesope, O. O., Nesbit, J. C., & Liu, Q. (2014). Intelligent tutoring systems and learning outcomes: A meta-analysis. *Journal of Educational Psychology, 106*(4), 901–918.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Prinsloo, P., & Slade, S. (2017). Ethics and learning analytics: Charting the (un)charted. In C. Lang, G. Siemens, A. Wise, & D. Gašević (Eds.), *Handbook of learning analytics* (pp. 49–57). SoLAR. https://doi.org/10.18608/hla17.004

Rubel, A., & Jones, K. M. L. (2016). Student privacy in learning analytics: An information ethics perspective. *The Information Society, 32*(2), 143–159.

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE, 10*(3), e0118432.

Sekeroglu, B., Abiyev, R., Ilhan, A., Arslan, M., & Idoko, J. B. (2021). Systematic literature review on machine learning and student performance prediction: Critical gaps and possible remedies. *Applied Sciences, 11*(22), 10907. https://doi.org/10.3390/app112210907

Siemens, G., & Long, P. (2011). Penetrating the fog: Analytics in learning and education. *EDUCAUSE Review, 46*(5), 30–40.

Steenbergen-Hu, S., & Cooper, H. (2014). A meta-analysis of the effectiveness of intelligent tutoring systems on college students' academic learning. *Journal of Educational Psychology, 106*(2), 331–347.

VanLehn, K. (2011). The relative effectiveness of human tutoring, intelligent tutoring systems, and other tutoring systems. *Educational Psychologist, 46*(4), 197–221.

Viberg, O., Hatakka, M., Bälter, O., & Mavroudi, A. (2018). The current landscape of learning analytics in higher education. *Computers in Human Behavior, 89*, 98–110.

Waheed, H., Hassan, S.-U., Aljohani, N. R., Hardman, J., Alelyani, S., & Nawaz, R. (2020). Predicting academic performance of students from VLE big data using deep learning models. *Computers in Human Behavior, 104*, 106189.

Zawacki-Richter, O., Marín, V. I., Bond, M., & Gouverneur, F. (2019). Systematic review of research on artificial intelligence applications in higher education – where are the educators? *International Journal of Educational Technology in Higher Education, 16*, 39. https://doi.org/10.1186/s41239-019-0171-0

---

## Project Management

### Plan

**[TODO: replace with a Gantt chart figure with real dates.]**

| Period | Activities | Milestone |
|---|---|---|
| Semester 1 (done) | Topic selection, literature search, supervisor meetings, dataset choice (OULAD), ethics application and revision | Ethics form submitted; D1 draft |
| Semester 1 (remaining) | Data exploration, first baseline (Feasibility), D1 final | D1 submitted |
| Semester 2, phase 1 | Label/feature pipeline for all cut-offs, leakage checks | Feature pipeline |
| Semester 2, phase 2 | Model training, tuning, threshold selection, out-of-time validation | Model comparison |
| Semester 2, phase 3 | SHAP and fairness analysis | Interpretability & fairness results |
| Semester 2, phase 4 | Support-layer prototype and scenario tests | Working prototype |
| Semester 2, phase 5 | Dissertation writing, demo preparation | Final submission and presentation |

### Risk analysis

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Data leakage inflates results | Medium | High | Group-aware splits, cut-off-only features, out-of-time test |
| Models underperform targets | Medium | Medium | Targets are preliminary; report honestly; add feature engineering; analyse errors |
| Class imbalance distorts metrics | Medium | Medium | Recall/F1/PR-AUC, class weights, SMOTE inside folds |
| `studentVle` too large for laptop memory | Medium | Medium | Aggregate in chunks; use efficient dtypes; Parquet |
| Biased model performance across groups | Medium | High | Fairness analysis, Experiment A vs B, report limitations |
| Scope creep (prototype, LLM features) | High | Medium | MoSCoW; prototype only after Musts are done |
| Time pressure from other courses | High | Medium | Weekly plan, fixed supervisor meetings, buffer weeks |
| Loss of code or work | Low | High | GitHub, regular commits, local backup |
| Ethics scope change (e.g. user study) | Low | High | Keep user studies as future work or apply for amendment first |

### PLES (Professional, Legal, Ethical and Social issues)

- **Professional:** follow the BCS Code of Conduct (public interest, professional competence and integrity, duty to relevant authority, duty to the profession); report results honestly, including limitations and negative results; reproducible code.
- **Legal:** OULAD is anonymised and licensed under CC BY 4.0, which requires attribution (Kuzilek et al., 2017); dataset files are not redistributed in the public repository. Data protection principles (UK GDPR / Data Protection Act 2018) are respected through data minimisation and no re-identification. Open-source libraries are used under their licences. **[TODO: mention any relevant Kazakhstan data-protection law if required by the campus.]**
- **Ethical:** risk labels can stigmatise students, so predictions are framed as decision support, not decisions; humans approve every action; fairness is checked across demographic groups; explanations are provided for every flag. Ethics approval obtained from Heriot-Watt for secondary analysis.
- **Social:** early warning could improve access to support, but over-reliance on AI and surveillance concerns must be avoided; results from a UK distance-learning university may not generalise to other contexts.

### GenAI Summary

**[TODO: complete honestly in your own words. Draft structure:]**

- **Tools used:** ChatGPT and Claude (Anthropic).
- **Purposes:** finding candidate literature; suggesting a project scope and structure; drafting and restructuring sections of this proposal; checking dataset licence and citation details; preparing repository files.
- **Verification:** every reference is checked against the original source before citation (status tracked in `literature.md`); all text is reviewed, rewritten and owned by the student; no GenAI tool was given any dataset files.
- **Not used for:** generating results or data; final judgements about research decisions without supervisor discussion.

### Appendices

**[TODO: e.g. ethics form summary, additional EDA figures, prototype screenshots.]**
