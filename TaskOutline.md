# CITS4012 Natural Language Processing
## Group Project
**September 2026**

---

## Overview

In this project, you will design, implement, and evaluate models for a Question Answering task using one of the provided datasets.

The central requirement is to design and train your own neural model. Your model may use RNN/LSTM/GRU/Transformer structures, or a combination of these, and must incorporate an attention mechanism. You should justify your architectural design in relation to the characteristics and challenges of your selected dataset.

Your investigation should include quantitative evaluation using appropriate, justified metrics; controlled ablation studies examining the contribution of selected model components; and qualitative analysis of model predictions and attention patterns. These analyses should help you evaluate your design decisions, identify limitations, and explain the differences observed between models.

Your final submission will include a reproducible implementation and a report presenting your methodology, experimental design, results, and critical analysis.

This project is to be completed in a group of maximum 3 students. You're allowed to complete this project individually if you strongly prefer not to work in a group, but please note **NO bonus mark** will be given for individual submission.

* **Weighting:** This assignment is worth 30% of your overall unit mark.
* **Deadline:** 18th October 2026, 11:59PM AWST

---

## Datasets

You will select **ONE** of the following four datasets:

* **Tweet QA:** Focuses on answering questions about tweets, introducing challenges associated with informal language and social media content.
* **MultiRC:** Involves understanding and combining information across multiple sentences.
* **ReClor:** Emphasises logical reasoning in reading comprehension.
* **PIQA:** Focuses on physical commonsense in everyday situations.

These datasets differ in their input structure and answer format. You are encouraged to research your chosen dataset by consulting its original publication and documentation. The datasets can be found in [this shared folder](https://drive.google.com/drive/folders/1j5lP1gBrewzpFSH5Ya1SQ6echsiJaI3j?usp=drive_link).

---

## Report Writing

### Required Format

* You **MUST** write your report in $\mathrm{\LaTeX}$ using [the official ACL template](https://github.com/acl-org/acl-style-files).
* Include only your group number beneath the title using the `\author` field, for example, `\author{Group 12}`. Do not include student names in this field.
* To display your group number, use the template's final version by replacing `\usepackage[review]{acl}` with `\usepackage[final]{acl}`.
* Submit your report as a **PDF containing no more than six pages of content**, including figures and tables but excluding the team contribution statement and references. Select and organise your content carefully to present a coherent and concise account of your work.
* Reports that exceed the page limit or otherwise fail to comply with the required format will not be accepted.

### Suggested Structure

The following structure is recommended for your report:

#### Title
Provide a descriptive title for your project and display your group number beneath it.

#### Introduction
Introduce your chosen question answering task and explain its significance, key challenges, and potential applications. Clearly state the aims of your project, outline your modelling approach and experimental investigation, and briefly summarise your main findings. Provide sufficient context to explain the motivation for your work, supporting your discussion with references to relevant existing research.

#### Methodology
This section should describe and justify the design of your main neural model for the selected question answering task. You must design, implement, and train this model from scratch. A pretrained model, a fine-tuned pretrained model, or an LLM-based system does not satisfy this requirement.

Your model must use RNN, LSTM, GRU, or Transformer structures, or a combination of these, for sequence modelling and must incorporate an attention mechanism. Your architecture should demonstrate originality through thoughtful adaptations, combinations, or extensions that address the characteristics and challenges of your chosen task. Simply applying a standard architecture without meaningful task-specific design will not be sufficient.

Provide a complete and coherent technical description of your model, supported by clear notation, appropriate equations, and an architecture diagram. Explain how the model operates and justify its principal design choices, including the role of attention. Your description should provide sufficient detail for a reader to understand and reproduce your architecture. You are expected to identify and explain the technical details relevant to your particular design.

Clearly distinguish your own design contributions from components adapted from existing work. Cite the relevant publications, explain any modifications you make, and state any important assumptions or limitations of your approach.

#### Experimental Setup
This section should include the following aspects:
* **Dataset description:** Describe your selected dataset, explain how the data are prepared and organised for training, validation, and testing, including any preprocessing you apply.
* **Training procedure and hyperparameters:** Describe your training procedure and report the hyperparameter settings needed to reproduce your experiments. Explain how these settings were selected.
* **Baseline models:** Investigate relevant baseline approaches to contextualise the performance of your own model. These may include pretrained models, fine-tuned pretrained models, open-source LLMs (closed-source APIs or proprietary systems such as OpenAI GPT, Claude, Gemini are not allowed), or approaches proposed in existing research. You are encouraged to explore complementary approaches that represent different modelling or adaptation strategies. Justify your selections and explain what each comparison contributes to your investigation. A strong baseline investigation should demonstrate thoughtful exploration of relevant techniques and provide informative comparisons that reveal the strengths and limitations of your own approach. Identify the specific models used, cite relevant sources, and describe how each baseline is implemented or prompted. All baseline results included in your experimental comparison must come from experiments conducted by your group. Results reported in published work may be discussed for context, but must not replace your own baseline experiments.
* **Evaluation metrics:** Define your evaluation metrics and justify their suitability for the task and answer format of your selected dataset. Explain how the metrics are calculated.

#### Results and Analysis
Present and critically analyse your experimental findings using clear, appropriately labelled tables, figures, and visualisations. Your discussion should interpret the results and explain their implications. This section should include the following:
* **Overall performance comparison:** Compare the performance of your main model with your selected baselines using suitable tables or figures and discuss the principal findings.
* **Ablation studies:** Design controlled experiments to investigate the contribution of selected components or design choices in your main model. Clearly explain the question or hypothesis motivating each ablation, why the comparison is informative, and how relevant experimental conditions are controlled. Present the results using appropriate tables or figures and discuss whether they support your original design decisions.
* **Qualitative analysis of attention:** Use informative visualisations to examine the attention mechanism in your trained model. Select examples that demonstrate both successful and failed predictions, providing the relevant inputs, expected answers, and model predictions so that the visualisations can be interpreted. Analyse how the observed attention patterns relate to the model's behaviour and what they reveal about its strengths and limitations.

#### Conclusion
Summarise the main findings of your project and what you have learnt. Highlight your principal achievements and acknowledge the main limitations of your work. You may also suggest directions for future research or improvements.

#### Team Contributions
*(This section does not count towards the page limit.)*
If your team has more than one member, briefly describe each member's contributions to the project.

#### References
*(This section does not count towards the page limit.)*
Generate your references using BibTeX.

> **Important:** Your report MUST accurately describe the system you implemented and the experiments you conducted. Marks will be deducted for inconsistencies between the report and the submitted code. A strong report should interpret its findings and provide well-supported explanations for the model's successes and failures.

---

## Important Notes

* You may use deep learning libraries, such as PyTorch, to implement standard neural network components. You are not required to reimplement the internal operations of these library components.
* You are encouraged to consult relevant publications to guide your design and must acknowledge these sources appropriately. However, you **MUST NOT** copy implementation code from open-source projects, including GitHub repositories or code released with publications. Your model and experimental pipeline must be implemented by your team.
* You **MUST** use the provided [code template](https://colab.research.google.com/drive/1LF8KINrQHKXZBOhDyVe7y3EA5dm5_iYX) for your implementation.
* Your report **MUST** accurately reflect your submitted implementation and experimental results. Include the relevant training and evaluation logs, together with the results reported in your report, as saved outputs in your submitted `.ipynb` file.
* You **MUST NOT** introduce additional external datasets for training or evaluation during this project. This restriction does not prevent you from using pretrained checkpoints for baseline comparisons.
* Your main model **MUST** be trained from scratch. Pretrained models and checkpoints may be used for baseline experiments only. Clearly distinguish your main model from these baselines in both your report and implementation.
* You **MUST** conduct your own baseline experiments. You may use standard library interfaces and pretrained checkpoints, but must implement and run the experimental workflow yourselves. Published performance scores may be discussed for context but must not replace results obtained through your own experiments.
* You **MUST NOT** use the test set to guide training, design decisions, or model selection. Reserve it for final evaluation. You may split your validation data for hyperparameter tuning and model selection.
* Ensure that your implementation is reproducible. Document the required dependencies, settings, and execution steps, and check that your notebook can run from a clean runtime on Google Colab.

---

## Submission Instructions

Submit your work via LMS. For group submissions, only one group member should submit on behalf of the entire group.

You **MUST** submit the following two files:
* **Report:** A PDF file named `CITS4012_YourGroupID.pdf`.
* **Implementation:** A Jupyter notebook named `CITS4012_YourGroupID.ipynb`. This notebook must contain your complete project implementation, setup and execution instructions, saved training and evaluation logs, and reported results. Include any links and code required to download and load saved models (if required).

*(Replace `YourGroupID` with your assigned group ID in both filenames.)*

---

## Late Submission of Assignment

A penalty of 5 per cent of the total mark allocated for the assessment item is deducted per day for the first 7 days (including weekends and public holidays) after which the assignment is not accepted. For the first two days there will be a penalty waiver, which means that if you submit within that 48 hours period, the assessment will be marked late but the late penalty will not be applied. After 48 hours, the accrued penalty will apply, i.e. a 15% deduction will be applied on Day 3 (after 48 hours), with an additional 5% per day after that (to day 7).

---

## Indicative Marking Guide

This project will be marked out of **30 marks**. The following guide outlines the broad areas of assessment and the qualities expected in a strong submission. It provides guidance on how to approach the project. Assessment will consider the quality, breadth, depth, and rigour of your work, including how effectively your design choices, experiments, and analysis contribute to a coherent and well-supported investigation.

* **Model design and implementation:** A technically sound, clearly explained, and correctly implemented architecture that addresses the characteristics of the selected task. Strong submissions will demonstrate thoughtful originality, meaningful integration of attention, and well-justified design decisions.
* **Experimental investigation:** The breadth, relevance, and rigour of the experiments conducted. Strong submissions will investigate informative baseline approaches, design well-motivated and controlled ablation studies, and use appropriate training and evaluation procedures. Experimental choices should contribute to a purposeful investigation of the model and task.
* **Results and critical analysis:** Insightful interpretation of quantitative results, ablation findings, and attention visualisations. Strong submissions will provide evidence-supported explanations, examine both successful and failed predictions, and critically discuss limitations and unexpected findings.
* **Writing and presentation:** A coherent, concise, and well-organised report with clear technical explanations, appropriate referencing, and effective use of tables, figures, equations, and architecture diagrams. The report should follow the required format.
* **Engagement with relevant research:** Evidence of understanding and meaningfully applying relevant published work to inform the model design, experimental investigation, or interpretation of findings. Clearly explain how your work relates to or extends existing approaches.

Marks will reflect the substance of the work and the understanding demonstrated. Originality may be demonstrated through well-justified adaptations, combinations, or extensions of established methods. A reproducible implementation and supporting logs are mandatory requirements. Missing evidence or inconsistencies between the report, code, and recorded results will lead to deductions under the relevant marking criteria.

Marks are not awarded solely on the predictive performance of your model. Greater emphasis is placed on the soundness of your design, the justification of your choices, the quality of your experiments, and the depth of your analysis. Nevertheless, you are expected to demonstrate meaningful learning and evaluate your model against appropriate baselines.
