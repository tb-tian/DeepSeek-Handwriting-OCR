# Fine-Tuning DeepSeek-OCR for Vietnamese Handwritten Recognition

This repository contains the research, implementation, and evaluation of fine-tuning the **DeepSeek-OCR** model for **Vietnamese handwritten text recognition**.

---

## Project Overview

Handwritten recognition in Vietnamese presents unique challenges due to the complex system of diacritics (tone marks) and vowel modifiers ($a, \hat{e}, \hat{o}$) which are often written quickly, blurred, or displaced. This project leverages the **DeepSeek-OCR** architecture to bridge the gap between standard OCR and high-variability Vietnamese handwriting.

## Methodology

### Dataset

The project utilizes the **UIT-HWDB** dataset. Data was sampled to accommodate resource constraints (Google Colab T4):

* **Training Set:** 5,627 samples (50% of Line/Paragraph data and 500 Word samples).


* **Test Set:** 421 samples (100% of Line data, 20 Paragraph samples, and 200 Word samples).

### Fine-Tuning Strategy

The model was trained using **QLoRA** with the **PEFT** (Parameter-Efficient Fine-Tuning) mechanism.

* **Target Modules:** Adapters applied to projection layers ($q, k, v, o\_proj$) and MLP layers ($gate, up, down\_proj$).
* 
* **Hyperparameters:**
* Rank ($r$): 16 
* Learning Rate: 2e-4 
* Optimizer: AdamW 8-bit 
* Epochs: 1 

---

## Performance Evaluation

The primary metric used is **Character Error Rate (CER)**. Fine-tuning resulted in a significant reduction in errors compared to the baseline model.

| Evaluation Level | Baseline CER | Fine-tuned CER | Improvement |
| --- | --- | --- | --- |
| **Word** | 7.6715 | 0.5789 | +7.0926 |
| **Line** | 1.4304 | 0.2574 | +1.1730 |
| **Paragraph** | xx | xx | -xx |



### Analysis 

* **Baseline Hallucinations:** Before fine-tuning, the model often hallucinated Vietnamese handwriting into mathematical formulas (e.g., $4\pi^{2}$) or nonsense English.

* **Paragraph Outliers:** While Word and Line accuracy improved drastically, the average CER for Paragraphs increased due to an "infinite loop" bug in the EOS (End of Sentence) token detection on some outlier samples.

---

## Limitations & Future Work

1. **Visual Confusion:** The model still occasionally confuses characters with similar shapes (e.g., "đốn" vs. "đơn").

2. **Repetition Issues:** Implementing `repetition_penalty` or `no_repeat_ngram_size` during inference is recommended to solve paragraph-level loops.


3. **Data Expansion:** Future iterations will focus on **Synthetic Data** to provide more diverse training examples.

---

## Project Structure

* `Deepseek_OCR_report.pdf`: Detailed research report.


* `full_pipeline.ipynb`: Full pipeline for processing, training, and evaluation.
