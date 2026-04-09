# 📊 Resume / Candidate Screening System (ML Project)

## 🔍 Overview

This project implements a **Machine Learning-based Resume Screening System** that automatically analyzes and ranks candidates based on their relevance to a given job description.

The system uses **Natural Language Processing (NLP)** techniques to:

* Process resume text
* Extract relevant skills
* Compare resumes with job requirements
* Rank candidates based on suitability

---

## 🎯 Objective

To build a system that:

* Screens resumes automatically
* Matches candidate profiles with job descriptions
* Identifies missing skills
* Ranks candidates based on job relevance

---

## 🛠️ Technologies Used

* Python
* Pandas
* NLTK
* Scikit-learn (TF-IDF, Cosine Similarity)

---

## 📁 Dataset

* Resume Dataset (`resume.csv`)
* Contains resume text (`Resume_str`), categories, and IDs

---

## ⚙️ Methodology

### 1. Text Preprocessing

* Converted text to lowercase
* Removed punctuation
* Removed stopwords using NLTK

### 2. Feature Extraction (TF-IDF)

* Converted text into numerical vectors using TF-IDF
* Captures importance of words in resumes

### 3. Similarity Calculation

* Used **Cosine Similarity** to compare resumes with job description
* Generates similarity score

### 4. Skill Extraction

* Extracted predefined skills like:

  * Python
  * Machine Learning
  * SQL
  * Data Analysis

### 5. Skill Gap Analysis

* Compared candidate skills with job-required skills
* Identified missing skills

### 6. Skill Weighting (Improvement)

* Added additional scoring based on skill matches
* Final score = similarity score + skill weight

### 7. Ranking System

* Ranked candidates based on final score
* Higher score = better match

---

## 📊 Output

The system generates:

* Ranked list of candidates
* Skills identified in resumes
* Missing skills for each candidate

---

## 🚀 How to Run

```bash
python task-3.py
```

---

## 📂 Output File

* `ranked_candidates.csv`

--- 
##OUTPUT SCREENSHOTS
<img width="1366" height="692" alt="Screenshot 2026-04-09 221303" src="https://github.com/user-attachments/assets/0ae5b80e-f233-4f4b-9fd5-9ab268c473d0" />

---

## 📌 Key Features

✔ Resume ranking
✔ Skill extraction
✔ Skill gap analysis
✔ Improved scoring using skill weighting

---

## 🎯 Conclusion

This project demonstrates how Machine Learning can be used in real-world hiring systems to automate resume screening, improve efficiency, and support better decision-making.

---
