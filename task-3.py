import pandas as pd
import nltk
import string
import os

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# डाउनलोड stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -------------------------------
# 1. LOAD RESUME DATA
# -------------------------------
resume_df = pd.read_csv("resume/resume.csv")

# Check columns
print("Resume Columns:", resume_df.columns)

# IMPORTANT: update column name if needed
text_column = "Resume_str"   # auto-detect first column

# -------------------------------
# 2. CLEAN TEXT
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

resume_df['cleaned'] = resume_df[text_column].apply(clean_text)

# -------------------------------
# 3. TARGET JOB DESCRIPTION (YOU CAN CHANGE THIS)
# -------------------------------
target_job = """
Looking for a data scientist with skills in python, machine learning,
data analysis, SQL, and statistics
"""

target_cleaned = clean_text(target_job)

# -------------------------------
# 4. TF-IDF VECTORIZATION
# -------------------------------
vectorizer = TfidfVectorizer()

all_text = resume_df['cleaned'].tolist() + [target_cleaned]

vectors = vectorizer.fit_transform(all_text)

resume_vectors = vectors[:-1]
job_vector = vectors[-1]

# -------------------------------
# 5. SIMILARITY
# -------------------------------
scores = cosine_similarity(resume_vectors, job_vector)

resume_df['score'] = scores

# -------------------------------
# 6. SKILL EXTRACTION
# -------------------------------
skills_list = ['python', 'machine learning', 'sql', 'data analysis', 'java', 'excel']

def extract_skills(text):
    found = []
    for skill in skills_list:
        if skill in text:
            found.append(skill)
    return found

resume_df['skills'] = resume_df['cleaned'].apply(extract_skills)

# -------------------------------
# 7. SKILL GAP
# -------------------------------
job_skills = extract_skills(target_cleaned)

def skill_gap(resume_skills):
    return list(set(job_skills) - set(resume_skills))

resume_df['missing_skills'] = resume_df['skills'].apply(skill_gap)


# Skill-based scoring boost
def skill_score(resume_text, job_skills):
    score = 0
    for skill in job_skills:
        if skill in resume_text:
            score += 1
    return score

# Apply skill score
resume_df['skill_score'] = resume_df['cleaned'].apply(lambda x: skill_score(x, job_skills))

# Final score = similarity + skill weight
resume_df['final_score'] = resume_df['score'] + (0.2 * resume_df['skill_score'])

# Sort using final score
ranked_df = resume_df.sort_values(by='final_score', ascending=False)


# -------------------------------
# 9. SAVE OUTPUT
# -------------------------------
ranked_df.to_csv("ranked_candidates.csv", index=False)

# -------------------------------
# 10. SHOW TOP RESULTS
# -------------------------------
print("\n TOP CANDIDATES:\n")
print(ranked_df[[text_column, 'score', 'skills', 'missing_skills']].head(10))


