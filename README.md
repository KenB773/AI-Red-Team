# IMDB Sentiment Classification (HTB Skills Assessment)

This project tackles a binary sentiment classification task using the [IMDB movie reviews dataset](https://ai.stanford.edu/~amaas/data/sentiment/), as part of a HackTheBox skills assessment. The goal was to train a model capable of predicting whether a given movie review is **positive (1)** or **negative (0)**.

---

## Dataset Overview

* **Format**: JSON
* **Structure**:

  ```json
  {
    "text": "Review text here...",
    "label": 1
  }
  ```
* **Size**: 50,000 total reviews (25k train, 25k test)
* **Balanced classes**: equal positive and negative labels

---

## Approach

We used **scikit-learn's Pipeline** to bundle preprocessing and modeling into a deployable unit:

1. **Text Preprocessing**:

   * `TfidfVectorizer` converts text into a weighted bag-of-words representation.
   * Handles tokenization, stopword removal, and sublinear term frequency scaling.

2. **Model Training**:

   * `LogisticRegression` with `max_iter=1000`
   * Trained directly on raw review text via the pipeline

3. **Evaluation**:

   * Final model reached **\~88.3% accuracy** on the held-out test set

---

## Sample Code Snippet

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(sublinear_tf=True)),
    ("classifier", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "skills_assessment.joblib")
```

---

## Submission & Deployment Notes

* HTB required the model to be submitted as a **single `.joblib` file**
* It had to include:

  * The trained `TfidfVectorizer`
  * The fitted classifier
* **Common mistake**: submitting only the model without preprocessing caused the server to silently fail (0% accuracy)
* **Fix**: Bundle both steps into a `Pipeline` and train it **before** serialization

---

## Result

```json
{
  "accuracy": 0.88256,
  "precision": 0.88,
  "recall": 0.88,
  "f1-score": 0.88
}
```

The final model passed HTB evaluation and received the challenge flag.

---

## Key Takeaways

* Always serialize **preprocessing and model logic** together
* `Pipeline` makes your ML models portable and production-friendly
* Silent failures in remote model evaluators often point to missing preprocessing

---

## Files

* `skills_assessment.joblib`: Full trained pipeline
* `notebook.ipynb`: Development and evaluation code
* `train.json` / `test.json`: Raw dataset (if needed)

---

**Bonus Tip**: You can adapt this exact pipeline for toxicity classification, text moderation, or customer feedback analysis with minimal changes.
