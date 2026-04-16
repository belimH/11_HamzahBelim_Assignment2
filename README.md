#  Sentiment analysis on Artemis 

## (1) Problem Statement
Social media platforms generate large volumes of user-generated comments daily. Manually analyzing the sentiment of these comments is inefficient.
This project aims to automatically classify comments from Artemis’s photo posts into Positive, Negative, and Neutral sentiments using machine learning models.

## (2) Objective
Collect and manually label 100 comments from Artemis’s social media posts
Categorize comments into Positive, Negative, and Neutral
Apply machine learning classifiers:
Naive Bayes
Logistic Regression
Support Vector Machine (SVM)
Evaluate model performance using Precision, Recall, F1-Score, and Accuracy
Compare models and extract insights

## (3) Dataset
- Source: Comments collected from Artemis’s publicly available photo posts on social media
- Features:
    Raw text (original comment)
    Cleaned text (processed version)
    TF-IDF features (numerical representation)
    Sentiment label (Positive / Negative / Neutral)
- Size:
    Total: 100 comments
    Training: 80
    Testing: 20

## (4) Methodology
1. Data Preprocessing 
    Convert text to lowercase
    Remove punctuation, URLs, and special characters
    Remove stopwords
    Convert text to numerical form using vectorization
2. EDA  
    Visualized sentiment distribution
    Identified class imbalance
  3. Model Building  
    Naive Bayes (MultinomialNB)
    Logistic Regression
    Support Vector Machine (SVM)
4. Evaluation
    Precision
    Recall
    F1-Score
    Accuracy
    Confusion Matrix

## (5) Results
- Metrics and insights
Model	Accuracy
Naive Bayes	80%
Logistic Regression	80%
SVM	80%

## (6) How to Run
```bash
pip install -r requirements.txt
python main.py
```

## (7) Conclusion
    This project demonstrates the effectiveness of classical machine learning models for sentiment analysis.
    Although all models achieved similar accuracy, performance was affected by dataset imbalance, especially for           Neutral class.
    
    Future Improvements:
    Increase dataset size
    Balance dataset (more Neutral samples)
    Use advanced models like BERT
    Apply techniques like oversampling

## (8) Student's details
- Name: Hamzah Belim
- Roll No: 11
- UIN: 231A014
- YEAR: TE-AIDS
