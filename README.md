# Automated Resume Screening System 📄🔍

## 📌 Overview
Recruitment processes often involve manually filtering through thousands of resumes, which is time-consuming and prone to human error. This project automates the task by using **Natural Language Processing (NLP)** and **Machine Learning** to classify resumes into their respective job domains (e.g., Data Science, Web Development, HR, etc.) with high accuracy.

## 🚀 Features
* **Automated Categorization:** Classifies resumes into **25** specific job categories.
* **Text Preprocessing:** Cleans raw text by removing URLs, hashtags, mentions, and special characters using **RegEx**.
* **Feature Extraction:** Converts text data into numerical vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)**.
* **High Accuracy:** Achieves **~99% accuracy** on the test set using the KNN algorithm.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Matplotlib, Seaborn
* **NLP:** NLTK, Regular Expressions (re)
* **Machine Learning:** Scikit-Learn (KNeighborsClassifier, OneVsRestClassifier, TfidfVectorizer)

## 📊 Methodology
1.  **Data Loading:** Utilized a dataset containing **962 resumes** labeled with their respective categories.
2.  **Data Cleaning:** Implemented a helper function to strip unnecessary data:
    * URLs (`http\S+`)
    * RT and cc
    * Hashtags and Mentions
    * Punctuation and Special Characters
3.  **Visualization:** Analyzed category distribution using Seaborn count plots and pie charts.
4.  **Encoding:** Encoded categorical labels using `LabelEncoder`.
5.  **Vectorization:** Transformed the cleaned text into TF-IDF features.
6.  **Model Training:** Trained a **K-Nearest Neighbors (KNN)** classifier wrapped in a **One-vs-Rest** strategy to handle the multi-class classification problem.

## 📈 Results
The model was evaluated using a split of the dataset (80% training, 20% testing).
* **Training Accuracy:** ~100%
* **Test Accuracy:** ~99%

## 🔮 Future Scope
* Integration with a web application (Streamlit/Flask) for real-time file uploading.
* Support for parsing raw PDF and DOCX files directly.
* Expansion of the dataset to include more diverse job roles.