🛡️ Fraudulent Job Listing Detection using Machine Learning

 📌 Project Overview

This project aims to detect **fraudulent job postings** using Natural Language Processing (NLP) and machine learning techniques. It processes real-world job listing data, transforms textual features using TF-IDF vectorization, applies class balancing via SMOTE, and trains a `RandomForestClassifier` to classify listings as genuine or fraudulent.

It includes an end-to-end pipeline from preprocessing to model deployment, along with visual dashboards and an alert mechanism via email.


OUR GOAL:
our goal is to train a binary classifier for detecting fraud job listings. Our accuracy lies in the fact that maximum fraudulant jobs are being listed as fraud and labelled the same. We have forgone to let the genuine job listings be less accurate in marking as genuine, but we traded off the recall and precision for a better f1 score. Indicating our dedication in finding the fraud jobs and saving people from the hassle of fake recruiters trying to scam them.
---
BINARY CLASSIFIER:
A binary classifier learns to predict whether a new data point belongs to a "positive" class or a "negative" class. These classes can represent various things depending on the application, such as: Spam or not spam (email filtering), Fraudulent or not fraudulent (credit card transactions), Positive or negative (sentiment analysis).
---
HYPER PARAMETER TUNING:
Hyperparameter tuning is the process of finding the optimal set of hyperparameters for a machine learning model, which are parameters that are set before training begins and influence the learning process. These hyperparameters significantly impact a model's performance, complexity, and training speed. By adjusting these settings, hyperparameter tuning aims to improve a model's accuracy and overall effectiveness.

---
SET UP INSTRUCTIONS

QUICK SETUP:

github: https://github.com/vikashcoder8789/JobShield-Intelligent-Fraudulent-Job-Listing-Detector.git 



TO RUN THE PROGRAMS:

1. open terminal, in the folder where the folder 'dasashboard 'is located 

---

2. To run all the programs and the streamlit server use command: python run_all.py
This is a wrapper file.


---

3. to run everything else use separate python and streamlit commands on the terminal.
like
to run binary classifier use command:
python fraudulent_job_project.py

---
4. After running the classifier, you will recieve a test_predictions.csv file, which will further be used in the fraud_dashboard.py file.


---
5. DASHBOARD
---
Streamlit server is up and running, datasets and predictions are in the same folder. 
COMMAND to RUN : streamlit run dashboard.py
---
DASHBOARD in PYTHON FILE
visualization of dasboard in the python file is available. take a look to find it there.

 ⚙️ Key Features & Technologies Used

 ✅ Features
- Preprocessing of multiple text columns (`title`, `description`, etc.)
- Feature extraction using TF-IDF (bi-grams)
- Handling class imbalance using SMOTE
- Model training using RandomForestClassifier
- Threshold tuning for better fraud detection
- Confusion matrix and performance metrics
- Prediction dashboard and visualizations
- Email alert for high-risk job postings

 🧰 Technologies
- streamlit
- Python 3.x
- pandas, numpy
- scikit-learn
- imbalanced-learn (SMOTE)
- matplotlib, seaborn
- joblib
- smtplib

---


 🛠️ Setup Instructions

### 1. 🔽 Install Dependencies

Make sure you have Python 3 installed. Then run:


pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib
2. 📁 Prepare Dataset
Place the following files in the same directory as the script:

train_data.csv – contains training data with job features and labels

test_data.csv – contains test job listings to predict fraudulence

3. ▶️ Run the Script
python fraudulent_job_project.py

4. 📦 Output Files
fraud_model.pkl – Serialized model pipeline

test_predictions.csv – Final predictions with fraud probability and labels

5. 📊 Visualization
The script auto-generates interactive visualizations including:

Top job listings

Histogram of fraud probabilities

Pie chart of fraud vs real

Bar chart of most suspicious jobs

6. 📧 (Optional) Email Alerts
To enable email alerts for high-risk jobs:

Enable 2FA on your Gmail account

Create an app password

Replace the placeholder email/password in the script

send_email_alert(job_title="Data Analyst", fraud_prob=0.92)
📣 Notes
You can adjust fraud detection threshold by changing new_threshold in the script.





----
IMPORTTANT LINKS


github: https://github.com/vikashcoder8789/JobShield-Intelligent-Fraudulent-Job-Listing-Detector.git 

video presentation: https://drive.google.com/file/d/1_0S3Yj29Go8NNWP6ncwytfd-UFC5XGzl/view?usp=sharing 



created by:
vaibhav gauraha email: vaibhavgauraha62@gmail.com
vikash kumar email: vikash8298020427@gmail.com

for:
ANVESHAN HACKATHON
masai school

