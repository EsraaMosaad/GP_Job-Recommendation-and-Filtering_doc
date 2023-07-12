
# **Job Recommendation and Filtering System**
This repository contains the code for our project that aims to provide job recommendations and filtering based on skills. We used several models to train on our dataset and extract skills, including Doc2Vec, Word2Vec, TF-IDF, FastText, and Topic Modeling LDA.

#Dataset
We used a dataset of job listings that contains job titles, descriptions, and required skills. Our dataset is available at the following link: https://drive.google.com/file/d/1-608bi-T6aMGUQwqJbyWrIQkTR0HI0TN/view?usp=sharing ↗

#Models
We trained several models on our dataset, including:

Doc2Vec: A neural network-based model that learns vector representations of documents.
Word2Vec: A neural network-based model that learns vector representations of words.
TF-IDF: A statistical model that measures the importance of words in a document.
FastText: A neural network-based model that learns vector representations of words, as well as character n-grams.
Topic Modeling LDA: A statistical model that extracts topics from a collection of documents.
After evaluating the performance of each model, we found that the TF-IDF model provided the best results. We used it to generate job recommendations and filtering based on skills.

#Deployment
To deploy the project, you can use the code provided in the app.py file. The app allows users to upload their CV and job requirements and receive job recommendations and filtering based on their input.

#Documentation
You can find the documentation for this project in the file named Job Recommendation and Filtering_doc.pdf. The documentation includes information on the dataset, model selection, and results.

#Team Members
Mohammed Zaki
Moaz Gamal
Abdelrahman Mohammed
Mostafa Mahmoud
Esraa Mosaad
Alaa Taher

#Supervisor
Dr. Mohammed Wahby
