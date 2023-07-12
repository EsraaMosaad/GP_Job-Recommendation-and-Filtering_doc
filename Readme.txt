

Job Recommendation and Filtering System
This repository contains the code for our project that aims to provide job recommendations and filtering based on skills and job requirements. We used several models to train on our dataset and extract skills, including Doc2Vec, Word2Vec, TF-IDF, FastText, and Topic Modeling LDA. We  update data set  new_data.csv for training.

Dataset
We used a dataset of job listings that contains job titles, descriptions, and required skills. Our dataset is available in the data folder and is split into train and test sets.

Models
We trained several models on our dataset, including:

Doc2Vec: A neural network-based model that learns vector representations of documents.
Word2Vec: A neural network-based model that learns vector representations of words.
TF-IDF: A statistical model that measures the importance of words in a document.
FastText: A neural network-based model that learns vector representations of words, as well as character n-grams.
Topic Modeling LDA: A statistical model that extracts topics from a collection of documents.
After evaluating the performance of each model, we found that the TF-IDF model provided the best results. We used it to generate job recommendations and filtering based on skills.

Results
We evaluated our model using accuracy, precision, recall, and F1 score metrics. Our model achieved an accuracy of 85% on the test set. We also performed a sensitivity analysis to evaluate the robustness of our model.

Deployment
To deploy the project, you can use the code provided in the app.file file. The app allows users to input their skills and job requirements and receive job recommendations and filtering based on their input.

Documentation
You can find the documentation for this project in the file named documentation.md. The documentation includes information on the dataset, model selection, and result.


