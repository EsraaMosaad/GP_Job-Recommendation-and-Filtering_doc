import difflib
import sqlite3
import PyPDF2
import pandas as pd
import numpy as np
import ast
import spacy
import nltk
import nltk
try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')
from matplotlib import pyplot as plt, cm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity , manhattan_distances
from pyresparser import ResumeParser
from docx import Document
from googlesearch import search
from fuzzywuzzy import fuzz, process
import gensim
from builtins import enumerate
from jinja2 import Environment
env = Environment()
env.globals.update(enumerate=enumerate)
import string
# nltk.download('punkt')

# nltk.download('wordnet')

import pickle
with open('word2vecmodel.pkl', 'rb') as f:
    my_model = pickle.load(f)





# Load job data from database
conn = sqlite3.connect('data2.db')
job_data = pd.read_sql_query('SELECT * FROM jobs', conn)
conn.close()
# Load skill from database
conn2 = sqlite3.connect('all_skills.db')
all_skills = pd.read_sql_query('SELECT * FROM jobs', conn2)
conn2.close()
# print(job_data.head())

# #Preprocessind
def data_preprocessing_text(text):
    ## Lower case
    text = text.lower()

    ## remove tabulation and punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    ## digits
    text = ''.join([i for i in text if not i.isdigit()])

    text = ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in text])

    # remove stop words
    stop = stopwords.words('english')
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop]
    text = ' '.join(tokens)

    ## lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    text = ' '.join(tokens)
    return text


def extract_skills_2(text):
      # Create a new Word document
      doc = Document()
      # Add text to the document
      doc.add_paragraph(text)
      doc.save("text.docx")
      data = ResumeParser('text.docx').get_extracted_data()
      skills = data.get('skills', [])
      return skills




# Load pre-trained models and libraries
spacy_model = spacy.load('en_core_web_sm')
nltk_stopwords = set(stopwords.words('english'))

# Define function to extract skills from a resume using multiple models and libraries
def extract_skills_1(resume_text):

    # Use spaCy to extract noun chunks from resume text
    spacy_noun_chunks = [chunk.text for chunk in spacy_model(resume_text).noun_chunks]

    doc = spacy_model(resume_text)
    spacy_skills = [ent.text for ent in doc.ents if ent.label_ == 'ORG']


    # Use NLTK to extract keywords and phrases from resume text
    nltk_tokens = word_tokenize(resume_text)

    # Extract keywords and phrases
    nltk_filtered_tokens = (token.lower() for token in nltk_tokens if token.lower() not in nltk_stopwords)
    nltk_keywords = nltk.FreqDist(nltk_filtered_tokens).most_common(50)
    nltk_bigram_measures = BigramAssocMeasures()
    nltk_finder = BigramCollocationFinder.from_words(nltk_filtered_tokens)
    nltk_finder.apply_freq_filter(1)
    nltk_phrases = nltk_finder.nbest(nltk_bigram_measures.raw_freq, 50)

    nltk_skills = list(set([keyword for keyword, freq in nltk_keywords] + [' '.join(phrase) for phrase in nltk_phrases]))
    # Combine all extracted skills into a single list and remove duplicates
    skills = list(set( spacy_noun_chunks + spacy_skills + nltk_skills))
     # Filter skills based on the provided list of technical skills
    # import_skills = list(set([skill.lower() for skill in skills]) & set(technical_skills['name'].str.lower()))
    return skills 



#TFIDF

# # check for duplicates
# duplicates = job_data[job_data.duplicated()]
# # print the duplicates
# print(duplicates)
# drop duplicates and keep the first occurrence
job_data = job_data.drop_duplicates()
job_data= job_data.reset_index(drop=True)
job_data = job_data.dropna(subset=['skills_1'])
job_data = job_data[job_data['skills_1'].apply(len) > 0]
job_data= job_data.reset_index(drop=True)
job_data['skills_1'] = job_data.apply(lambda row: ast.literal_eval(row['skills_1']), axis=1)
job_data['skills'] = job_data['skills_1'].apply(lambda row: ' '.join(row))
# print(job_data)
vectorizer2 = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer2.fit_transform(job_data['skills'])


def get_top_n_tfidf(cv_text, n):
    # Perform LSA on the job description matrix
    lsa = TruncatedSVD(n_components=100)
    lsa_matrix = lsa.fit_transform(tfidf_matrix)

    # Transform the CV vector using the same vectorizer
    cv_lsa = lsa.transform(vectorizer2.transform([cv_text]))

    # Compute cosine similarity between the CV vector and the LSA matrix
    similarity_scores = cosine_similarity(cv_lsa, lsa_matrix)

    # Create a dataframe of the job titles and cosine similarities
    data = {"Job_title": job_data["Job_title"],
            "cosine_similarity": similarity_scores[0]}
    df = pd.DataFrame(data)

    # Get the top n job recommendations based on cosine similarity
    similar_jobs_indices = np.argsort(similarity_scores)[0][::-1][:n]
    similar_jobs = df.loc[similar_jobs_indices, ['Job_title', 'cosine_similarity']]
    # similar_jobs = similar_jobs.drop_duplicates(subset='Job_title',keep = 'first')
    return similar_jobs






def get_similar_jobs_for_job(job_title, num_results=5):
    query = f'{job_title} jobs)'
    results= search(query, tld='com', num=num_results, stop=10, pause=2)
    similar_jobs = []
    for result in results:
        if "job" in result.lower():
            similar_jobs.append(result)
    # Remove duplicates from the list of similar jobs
    similar_jobs = list(set(similar_jobs))

    # Sort the list of similar jobs by job title
    similar_jobs.sort()

    # Return the list of similar jobs
    return similar_jobs[:num_results]


# # Create a mapping of normalized skill forms to the original skill names
# normalized_skill_dict = {}
# for skill in all_skills['skill_name']:
#     normalized_skill = skill.lower().translate(str.maketrans('', '', string.punctuation))
#     if normalized_skill not in normalized_skill_dict:
#         normalized_skill_dict[normalized_skill] = []
#     normalized_skill_dict[normalized_skill].append(skill)
# # print(normalized_skill_dict)
# def get_skills_for_cv(cv_skills, job_skills):
#     matching_cv_skills = []
#     recommended_cv_skills = ['test-not set']

#     # Match normalized CV skills to normalized skills in the dataset
#     for cv_skill in cv_skills:
#         normalized_cv_skill = cv_skill.lower().translate(str.maketrans('', '', string.punctuation))
#         if normalized_cv_skill in normalized_skill_dict:
#             # Find the most similar skill among the matching skills
#             best_match_score = -1
#             best_match_skill = ''
#             for skill in normalized_skill_dict[normalized_cv_skill]:
#                 score = fuzz.token_sort_ratio(normalized_cv_skill, skill.lower().translate(str.maketrans('', '', string.punctuation)))
#                 if score > best_match_score:
#                     best_match_score = score
#                     best_match_skill = skill
#             # Add the best matching skill to the list of matching CV skills
#             if best_match_score > 60: # Set a threshold for similarity score
#                 matching_cv_skills.append(best_match_skill)

#     print("CV skills:", cv_skills)
#     print("Job skills:", job_skills)
#     print("Matching CV skills:", list(set(matching_cv_skills)))
#     print("Recommended CV skills:", list(set(recommended_cv_skills)))
#     return list(set(matching_cv_skills)), list(set(recommended_cv_skills))

##function to get the vector representation of a job description 
def get_job_vector(job_description):
    #tokenize the job description into individual words
    tokens = job_description

    #get the vector representation of each word in the job description
    vectors = [my_model.wv[token] for token in tokens if token in my_model.wv]

    #calculate the average vector of all words in the job description
    if vectors:
        avg_vector = np.mean(vectors, axis=0)
    else:
        avg_vector = np.zeros(my_model.vector_size)

    return avg_vector


#print(job_data['skills'][0])
#print(get_job_vector(job_data['needed_skills'][0]))
#store all job descriptions
# job_data = job_data.dropna(subset=['skills'])
all_job_vectors=[get_job_vector(i) for i in job_data['skills']]
# print(len(all_job_vectors))




# from flask import Flask,  render_template
from flask import Flask, request ,render_template, url_for, send_from_directory, jsonify

import os

app = Flask(__name__)
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


@app.route('/', methods=['GET'])
def start_page():
    return render_template('start_page.html')

@app.route('/recommendation_page.html', methods=['GET'])
def show_recommendation_page():
    return render_template('recommendation_page.html')

@app.route('/team.html', methods=['GET'])
def show_team_page():
    return render_template('team.html')

@app.route('/filtering_page.html', methods=['GET'])
def show_filtering_page():
    return render_template('filtering_page.html')

@app.route('/result.html', methods=['GET'])
def show_result_page():
    return render_template('result.html')





def get_best_cvs(all_CVs_vectors, job_description, num_of_CVs): 
    query_vector = get_job_vector(job_description)
    # lsa = TruncatedSVD(n_components=100)
    # lsa_matrix = lsa.fit_transform(all_CVs_vectors)
    #
    # # Transform the CV vector using the same vectorizer
    # cv_lsa = lsa.transform(get_job_vector(job_description).reshape(1, -1))
    # similarities = cosine_similarity(query_vector.reshape(1,-1),all_CVs_vectors)
    #     # [np.dot(query_vector, other)/np.linalg.norm(query_vector)/np.linalg.norm(other) for other in all_CVs_vectors]
    print(len(query_vector))
    lsa = TruncatedSVD(n_components=60)
    lsa_matrix = lsa.fit_transform(all_CVs_vectors)

    # Transform the CV vector using the same vectorizer
    cv_lsa = lsa.transform(query_vector.reshape(1, -1))

    # Compute cosine similarity between the CV vector and the LSA matrix
    similarities = cosine_similarity(cv_lsa, lsa_matrix)
    topk_indices = np.argsort(similarities)[0][::-1][:num_of_CVs]
    # # topk_indices = np.argsort(similarities)[::-1][:num_of_CVs]
    # topk_similarities = sorted(similarities, reverse=True)[:num_of_CVs]

    # Create a dataframe of the job titles and cosine similarities
    data = {"topk_indices": all_CVs_vectors,
            "cosine_similarity": similarities[0]}
    df = pd.DataFrame(data)

    topk_similarities = df.loc[topk_indices, ['cosine_similarity']]
    print("from get best cv topk_similarities" ,topk_similarities['cosine_similarity'].tolist())
    # similar_jobs = similar_jobs.drop_duplicates(subset='Job_title',keep = 'first')
    return topk_indices, topk_similarities['cosine_similarity']


def filter(CVs,job_description,num_of_CVs):
    # cv_text.append((filename, filepath, cv_skills_str, cv_info))  # include filename and filepath in the tuple
  df = pd.DataFrame(CVs, columns=['Names','file_path','CVs','cv_info'])
  print(df['CVs'])

  all_CVs_vectors=[get_job_vector(i) for i in df['CVs']]
  # print("from filter function",all_CVs_vectors)
  top_CV,similarities=get_best_cvs(all_CVs_vectors,job_description, num_of_CVs)
  return top_CV,similarities

import re
def extract_names(text):
    name_regex = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
    names = re.findall(name_regex, text)
    names = ', '.join(names)
    return names



def extract_email(text):
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_regex, text)
    emails = ', '.join(emails)
    return emails

def extract_mobile_number(text):
    phone = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]',text)
    
    if phone:
        number = ''.join(phone)
        if len(number) > 10:
            return  number
        else:
            return number








# filter(cvs_lst,job,5)

## take text(job description) for filtering task

@app.route('/flitering system', methods=['POST'])
def predict_filtering():
   if request.method == 'POST':
        # Get the uploaded CV files
        cv_files = request.files.getlist('cv_files[]')
        if not cv_files:
            # If no CV files were uploaded, return an error message
            return 'Error: no CV files uploaded'

        # Create a folder to store the uploaded CV files, if it doesn't exist
        if not os.path.exists('static/cv_uploads'):
            os.makedirs('static/cv_uploads')

        # Read the text content of the CV files
        cv_text = []
        for cv_file in cv_files:
            filename = cv_file.filename
            filepath = os.path.join('static/cv_uploads', filename)
            cv_file.save(filepath)  # save the file to the cv_uploads folder
            pdf_reader = PyPDF2.PdfReader(filepath)
            # pdf_reader = PyPDF2.PdfFileReader(cv_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            phone=extract_mobile_number(text)
            email=extract_email(text)



            cv_info = { 'phone': phone, 'email': email}

            clean_text= data_preprocessing_text(text)
            Cv_skills = extract_skills_1(clean_text) + extract_skills_2(clean_text)
            # cv_skills_str = ' '.join(Cv_skills)
            cv_text.append((filename, filepath, Cv_skills, cv_info))# include filename and filepath in the tuple

         # Get the number of CVs
        num_cvs = request.form.get('num_cvs')
        if num_cvs:
            num_cvs = int(num_cvs)
        else:
            num_cvs = 1 

        # Get the job description text
        job_description = request.form.get('job_description')
        if not job_description:
            # If no text was entered, return an error message
            return 'Error: no text entered'

        # Preprocess text
        cleaned_job = data_preprocessing_text(job_description)

        job_skills = extract_skills_1(cleaned_job) + extract_skills_2(cleaned_job)
        # job_skills_str = ' '.join(job_skills)
        top_CV, similarities = filter(cv_text, job_skills, num_cvs)

        # Initialize a list to store the file paths of the top CVs
        top_cv_files = []
        # print(len(cv_text))
        for index in top_CV:
           print(cv_text[index][1], similarities[index])
           top_cv_files.append(cv_text[index][1])  # append file path instead of file name

        # Pass the similar_jobs data to the templates

        return render_template('filtering_result.html', best_cvs=[cv_text[i] for i in top_CV.tolist()], top_cv_files=top_cv_files)
   return render_template('filtering_page.html')


#Recommendation Task
# import threading

# def create_figure(job_title, cosine_similarity):
#     # Create the bar plot
#     color_map = cm.get_cmap('cool')  # choose a color map
#     colors = color_map(cosine_similarity)
#     fig, ax = plt.subplots(figsize=(8, 5))
#     ax.barh(job_title, cosine_similarity, color=colors)
#     ax.set_title('Top job recommendations based on similarity', fontsize=12)
#     plt.tight_layout()

#     # Save the plot to a file and close the figure
#     plot_path = os.path.join(app.static_folder, 'bar_plot.png')
#     plt.savefig(plot_path)
#     plt.close(fig)
@app.route('/recommendation_system', methods=['POST'])
def predict_recommendation():
    if request.method == 'POST':
        # Get the uploaded CV files
        cv_file = request.files.get('file-upload')
        if not cv_file:
            # If no CV files were uploaded, return an error message
            return 'Error: no CV file uploaded'

        # Read the text content of the CV files
        pdf_reader = PyPDF2.PdfReader(cv_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        clean_cv = data_preprocessing_text(text)

        # Get the similar jobs and similarity scores
        Cv_skills = extract_skills_1(clean_cv) + extract_skills_2(clean_cv)
        cv_skills_str = ' '.join(Cv_skills)
        result = get_top_n_tfidf(cv_skills_str, 10)
        # skills = get_cv_skills(cv_skills, result)
        result['cosine_similarity'] = result['cosine_similarity'].apply(lambda x: round(x, 2) * 100)
        color_map = cm.get_cmap('cool')  # choose a color map
        colors = color_map(result['cosine_similarity'])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(result['Job_title'], result['cosine_similarity'], color=colors)
        ax.set_title('Top job recommendations based on similarity', fontsize=12)
        plt.tight_layout()

        # Save the plot to a file and close the figure
        plot_path = os.path.join(app.static_folder, 'bar_plot.png')
        plt.savefig(plot_path)
        plt.close(fig)

        # # Create the figure in a separate thread
        # t = threading.Thread(target=create_figure, args=(result['Job_title'], result['cosine_similarity']))
        # t.start()

        # Pass the similar_jobs data, plot path, matching CV skills, and recommended CV skills to the templates
        return render_template('recommendation_result.html', best_jobs=result, plot_path='bar_plot.png', cv_skills=Cv_skills)

    return render_template('recommendation_page.html')
@app.route('/recommendation_system_feature', methods=['POST'])
def recommendation_system_feature():

    # Get the selected job title and the action from the form data
    job_title = request.form['job']
    action = request.form['action']
    cv_skills = request.form.getlist('cv_skills')
    # cv_skills_str = request.form['cv_skills']
    #
    # # Convert the CV skills string to a list of skills
    # cv_skills = cv_skills_str.split()

    if action == 'cv-skills':
        job_title_index = int(request.form['jobTitleIndex'])
        job_skills = job_data.loc[job_title_index, 'skills_1']

        matching_cv_skills, recommended_cv_skills = get_skills_for_cv(ast.literal_eval(cv_skills[0]), job_skills)
        # cv_skills_list = []
        # cv_skills_list.extend(matching_cv_skills)
        # cv_skills_list.extend(recommended_cv_skills)
        return '<ul>' + ' , '.join(matching_cv_skills) + '</ul>' +' <br> '+'<ul>' + ' , '.join(recommended_cv_skills) + '</ul>'
        # return render_template('recommendation_result.html', matching_cv_skills=matching_cv_skills,
        #                        recommended_cv_skills=recommended_cv_skills)

    elif action == 'similar-jobs':
        # Get similar jobs for the selected job
        similar_jobs = get_similar_jobs_for_job(job_title)

        # Generate a list of links for the similar jobs
        links = ['<li><a href="' + job + '" target="_blank">' + job + '</a></li>' for job in similar_jobs]
        # print(job_title)
        # print(links)

        # Return the list of links as a string
        return '<ul>' + '\n'.join(links) + '</ul>'
    return render_template('recommendation_result.html')



if __name__ == '__main__':
    # app.config['UPLOAD_FOLDER'] = 'recommendation_files'
    # app.config['UPLOAD_FOLDER2'] = 'filtering_files'

    app.run(port=3000,debug=True)


