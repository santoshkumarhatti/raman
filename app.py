from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
import PyPDF2
import re
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import joblib

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'super secret key'

# Setup directories and file extensions
path = os.getcwd()
UPLOAD_FOLDER = os.path.join(path, 'uploads')
MODEL_FOLDER = os.path.join(path, 'models')
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the machine learning model
model_path = os.path.join(MODEL_FOLDER, 'skill_model.pkl')
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    raise RuntimeError(f"The model file at {model_path} does not exist. Please ensure the model is trained and the path is correct.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(filepath):
    text = ""
    with open(filepath, 'rb') as pdfFileObj:
        pdfReader = PyPDF2.PdfReader(pdfFileObj)
        num_pages = len(pdfReader.pages)
        for count in range(num_pages):
            pageObj = pdfReader.pages[count]
            text += pageObj.extract_text()
    return text

def is_pdf_empty(filepath):
    text = extract_text_from_pdf(filepath)
    return not text.strip()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'warning')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'warning')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            if is_pdf_empty(file_path):
                flash('The PDF seems to be empty or black.', 'warning')
                return redirect(request.url)

            flash('File uploaded successfully.', 'success')
            return redirect(url_for('screening', name=filename) + "#hasil")
        else:
            flash('Upload a valid PDF file.', 'warning')
            return redirect(request.url)
    return render_template("index.html")

@app.route('/screening/<name>', methods=['GET', 'POST'])
def screening(name):
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            text = extract_text_from_pdf(filename)
    else:
        text = extract_text_from_pdf(os.path.join(app.config['UPLOAD_FOLDER'], name))

    text = clean_resume(text)

    if not text.strip():
        flash('No text found in the uploaded file.')
        return redirect(url_for('upload_file'))

    skills = predict_skills(text, threshold=5)
    summary, plot_url = generate_summary_plot(skills)
    bidang, data_all_list, pie_plot_url = existing_screening_functionality(name)

    return render_template("screening.html", filename=name, bidang=bidang, data_all_list=data_all_list,
                           summary=summary, plot_url=pie_plot_url, skills_summary=summary, ml_plot_url=plot_url)

def clean_resume(resumeText):
    resumeText = re.sub(r'http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub(r'RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub(r'#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub(r'@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
    resumeText = re.sub(r'\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText.lower()

def predict_skills(text, threshold=1):
    skills_pred = model.predict([text])[0]
    skills_prob = model.predict_proba([text])[0]

    # Filter out skills with probability below the threshold
    skills = {skill: prob * 100 for skill, prob in zip(model.classes_, skills_prob) if prob * 100 >= threshold}
    return skills

def generate_summary_plot(skills):
    # Create a DataFrame for better visualization
    df = pd.DataFrame(list(skills.items()), columns=['Skill', 'Probability'])

    # Define a list of colors
    colors = ['#FF6347', '#4682B4', '#32CD32', '#8A2BE2']  # Adjusted colors for 4 categories

    # Plot the data
    fig, ax = plt.subplots(figsize=(12, 8))
    df.plot(kind='bar', x='Skill', y='Probability', ax=ax, color=colors[:len(df)])

    # Set x-axis labels to be horizontal (no rotation)
    ax.set_xticklabels(df['Skill'], rotation=0, ha='center', fontsize=10)
    ax.set_xlabel('Skills', fontsize=12)
    ax.set_ylabel('Probability (%)', fontsize=12)
    ax.set_title('Predicted Skills Probability', fontsize=14)

    plt.tight_layout()

    # Convert plot to PNG image
    png_image = BytesIO()
    plt.savefig(png_image, format='png')
    png_image.seek(0)
    plot_url = base64.b64encode(png_image.getvalue()).decode('utf8')

    return df, 'data:image/png;base64,{}'.format(plot_url)


def existing_screening_functionality(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(file_path, 'rb') as pdfFileObj:
        pdfReader = PyPDF2.PdfReader(pdfFileObj)
        num_pages = len(pdfReader.pages)
        text = ""
        for count in range(num_pages):
            pageObj = pdfReader.pages[count]
            text += pageObj.extract_text()

    text = clean_resume(text)

    bidang = {
        'Project Management': ['administration', 'agile', 'feasibility analysis', 'finance', 'leadership',
                               'management', 'milestones', 'planning', 'project', 'risk management',
                               'teamwork', 'communication', 'organization', 'research',
                               'public speaking', 'problem solving', 'negotiation', 'team management',
                               'time management', 'adaptability', 'policy knowledge','reporting',
                               'motivation'],

        'Backend': ['flask', 'laravel', 'django', 'ruby on rails', 'express.js', 'codeigniter', 'golang', 'mysql',
                    'postgres', 'mongodb', 'relational database', 'non relational database', 'nosql',
                    'application programming interface', 'object oriented programming'],

        'Frontend': ['angular', 'vue.js', 'svelte', 'jquery', 'backbone.js ', 'ember.js', 'semantic-ui',
                     'html', 'css', 'bootstrap', 'javascript',  'xml', 'dom manipulation', 'json'],

        'Devops': ['testing','networking', 'tcp' ,'AWS','udp', 'microsoft azure', 'amazon web services', 'alibaba cloud',
                   'google cloud',
                   'docker', 'kubernetes', 'virtual machine', 'cloud computing', 'security', 'linux', 'ubuntu',
                   'debian', 'arch linux', 'kali linux', 'automation', 'containers', 'operations', 'security',
                    'troubleshooting']
    }

    project = 0
    backend = 0
    frontend = 0
    devops = 0

    project_list = []
    backend_list = []
    frontend_list = []
    devops_list = []

    for word, item in bidang.items():
        if word == 'Project Management':
            for data in item:
                if data in text:
                    project += 1
                    project_list.append(data)
        elif word == 'Backend':
            for data in item:
                if data in text:
                    backend += 1
                    backend_list.append(data)
        elif word == 'Frontend':
            for data in item:
                if data in text:
                    frontend += 1
                    frontend_list.append(data)
        elif word == 'Devops':
            for data in item:
                if data in text:
                    devops += 1
                    devops_list.append(data)

    data_all = {
        'Project Management': project,
        'Backend': backend,
        'Frontend': frontend,
        'Devops': devops
    }

    data_all_list = [project_list, backend_list, frontend_list, devops_list]

    # Create a bar plot
    fig, ax = plt.subplots()
    bars = ax.bar(data_all.keys(), data_all.values(), color=colors[:len(data_all)])

    # Set x-axis labels to be horizontal (no rotation)
    ax.set_xticklabels(data_all.keys(), rotation=0, ha='center', fontsize=10)
    ax.set_xlabel('Skills', fontsize=12)
    ax.set_ylabel('Probability (%)', fontsize=12)
    ax.set_title('Predicted Skills Probability', fontsize=14)

    plt.tight_layout()

    # Convert plot to PNG image
    png_image = BytesIO()
    plt.savefig(png_image, format='png')
    png_image.seek(0)
    plot_url = base64.b64encode(png_image.getvalue()).decode('utf8')

    return data_all, data_all_list, 'data:image/png;base64,{}'.format(plot_url)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
