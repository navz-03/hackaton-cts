from flask import Flask, request, render_template, session, redirect, url_for
import joblib
import pandas as pd
import google.generativeai as genai
from pymongo import MongoClient
from flask import jsonify
from padelpy import from_smiles
import pickle
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from sklearn.feature_selection import SelectKBest, f_regression


# Configure the Generative AI model
genai.configure(api_key="AIzaSyCFEHZDqDBkBMkFhl_R36Rj1YXZt_DcRTo")
model = genai.GenerativeModel('gemini-1.5-flash')



# Load the trained model and encoders
model_path = 'drug_prediction_model.pkl'
label_encoder1_path = 'label_encoder1.pkl'
label_encoder2_path = 'label_encoder2.pkl'
label_encoder_drug_path = 'label_encoder_drug.pkl'

model2 = joblib.load('finalized_ht_dtr_model.pkl')
drug_model = joblib.load(model_path)
label_encoder1 = joblib.load(label_encoder1_path)
label_encoder2 = joblib.load(label_encoder2_path)
label_encoder_drug = joblib.load(label_encoder_drug_path)

# Load the original dataset for validation
file_path = 'dataset\dataset.csv'
data = pd.read_csv(file_path)

# Column names for compounds and drug
compound1_col = 'compound1'
compound2_col = 'compound2'
drug_col = 'Drug'

# Function to predict drug name based on user input
def get_drug_description(prompt):
    
    try:
        # Make the API call without specifying the role
        # response = genai.chat(prompt=prompt)
        response = model.generate_content(prompt)
        # Log the full response for debugging
        return response.text


        # Check if response is valid and contains the expected data
        # # if response and "choices" in response and len(response["choices"]) > 0:
        # #     return response["choices"][0]["text"].strip()
        # else:
        #     return "Unable to fetch description. Please try again."

    except Exception as e:
        # Log the exception for debugging
        print("Exception occurred:", str(e))
        return f"An error occurred: {str(e)}"

def predict_drug(compound1, compound2):
    # Check if both compounds exist in the dataset in either order
    valid_row1 = data[(data[compound1_col] == compound1) & (data[compound2_col] == compound2)]
    valid_row2 = data[(data[compound1_col] == compound2) & (data[compound2_col] == compound1)]

    if not valid_row1.empty:
        # Compounds are in the same order as the input
        compound1_encoded = label_encoder1.transform([compound1])[0]
        compound2_encoded = label_encoder2.transform([compound2])[0]
    elif not valid_row2.empty:
        # Compounds are swapped in the input
        compound1_encoded = label_encoder1.transform([compound2])[0]
        compound2_encoded = label_encoder2.transform([compound1])[0]
    else:
        # Compounds do not exist in the dataset in any order
        return "invalid"
    
    # Prepare input data
    input_data = [[compound1_encoded, compound2_encoded]]
    
    # Predict the drug name
    prediction_encoded = drug_model.predict(input_data)
    prediction = label_encoder_drug.inverse_transform(prediction_encoded)
    return prediction[0]

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'AIzaSyCFEHZDqDBkBMkFhl_R36Rj1YXZt_DcRTo'

# MongoDB setup
client = MongoClient("mongodb+srv://raje21ad051:xjsV8nZ3E11PdghD@loginpage.fhbg0em.mongodb.net/?retryWrites=true&w=majority&appName=Loginpage")
db = client.get_database('user_data')
users_collection = db.get_collection('user')

@app.route('/register', methods=['POST'])
def register():

    print("hii")
    if request.method=='POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        print(username)
        
        password = request.form['password']
        # email = request.form['']
        if users_collection.find_one({'username': username}):
            return jsonify({"error": "User already exists"}), 400

        users_collection.insert_one({'username': username,'password': password,'email':email})
    return render_template('home.htmlpy')
    #return jsonify({"message": "User registered successfully"}), 201

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/loginpage')
def loginredirect():
    return render_template('login.html')

@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = users_collection.find_one({'username': username})
        
        if user and user['password'] == password:
            session['loggedin'] = True
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return "Incorrect username or password", 401
    
    return render_template('login.html')

@app.route('/logout')           
def logout():
    session.pop('loggedin', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/home', methods=['GET', 'POST'])
def home():
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/drug_discovery')
def drug_discovery():
    return render_template('index_1.html')

@app.route('/drug_discovery_utils', methods=['POST', 'GET'])
def drug_discovery_utils():
    if request.method == 'POST':
        compound1 = request.form['compound1']
        compound2 = request.form['compound2']
        prediction = None
        description = None
        error = None

        if compound1 and compound2:
            try:
                predicted_drug = predict_drug(compound1, compound2)
                if predicted_drug != "invalid":
                    description = get_drug_description(f"Provide a 3 line description of the drug named {predicted_drug}.")
                else:
                    predicted_drug = ' '
                    predict =f'''I am studying potential drug interactions. For the following two compounds, typically found in medications, predict the outcome of their combination:{compound1} and {compound2}.
Specifically, tell me:
        Drug Names: What are the common drug names associated with each compound?
        Interaction Type: Will these compounds likely interact? If so, is it a synergistic, additive, antagonistic, or other type of interaction?
        Clinical Significance: Briefly describe any potential health risks or benefits associated with this drug interaction.'''

                    description = get_drug_description(predict)
            except Exception as e:
                error = f"An error occurred: {str(e)}"

            return render_template('index_1.html', prediction=predicted_drug, description=description, error=error)
        else:
            error = "Please enter both compound names."
            return render_template('index_1.html', prediction=prediction, description=description, error=error)

    return render_template('index_1.html')

@app.route('/drug_interaction')
def drug_interaction():
    return render_template('model2.html',PIC50="",smiles="")

@app.route("/predict",methods=['POST', 'GET'])
def predict():
        selected_option = request.form.get('text')
        smiles_input = request.form.get('Smiles',type=str)
        name,molwt,logp,hacceptors,hdonors = get_smile_description(smiles_input)
        pchem_fp=from_smiles(smiles_input, fingerprints=True, descriptors=False)
        input_fp=pd.DataFrame(pchem_fp,index=[0])
        pIC50 = model2.predict(input_fp) 
        report = f'''The compound {smiles_input}, commonly referred to as {name}, exhibits a pIC50 value of {pIC50} against {selected_option} disease. {name} is characterized by a molecular weight of {molwt}, a logP value of {logp}, and features {hacceptors} hydrogen bond acceptors and {hdonors} hydrogen bond donors.'''
        return render_template("model2.html",description=report)

def get_smile_description(smile):
    
    mol = Chem.MolFromSmiles(smile)
    prompt = f'Using your chemical knowledge base, please identify the chemical compound corresponding to the following SMILES notation:{smile}?'
    response = model.generate_content(prompt)
    MolWt = Descriptors.MolWt(mol)
    MolLogP = Descriptors.MolLogP(mol)
    NumHDonors = Lipinski.NumHDonors(mol)
    NumHAcceptors = Lipinski.NumHAcceptors(mol)

    return response,MolWt,MolLogP,NumHAcceptors,NumHDonors

if __name__ == '__main__':
    app.run(debug=True)
