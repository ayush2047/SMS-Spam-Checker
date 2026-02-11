# SMS Spam Filtering with Machine Learning

This project focuses on building an intelligent SMS spam classifier using machine learning techniques. It uses text preprocessing, TF-IDF vectorization, and classification models to accurately detect spam messages.

## 🚀 Features
- Clean and labeled SMS dataset (5,572 messages)
- Preprocessing: Lowercasing, stopword removal, stemming, punctuation removal
- Vectorization using TF-IDF
- Multiple classifiers tested: Naive Bayes, SVM, Random Forest, etc.
- Streamlit web app for live spam classification
- Pickle-based model deployment

  
## 🧠 Algorithms Used
- **Multinomial Naive Bayes**: High precision, fast, effective on small datasets
- **Support Vector Classifier (SVC)**: High accuracy, works well with sparse text data
- **Random Forest & Ensemble Models**: Tested for comparison
- **Voting & Stacking Classifiers**: Combined models to improve results

## 📊 Model Performance (on test set)
| Model           | Accuracy | Precision |
|----------------|----------|-----------|
| Naive Bayes     | 97.2%    | 100%      |
| SVC             | 97.9%    | 93.3%     |
| Random Forest   | 98.0%    | 99.1%     |
| Stacking Classifier | 97.8% | 93.3%     |

> Note: Naive Bayes showed highest precision, while SVC and ensemble methods gave balanced accuracy.

## 📦 Dataset
- **Source:** [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- **Size:** 5,572 labeled SMS messages
- **Classes:** `ham` (not spam), `spam`

## 🧪 How to Run

### ⚙️ Install dependencies
```bash
pip install streamlit
pip install scikit-learn
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install nltk
pip install -r requirements.txt
pip install joblib
pip install pickle-mixin
streamlit run app.py




📄 Instructions to Run — SMS Spam Classifier (Streamlit Version)


✅ Prerequisites
	Before running the project, make sure the following are installed:
	Python (version 3.7 or above)
	pip (Python package manager)
	Git (optional, to clone the repository)
	Internet connection (to download NLTK resources)




📁 Project Structure

sms-spam-classifier/
├── spam.csv                  # Dataset
├── sms-spam-detection.ipynb  # Jupyter Notebook with EDA & model building
├── model.pkl                 # Trained ML model (e.g., SVC or CatBoost)
├── vectorizer.pkl            # TF-IDF vectorizer or CountVectorizer
├── streamlit_app.py          # Streamlit web application (new)
├── nltk.txt                  # NLTK downloads list
├── requirements.txt          # Python dependencies
└── Instructions to Run.txt   # You're reading this!




⚙️ Step-by-Step Setup Instructions

	1. Clone the Repository
		git clone https://github.com/campusx-official/sms-spam-classifier.git
		cd sms-spam-classifier

	2. Create and Activate Virtual Environment (Recommended)
		python -m venv venv
		# On Windows:
		venv\Scripts\activate
		# On macOS/Linux:
		source venv/bin/activate

	3. Install Required Libraries
		pip install -r requirements.txt
		If requirements.txt is missing or incomplete, install manually:
		pip install streamlit scikit-learn pandas numpy matplotlib seaborn nltk
	
	4. Download NLTK Resources
		If not already done, download the required NLTK data:

		# Run this once in Python shell
			import nltk
			nltk.download('punkt')
			nltk.download('stopwords')





🚀 Running the Streamlit App

	1. Ensure that model.pkl and vectorizer.pkl are present in the project directory.
	
	2. Run the Streamlit app:
		streamlit run streamlit_app.py

	3. After a few seconds, a browser window will open at:
		http://localhost:8501





🖥️ Web App Usage Instructions
	
	Enter an SMS message in the input box.
	Click "Predict" or similar button.
	The app will display:
	Prediction: Spam or Not Spam





✅ Expected Output Format

	Input: SMS text (e.g., "Congratulations! You've won a prize!")
	Output:
	Label: Spam or Not Spam






🧪 Testing the Model (Optional)
	You can also test directly using Python:
		import pickle
		model = pickle.load(open('model.pkl', 'rb'))
		vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

		text = ["Congratulations! You've won a free ticket."]
		transformed = vectorizer.transform(text)
		prediction = model.predict(transformed)

		print("Spam" if prediction[0] == 1 else "Not Spam")





🌐 Deployment 
	
	You can deploy the Streamlit app using:
	Streamlit Community Cloud (https://streamlit.io/cloud)
	Render.com
	Heroku (if Flask is preferred)
	AWS / Azure / GCP (for advanced hosting)


	For Streamlit Cloud:

		Push your code to a public GitHub repo.
		Go to streamlit.io/cloud
		Click "New App" > Connect your GitHub > Select the streamlit_app.py

		Deploy!
