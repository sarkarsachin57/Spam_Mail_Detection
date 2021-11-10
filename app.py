import joblib, re
vectorizer = joblib.load('text_vectorizer.joblib')
model = joblib.load('spam_mail_detector.joblib')


def cleaner(text):
    text = re.sub('[^A-Za-z]',' ',text)
    return text.lower()

def prediction(msg):
    msg = cleaner(msg)
    msg = vectorizer.transform([msg]).toarray()
    if model.predict(msg) == 1:
        return "The mail you have entered is a spam mail"
    else:
        return "The mail you have entered is a proper mail"

from flask import Flask,render_template,redirect,url_for,request

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home(prediction='Your Results will show here.'):
    return render_template('prediction.html',text=prediction)

@app.route('/predict',methods=['POST'])
def predict():
    for x in request.form.values():
        msg = x
    is_spam = prediction(msg)    
    return render_template('prediction.html',text=is_spam)


app.run()