from flask import Flask,render_template,url_for,request
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np 
import pandas as pd

app=Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():



	data=pd.read_csv('data/data1.csv',sep=';',encoding="latin-1")
	data=data.drop(['Unnamed: 0'],1)
	
	Y = data["grav"]
	X = data.drop(['grav'],1)
	#X.reshape(-1,1)
	SXtrain, SXtest, Sytrain, Sytest = train_test_split(X, Y, random_state=0)
	logreg = LogisticRegression(multi_class='ovr')
	logreg.fit(SXtrain, Sytrain)
	#cv=CountVectorizer()
	
	
	#print('Logistic Regression score :',logreg.score(SXtest, Sytest))
	if request.method=='POST': 
		larrout=int(request.form['larrout'])
		surf=int(request.form['surf'])
		infra=int(request.form['infra'])
		situ=int(request.form['situ'])
		env1=int(request.form['env1'])		
		place=int(request.form['place'])
		catu=int(request.form['catu'])
		data=[[larrout,surf,infra,situ,env1,place,catu]]
		#pada=pd.DataFrame(data)
		#vect=np.asarray(pada)
		Y_pred = logreg.predict(data)



    
	return render_template('Resultat.html',predction=Y_pred)



if __name__ == '__main__':
	app.run(debug=True)