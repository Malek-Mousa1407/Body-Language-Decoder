import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pickle

dataFrame = pd.read_csv('coords.csv')
#
x = dataFrame.drop('class', axis=1) # features
y = dataFrame['class'] # target value

# Spliting the data set into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)

# Creating the ML pipeline with different classification models
pipelines = {
    'lr': make_pipeline(StandardScaler(), LogisticRegression()),
    'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}

trainedModel = {}

# for algo, pipeline in pipelines.items():
#     model = pipeline.fit(X_train, y_train)
#     trainedModel[algo] = model

# Training the model using the RidgeClassification Model
model = pipelines['rf'].fit(X_train, y_train)
trainedModel['rf'] = model

# Exporting the model in binary format
with open('bodyLangModel.pkl', 'wb') as f:
    pickle.dump(trainedModel['rf'], f)
