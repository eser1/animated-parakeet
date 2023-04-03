import pandas as pd

#importing training and test data
training_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
test_ids = test_data["PassengerId"]

#function for remove unnecessary columns from data
def clean(data):
    data = data.drop(["Ticket", "Cabin", "Name", "PassengerId"], axis=1)
    
    columns = ["SibSp", "Parch", "Fare", "Age"]
    
    for col in columns:
        data[col].fillna(data[col].median(), inplace=True)
        
    data.Embarked.fillna("U", inplace=True)
    
    return data

#cleaning unnecessary columns from each data
training_data = clean(training_data)
test_data = clean(test_data)

from sklearn import preprocessing
label = preprocessing.LabelEncoder()

columns = ["Sex", "Embarked"]

#change string data with numerical data (e.g.) male: 0, female:1
for col in columns:
    training_data[col] = label.fit_transform(training_data[col])
    test_data[col] = label.fit_transform(test_data[col])

#importing LogisticRegression model for predictions
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#setting parameters
y = training_data["Survived"]
X = training_data.drop("Survived", axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=45)

clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)

#making prediction
predictions = clf.predict(X_val)

from sklearn.metrics import accuracy_score

accuracy_score(y_val, predictions)

#prinitng our accuracy score
print(accuracy_score(y_val, predictions))

submission_preds = clf.predict(test_data)

#defining keys and values for submission
df = pd.DataFrame({"PassengerId":test_ids.values,
                   "Survived": submission_preds,})

#creating our submission file
df.to_csv("submission.csv", index=False)

