import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# importing dagshub
import dagshub
dagshub.init(repo_owner='JishnudipSaha', repo_name='MLOps-Experiments-with-MLFlow', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/JishnudipSaha/MLOps-Experiments-with-MLFlow.mlflow")

# load dataset
wine = load_wine()
X = wine.data
y = wine.target

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# define the params for the RF model
max_depth = 8
n_estimators = 5

# Mention your experiment below
mlflow.set_experiment('MLOps-Exp-3')


with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred=y_pred)
    
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)
    
    # Creating sonfusion metrix plot
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Metrix')
    
    
    # save plot
    plt.savefig('Confusion-matrix.png')
    
    # log artifacts using ML flow
    mlflow.log_artifact('Confusion-matrix.png')
    mlflow.log_artifact(__file__)
    
    #tags
    mlflow.set_tags({'Author': 'Jishnudip', 'Project': 'Wine Classification'})
    
    # log the model
    mlflow.sklearn.log_model(rf, 'Random-Forest-Model')
    
    print(accuracy)