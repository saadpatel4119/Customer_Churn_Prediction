import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                    "Logistic Regression": LogisticRegression(),
                    "Decision Tree Classifier": DecisionTreeClassifier(),
                    "Random Forest Classifier": RandomForestClassifier(),
                    "Gradient Boosting Classifier": GradientBoostingClassifier(),
                    "AdaBoost Classifier": AdaBoostClassifier(),
                    "K-Neighbors Classifier": KNeighborsClassifier(),
                    "Naive Bayes": GaussianNB()
            }
            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            ROC_AUC_score = roc_auc_score(y_test, predicted)
            return ROC_AUC_score
            
        except Exception as e:
            raise CustomException(e,sys)