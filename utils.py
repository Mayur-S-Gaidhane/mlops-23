
# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pdb




# read digit function 

def read_digits():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.images
    return X, y


# Preprocessing data funtion 

def prepro_data(data):
    # flatten the images
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

# Split data into train , test and dev/ validation subset fuction .

def split_train_dev_test(X, y, test_size, dev_size):
    # First, split the data into train and test sets using train test split. 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    # Now, split the remaining data into dev and train sets.
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=dev_size, random_state=1)
    return X_train, X_dev, X_test, y_train, y_dev, y_test
    


def train_model(X_train, y_train, model_type="svm"):
    if model_type == "svm":
        # create a classifier SVM 
        clf = svm.SVC
        model = clf(gamma=0.001)
        # train SVM model 
        model.fit(X_train, y_train)
        return model


# Prediction and Evaluation function of SVM 

def predict_and_eval(model, X_test, y_test):
    # Make predictions using the model
    prediction = model.predict(X_test)

       

    # Calculate evaluation of SVM prediction model
    print(
        f"Classification report for classifier {model}:\n"
        f"{metrics.classification_report(y_test, prediction)}\n"
    )

    # Plot confusion matrix 
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, prediction)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    plt.show()

    # The ground truth and predicted lists
    y_true = []
    y_pred = []
    cm = disp.confusion_matrix

    # and 
    # predictions to the lists
    for gt in range(len(cm)):
        for pred in range(len(cm)):
            y_true += [gt] * cm[gt][pred]
            y_pred += [pred] * cm[gt][pred]

    evalution = print(
                    "Classification report rebuilt from confusion matrix:\n"
                    f"{metrics.classification_report(y_true, y_pred)}\n"
                )

    return prediction , evalution


     




