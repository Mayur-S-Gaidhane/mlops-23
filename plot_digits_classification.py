"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Standard scientific Python imports

from utils import prepro_data , split_train_dev_test , train_model , read_digits , predict_and_eval

# 1. Get the dataset
X,y = read_digits()

# 2. Data splitiing into train = 60 , test = 20 and dev = 20 .

X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(X, y, test_size=0.4, dev_size=0.5)

# 3. Data Preprocessing 

X_train = prepro_data(X_train)
X_dev   = prepro_data(X_dev)
X_test  = prepro_data(X_test)


# 4. Model Training

model = train_model(X_train, y_train ,model_type='svm')

# 5. Predict and Evaluate the model 

result = predict_and_eval(model, X_test, y_test)

print (result)
