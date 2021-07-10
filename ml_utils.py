from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
#Task3 Adding one more classifier.
from sklearn.svm import SVC

# define a Gaussain NB classifier
clf_g = GaussianNB()
# For Task3 defining another classifier namely SVC.
clf_svc = SVC(kernel='poly', degree=3, max_iter=300000)

# define the class encodings and reverse encodings
classes = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica"}
r_classes = {y: x for x, y in classes.items()}

# function to train and load the model during startup
def load_model():
    # load the dataset from the official sklearn datasets
    X, y = datasets.load_iris(return_X_y=True)

    # do the test-train split and train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf_g.fit(X_train, y_train)
    clf_svc.fit(X_train, y_train)

    # calculate the print the accuracy score
    acc_g = accuracy_score(y_test, clf_g.predict(X_test))
    print(f"GaussianNB Model trained with accuracy: {round(acc_g, 3)}")
    # calculate the print the accuracy score for the second model.
    acc_svc = accuracy_score(y_test, clf_svc.predict(X_test))
    print(f"SVC Model trained with accuracy: {round(acc_svc, 3)}")
    #Task3 We need to decide the best acccuracy model and continue using the best one out of the two
    # using a if else loop to decide the best model.
    if acc_svc >= acc_g:
        print("SVM model has better accuracy")
        #hence we return the svm so the it can be used further.
        return clf_svc
    else:
        print("GaussianNB model has better accuracy")
        #GaussianNB is better if we are in this loop, so returning it for using it further.
        return clf_g

#Here we are assigning the best model of two which we are returning in load_model()so that it can be used in prediction.
clf_best = load_model()
print(clf_best)

# function to predict the flower using the model
def predict(query_data):
    x = list(query_data.dict().values())
    prediction = clf_best.predict([x])[0]
    print(f"Model prediction: {classes[prediction]}")
    return classes[prediction]

# function to retrain the model as part of the feedback loop
def retrain(data):
    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.flower_class] for d in data]
    # fit the classifier again based on the new data obtained
    clf_g.fit(X, y)
