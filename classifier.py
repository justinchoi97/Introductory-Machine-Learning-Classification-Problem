
'''

Scaffolding code for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
Write a main function that calls the different functions to perform the required tasks 
and repeat your experiments.


'''
import numpy as np    # numpy library

from sklearn.naive_bayes import GaussianNB    #Gaussian Naive Baysean Classifier library

from sklearn.model_selection import train_test_split # for spliting data

from sklearn import neighbors   #KNN library

from sklearn import tree # Decision tree library

from sklearn import svm # support vector machine library

from sklearn.ensemble import RandomForestClassifier # random forest classifer used for DT

from sklearn import model_selection 

import matplotlib.pyplot as plt
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
#    return [ (1234567, 'Ada', 'Lovelace'), (1234568, 'Grace', 'Hopper'), (1234569, 'Eva', 'Tardos') ]
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def prepare_dataset(dataset_path):
    '''  
    Read a comma separated text file where 
	- the first field is a ID number 
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued

    Return two numpy arrays X and y where 
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
	X,y
    '''
    x = []      # create emptylists to store temporary attributes list
    x1 = []
    dic = {'M':1, 'B':0}    # dictionary for M or B
    with open(dataset_path, "rt") as f: 
        for line in f:  # line by line read the comma seperate text file 
            x.append(line.split(','))   # split the strings by comma and append to the list
            x1.append(dic[line.split(',')[1]])  #same thing but only store target value

    X = np.array(x) # convert to np array
    X = np.delete(X,np.s_[1], axis = 1) # remove target attribute column
    X = np.delete(X,np.s_[0], axis = 1).astype(np.float)    # remove ID attribute column
    y = np.array(x1)    

    return X,y        
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NB_classifier(X_training, y_training):
    '''  
    Build a Naive Bayes classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    
    model = GaussianNB()    # initiate a Gaussian Naive Baysean classifier model object
    model.fit(X_training, y_training)   # fit the data accordingly to the model
    final_score = (1-model_selection.cross_val_score(model,X_training,y_training,cv = 10,scoring = 'accuracy')).mean()
    print("cross-validated score with best k:",final_score)
    
    return model
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DT_classifier(X_training, y_training,plotting = False, reporting = False):
    '''  
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    k_range = list(range(1,10))
    k_scores = [1000]
    best_k = 0
    for k in k_range:
        #clf = tree.DecisionTreeClassifier(max_depth = k)
        clf = RandomForestClassifier(max_depth = k)
        scores = model_selection.cross_val_score(clf, X_training,y_training, cv = 10, scoring = 'accuracy')
        
        if ((1-scores).mean()) <= min(k_scores):
            best_k = k
            
        k_scores.append(round((1-scores).mean(),5))
    
    if reporting:
        print("Best K_value =", best_k)
    k_scores.remove(1000)
    if plotting: 
        plt.plot(k_range,k_scores)
        plt.xlabel("Value of k for knn")
        plt.ylabel("Cross-validated ERROR")
        plt.show()
      
    #DT=tree.DecisionTreeClassifier(max_depth = best_k) 
    DT = RandomForestClassifier(max_depth = best_k)
    DT.fit(X_training,y_training)
    
    final_score = (1-model_selection.cross_val_score(DT,X_training,y_training,cv = 10,scoring = 'accuracy')).mean()
    if reporting:
        print("cross-validated score with best k:",final_score)
        
    return DT

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NN_classifier(X_training, y_training,plotting = False, reporting = False):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    k_range = list(range(1,100))
    k_scores = [1000]
    best_k = 0
    for k in k_range:
        clf = neighbors.KNeighborsClassifier(n_neighbors = k)
        scores = model_selection.cross_val_score(clf, X_training,y_training, cv = 10, scoring = 'accuracy')
        
        if ((1-scores).mean()) <= min(k_scores):
            best_k = k
            
        k_scores.append(round((1-scores).mean(),5))
    
    if reporting:
        print("Best K_value =", best_k)
    k_scores.remove(1000)
    if plotting: 
        plt.plot(k_range,k_scores)
        plt.xlabel("Value of k for knn")
        plt.ylabel("Cross-validated accuracy")
        plt.show()
        
    knn = neighbors.KNeighborsClassifier(n_neighbors = best_k)
    knn.fit(X,y)
    
    final_score = (1-model_selection.cross_val_score(knn,X_training,y_training,cv = 10,scoring = 'accuracy')).mean()
    if reporting:
        print("cross-validated score with best k:",final_score)
    
    return knn
        
    

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SVM_classifier(X_training, y_training, plotting = False, reporting = False):
    '''  
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    k_range = list(np.linspace(0.01,1,20))
    #k_range = list([1,10,20,30,40,50])
    k_scores = [1000]
    best_k = 0
    for k in k_range:
        clf = svm.SVC(kernel = 'linear', C = k)
        scores = model_selection.cross_val_score(clf, X_training,y_training, cv = 10, scoring = 'accuracy')
        
        if ((1-scores).mean()) <= min(k_scores):
            best_k = k
            
        k_scores.append(round((1-scores).mean(),5))
    
    if reporting:
        print("Best K_value =", best_k)
    k_scores.remove(1000)
    if plotting: 
        plt.plot(k_range,k_scores)
        plt.xlabel("Value of k for gamma")
        plt.ylabel("Cross-validated error")
        plt.show()
        
    SVM =  svm.SVC(kernel = 'linear', C = best_k)
    SVM.fit(X_training,y_training)
    
    final_score = (1-model_selection.cross_val_score(SVM,X_training,y_training,cv = 10,scoring = 'accuracy')).mean()
    if reporting:
        print("cross-validated score with best k:",final_score)
    
    return SVM
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    
    [X,y] = prepare_dataset("medical_records.data")
    
    X_training, X_testing, y_training, y_testing = train_test_split(X, y, test_size=0.2)
    clf = build_NB_classifier(X_training,y_training)
    #clf = build_NN_classifier(X_training,y_training,plotting = True, reporting = True)
    #clf = build_DT_classifier(X_training,y_training,plotting = True, reporting = True)
    #clf = build_SVM_classifier(X_training,y_training,plotting = True, reporting = True)
    

    # the assignment sheet asks for validation error (which is 1 - validation accuracy)
    
    
    # trainig prediction error
    training_error = 1 - clf.score(X_training, y_training)
    
    # testing prediction error
    testing_error = 1 - clf.score(X_testing, y_testing)
    
