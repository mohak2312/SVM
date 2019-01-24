import pandas as pd                                                             # Library or managing the dataset
import numpy as np                                                              
from sklearn import preprocessing                                               # Library for preprocessing the data
from sklearn.model_selection import train_test_split                            # Library for sliting the data into rain and test set
from sklearn.svm import SVC                                                     # Library for Support vector machine classification
from sklearn import metrics             
import scikitplot as skplt
import matplotlib.pyplot as plt                                                 # Library for ploting the graph
from sklearn.metrics import recall_score,precision_score,accuracy_score         # Library for calculating accuracy, precision and recall
from numpy import linalg as LA                                                  # Libray for taking magnitude
import random as ra                                                             # Library for generating random values for feature selection

def get_dataset():                                                              # Create dataset 
    data=[]
    with open("spambase.data",'r') as file:                                     # read the spambase file to extract the feaures and labels
        d1=file.read()
        d=d1.splitlines()
        for values in d:
            x=values.split(",")
            x=[float(i) for i in x]
            data.append(x)
    data_frame= pd.DataFrame(data)
    y=data_frame[len(data[0])-1].values.tolist()                                # create a feature dataset for samples
    X=data_frame.loc[:,0:len(data[0])-2].values.tolist()                        # create a lable dataset for samples
    return X,y,data

def split_train_test(X,y,data):                                                 # function for spliting the data into half training and half test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=int(len(data)/2), train_size=len(data)-int(len(data)/2),random_state=42)

    return  X_train, X_test, y_train, y_test

def preproccesing( X_train, X_test, y_train, y_test):                           # finction for preproccesing
    X_train=np.array(X_train)                                                   # Convert into numy array
    scaler_train = preprocessing.StandardScaler().fit(X_train)                  # use fit method with standardscaler to do preproccesing
    X_train_scaled=scaler_train.transform(X_train)                              # scaled train dataset
    X_test_scaled=scaler_train.transform(X_test)                                # scaled test dataset using mean and standard deviation of train samples
    return X_train_scaled,X_test_scaled

def SVM_training_testing(X_train_scaled,X_test_scaled):                         # function for training and testing the dataset
    clf=SVC(kernel='linear',probability=True)                                   # use SVC class with linear kernel and probablity
    train=clf.fit(X_train_scaled,y_train)                                       # use fit method for training
    predict=clf.predict_proba(X_test_scaled)                                    # predict the probbility and class[0 or 1]
    predict_lable=clf.predict(X_test_scaled)                                    # predict the oputput of test data
    return predict,predict_lable,clf

def cal_results(y_test,predict_lable):                                          # Calculte the accuracy, precision and recall for predicted data of test set
    Precision = precision_score(y_test,predict_lable)
    Recall = recall_score(y_test,predict_lable)
    Accuracy = accuracy_score(y_test,predict_lable)
    return Precision,Recall,Accuracy 

def plot_roc_curve(y_test,predict):                                             # function for ploting roc curve for test data     
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predict[:,1])
    plt.xlabel('False Positive Rate')                                           # set x lable of graph
    plt.ylabel('True Positive Rate')                                            # set y lable of graph
    z=plt.plot(fpr,tpr)                                                         # plot the graph for fpr vs tpr
    m=plt.savefig("exp_1.jpeg")                                                 # save the graph
    plt.show()                                                                  # show the graph

def cal_weight_vector(clf):                                                     # function for calculating weight vector
    alpha=clf.coef_                                                             # alpha coefficent for support vector
    support_vector=clf.support_vectors_                                         # generate the support vectors
    alpha_a=np.array(alpha[0][:, None])
    support_vectors=np.array(np.transpose(support_vector))                      
    weight=support_vectors*alpha_a                                              # calculate the weight vector
    weights=LA.norm(weight,axis=1)                                              # take the magnitude of weight vector
    return weights


X,y,data=get_dataset()                                                          # get dataset from file
X_train, X_test, y_train, y_test= split_train_test(X,y,data)                    # split datset into train set and test set
X_train_scaled,X_test_scaled=preproccesing( X_train, X_test, y_train, y_test)   # scaled the train and test set
print("|----------------- Experiment 1 -----------------|\n")
predict,predict_lable,clf=SVM_training_testing(X_train_scaled,X_test_scaled)    # train the svm on training set and predict the output on test set
Precision,Recall,Accuracy =cal_results(y_test,predict_lable)                    # calculate the precision,recall and accuracy
print("Precision:",Precision)
print("Recall:",Recall)
print("Accuracy:",Accuracy)
plot_roc_curve(y_test,predict)                                                  # plot the roc curve for test data


print("|----------------- Experiment 2 -----------------|\n")
weights=cal_weight_vector(clf)
acc_h_weight=[]
for K in range(2,len(X_train_scaled[0])+1):                                     # select the highest weighted features from 2 to 57                                  
    highest_w=np.sort(np.argpartition(weights,-K)[-K:])
    if(K==5):
        print("Top 5 feature with highest weight :")
        for i in highest_w:
            print(i,weights[i])
    New_train_scaled= X_train_scaled[:,highest_w]                               # create new data set according to new features
    New_test_scaled= X_test_scaled[:,highest_w]
    Pred,pred_l,clf_1=SVM_training_testing(New_train_scaled,New_test_scaled)    # train and test on new datase
    P,R,A=cal_results(y_test,pred_l)                                            # calculate the accuracy
    acc_h_weight.append(A)
plt.ylabel('Accuracy')                                                          # set x lable of graph
plt.xlabel('Number of features')                                                # set y lable of graph
plt.xticks(np.arange(0, 70, 10))                                                # set the x tickes
plt.yticks(np.arange(0.5, 1.2, 0.05))                                           # set the y tickes
plt.plot(acc_h_weight)                                                          # ploat the accuracy on graph
plt.legend(['Feature Selection'], loc='lower right')   
plt.savefig("exp_2.jpeg")                                                       # save the graph
plt.show()                                                                      # plot the graph
a=np.argmax(np.array(acc_h_weight))                                             # find the maximum accuracy for numer of feature
print(a,acc_h_weight[a])
print(acc_h_weight)

print("|----------------- Experiment 3 -----------------|\n")
acc_r_weight=[]
for K in range(2,len(X_train_scaled[0])+1):                                     # generate random feature for training and testing
    random_f=ra.sample(range(0,57),K)
    #print(random_f)
    random_w=np.sort(random_f)                                                  # sort the generated feature
    New_train_scaled= X_train_scaled[:,random_w]                                # create training and testing set
    New_test_scaled= X_test_scaled[:,random_w]
    Pred,pred_l,clf_1=SVM_training_testing(New_train_scaled,New_test_scaled)    # train and predict the output for test data
    P,R,A =cal_results(y_test,pred_l)                                           # calculate the accuracy
    acc_r_weight.append(A)
plt.ylabel('Accuracy')                                                          # set x lable of graph
plt.xlabel('Number of features')                                                # set y lable of graph
plt.xticks(np.arange(0, 70, 10))
plt.yticks(np.arange(0.5, 1.2, 0.05))
plt.plot(acc_r_weight)                                                          # ploat the accuracy on graph
plt.legend(['Random features'], loc='lower right')   
plt.savefig("exp_3.jpeg")                                                       # Save the graph
plt.show()                                                                      # show the graph 
print(acc_r_weight)

plt.ylabel('Accuracy')                                                          # set x lable of graph
plt.xlabel('Number of features')                                                # set y lable of graph
plt.xticks(np.arange(0, 70, 10))
plt.yticks(np.arange(0.5, 1.2, 0.05))
plt.plot(acc_h_weight)                                                          # ploat the accuracy on graph for features selection 
plt.plot(acc_r_weight)                                                          # ploat the accuracy on graph for randomly generated features   
plt.legend(['Feature Selection','Random features'], loc='lower right')   
plt.savefig("exp_4.jpeg")                                                       # save the graph
plt.show()                                                                      # show the graph

