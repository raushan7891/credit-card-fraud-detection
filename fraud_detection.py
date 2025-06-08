import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

!pip3 install scikit-learn

pip list

# loading the dataset to a pandas dataframe
credit_card_data=pd.read_csv(r'C:\Users\Raushan\Desktop\Credit card fraud detection\creditcard.csv')

#first 5 rows of the dataset
credit_card_data.head()

#last 5 rows
credit_card_data.tail()

#no. of colums & data type
credit_card_data.info()


#no. of missing values in each column
credit_card_data.isnull().sum

## distribution of legit transaction & fraudulent transaction
credit_card_data['Class'].value_counts()


# separating the data for analysis
#0-> normal transaction,then entire row will be load in this legit variable
#1-> fraudulent transaction,then entire row will be load in this fraud variable

legit=credit_card_data[credit_card_data.Class==0]
fraud=credit_card_data[credit_card_data.Class==1]



# print the shape of two variables
print(legit.shape)
print(fraud.shape)

#statistical measures of the data 
legit.Amount.describe()


fraud.Amount.describe()



#compare the values for both transaction
credit_card_data.groupby('Class').mean()


#dealing with a unbalanced data.
#undersampling: building a sample dataset from original dataset
#containing similar distribution of normal transaction & fraudulent transaction
#randomly take 492 data from normal tranasaction
#we have normal or unimorm distribution of both
# distribution is even 
#taking variable 
legit_sample=legit.sample(n=492)


#concatenating two dataframe
new_dataset=pd.concat([legit_sample,fraud],axis=0)


#checing first 5 rows of dataframe
new_dataset.head()


#checing last 5 rows of dataframe
new_dataset.tail()


#uniformly distributed data
new_dataset['Class'].value_counts()



#now find groupby of these value
new_dataset.groupby('Class').mean()




#Now splitting the data into features & 
# X= store all the features in X
#features is time,V1,V2,amount like that
#axis=1 represent column
#class column drop kar diya
X=new_dataset.drop(columns='Class',axis=1)
Y=new_dataset['Class']
print(X)


print(Y)



#splitting the data into training data & testing data
#for this we are going to use train_test_split function 
#imported from sklearn.model_selection
#need to maintain the 4 variables
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
'''X_train: training features

X_test: testing features

Y_train: training labels

Y_test: testing labels '''
#X is a features
#Y is a class
#0.2-> 20% is testing data & 80% go to training data
#train->80% data
#test-> 20% data
#stratify evenly distributed two classes in both X_train & X_test
#random_state=2



print(X.shape,X_train.shape,X_test.shape)
#successfully splited our data into training data & testing data 


#next step to train our ML model 
#next step is to check the accuracy of our model
#model training:-> Use logistic regression model
#we can also try different model which model going to give better accuracy score
# we use Logistic regression for binary classification problem
model=LogisticRegression(solver='liblinear', max_iter=1000)
# loading LR model to tis particular varaible



#now we need  to train our data
#training the LR model with training data
#for training this particular data you need to mention the function fit 
# this will fit our data to our  LR model
#X_train-> contain all the features of training data
#Y_train-> the corresponding labels is 0 & 1
model.fit(X_train,Y_train)
#fit the X_train data into LR model then we can make some prediction on it




#give X_train value to our model & it will predict what is the class for this value 
#once it predicted it , he will try to compare the values predicted by our model & original value which is present in Y_train
#it will give us accuracy score
X_train_prediction=model.predict(X_train)
#X_train-> predicting the labels for X_train ,haven't given Y_train
#so i will predicting all the labels for all the training data







#then i will compare value predicted by our model->X_train_prediction with the original labels which is store in Y_train 
training_data_accuracy= accuracy_score(X_train_prediction,Y_train)




#it will compare the value & give us the accuracy score
# accuracy score stored in training_data_accuracy variable
#accuracy score of traning data 
#our model seen the X_train data 
print('Accuracy on Training data :' ,training_data_accuracy)





#our model didn't seen the X_test data.
#so that is why the evaluation of test_data is important
X_test_prediction=model.predict(X_test)
#i am predicting labels for X_test






#Accuracy of test data
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
#i am comparing X_test_prediction ,this prediction met by the model Y_test
#Y_test is the real labels for data set
# we have splitted our data into X_test & Y_test
print('Accuracy score on Test Data:' ,test_data_accuracy)




#this accuracy score is very similar to training_data accuracy.

























