
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.style.use('bmh')


# In[2]:



import sys
import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share your notebook.
client_09c657f29b02424fa7490dd3b3e67651 = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='W4VUcZtgE-Ec0Ub6jkoreyGspLNLwSXL_Qfrlb_JN5aX',
    ibm_auth_endpoint="https://iam.eu-gb.bluemix.net/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.eu-geo.objectstorage.service.networklayer.com')

body = client_09c657f29b02424fa7490dd3b3e67651.get_object(Bucket='teamb304kamal-donotdelete-pr-wou516kxtnupfb',Key='diabetes.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_1 = pd.read_csv(body)
df_data_1.head()


# In[ ]:


df_data_1.info()


# In[ ]:


corr=df_data_1.corr()
plt.figure(figsize=(10,4))
sns.heatmap(corr,annot=True,cmap='summer')
plt.show()


# In[ ]:


x=df_data_1.iloc[:,:-1].values # Independant variables
y=df_data_1.iloc[:,-1].values #dependant variables
x.shape,y.shape


# In[ ]:


plt.figure(figsize=(15,6))
plt.boxplot(x,vert =False,labels=['Pregnancies','Glucose','Blood Pressure','Skin Thickness','Insulin','BMI','DPF','Age'],
           patch_artist=True)
plt.show()


# In[ ]:


from sklearn.preprocessing import  StandardScaler,MinMaxScaler
sc=StandardScaler() #z-score
mms=MinMaxScaler() #(0-1)->normalisation


# In[ ]:


x_sc =sc.fit_transform(x)
x_norm=mms.fit_transform(x)


# In[ ]:


fig=plt.figure(figsize=(15,6))
plt.style.use('bmh')

# Without scaling
plt.boxplot(x,vert=False,labels=['Pregnancies','Glucose','Blood Pressure','Skin Thickness','Insulin','BMI','DPF','Age'],patch_artist=True)
plt.title('Without Scaling')
plt.show()

# Normalisation
fig=plt.figure(figsize=(15,6))
plt.boxplot(x_norm,vert=False,labels=['Pregnancies','Glucose','Blood Pressure','Skin Thickness','Insulin','BMI','DPF','Age'],patch_artist=True)
plt.title('Normalisation(0-1)')
plt.show()

# Standard scaling
fig=plt.figure(figsize=(15,6))
plt.boxplot(x_sc,vert=False,labels=['Pregnancies','Glucose','Blood Pressure','Skin Thickness','Insulin','BMI','DPF','Age'],patch_artist=True)
plt.title('Standard Scaling(Z-score)')
plt.show()


# In[ ]:


from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_sc,y,test_size=0.2,random_state=0)
x_train.shape,y_train.shape,x_test.shape,y_test.shape


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


model_log= LogisticRegression(C=10.0) # class
model_knn= KNeighborsClassifier(n_neighbors=3)
model_svm= SVC(kernel='rbf')
model_dt= DecisionTreeClassifier()
model_rf= RandomForestClassifier(n_estimators=100)


# In[ ]:


model_log.fit(x_train,y_train)
model_knn.fit(x_train,y_train)
model_svm.fit(x_train,y_train)
model_dt.fit(x_train,y_train)
model_rf.fit(x_train,y_train)
print('Model trained successfully')


# In[ ]:


y_pred_log=model_log.predict(x_test)
y_pred_knn=model_knn.predict(x_test)
y_pred_svm=model_svm.predict(x_test)
y_pred_dt=model_dt.predict(x_test)
y_pred_rf=model_rf.predict(x_test)


# In[ ]:


print(y_pred_log)
print(y_pred_knn)
print(y_pred_svm)
print(y_pred_dt)
print(y_pred_rf)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report


# In[ ]:


cm_log= confusion_matrix(y_test,y_pred_log)
cm_knn= confusion_matrix(y_test,y_pred_knn)
cm_svm= confusion_matrix(y_test,y_pred_svm)
cm_dt= confusion_matrix(y_test,y_pred_dt)
cm_rf= confusion_matrix(y_test,y_pred_rf)


# In[ ]:


sns.heatmap(cm_log,annot=True,cmap='summer')
plt.title('Logistic Regression')
plt.show()


sns.heatmap(cm_knn,annot=True,cmap='prism')
plt.title('K Nearest Neighbor ')
plt.show()


sns.heatmap(cm_svm,annot=True,cmap='brg',)
plt.title('Support Vector Machine')
plt.show()


sns.heatmap(cm_dt,annot=True,cmap='jet',)
plt.title('Decision Tree')
plt.show()


sns.heatmap(cm_rf,annot=True,cmap='gnuplot',)
plt.title('Random Forest Tree')
plt.show()


# In[ ]:


fig=plt.figure(figsize=(30,18))

plt.subplot(2,3,1)
sns.heatmap(cm_log,annot=True,cmap='summer')
plt.title('Logistic Regression')

plt.subplot(2,3,2)
sns.heatmap(cm_knn,annot=True,cmap='prism')
plt.title('K Nearest Neighbor ')

plt.subplot(2,3,3)
sns.heatmap(cm_svm,annot=True,cmap='brg',)
plt.title('Support Vector Machine')

plt.subplot(2,3,4)
sns.heatmap(cm_dt,annot=True,cmap='jet',)
plt.title('Decision Tree')

plt.subplot(2,3,5)
sns.heatmap(cm_rf,annot=True,cmap='gnuplot',)
plt.title('Random Forest Tree')
plt.show()


# In[ ]:


cr_log=classification_report(y_test,y_pred_log)
cr_knn=classification_report(y_test,y_pred_knn)
cr_svm=classification_report(y_test,y_pred_svm)
cr_dt=classification_report(y_test,y_pred_dt)
cr_rf=classification_report(y_test,y_pred_rf)


# In[ ]:


print("*"*20+'Logistic Regression'+"*"*20)
print(cr_log)

print("*"*20+'K Nearest Neighbor'+"*"*20)
print(cr_knn)

print("*"*20+'Support Vector Machine'+"*"*20)
print(cr_svm)

print("*"*20+'Decision tree'+"*"*20)
print(cr_dt)

print("*"*20+'Random Forest'+"*"*20)
print(cr_rf)


# # Watson Machine Learning Deployment
# #### Work with your WML instance
# #### First, you must import client libraries.

# In[ ]:


from watson_machine_learning_client import WatsonMachineLearningAPIClient


# In[ ]:


wml_credentials ={
  "username": "8ca38a68-61d6-446b-855d-5dd787f30be2",
  "password": "e5050cc5-6071-447e-befc-7642e62ff10e",
  "instance_id": "fc7e6a05-cdd3-465e-baac-000fe7bed127",
  "url": "https://ibm-watson-ml.eu-gb.bluemix.net"
}


# Create the API client.

# In[ ]:


client = WatsonMachineLearningAPIClient(wml_credentials)


# #### Get instance details.

# In[ ]:


import json

instance_details = client.service_instance.get_details()


# ###  Save the model to the WML repository<a id="save"></a>

# Define the model name, author name and email.

# In[ ]:


published_model = client.repository.store_model(model=model_log, meta_props={'name':'Diabetes'},                                                 training_data=x_train, training_target=y_train)


# #### Get information about a specific model in the WML repository.

# In[ ]:


published_model_uid = client.repository.get_model_uid(published_model)
model_details = client.repository.get_details(published_model_uid)

print(json.dumps(model_details, indent=2))


# #### Get information about all of the models in the WML repository.

# In[ ]:


models_details = client.repository.list_models()


# ### Load a model from the WML repository<a id="load"></a>

# In this subsection you will learn how to load a saved model from a specific WML instance.

# In[ ]:


loaded_model = client.repository.load(published_model_uid)


# Make test predictions to check that the model has been loaded correctly.

# In[ ]:


test_predictions = loaded_model.predict(x_test[:5])


# In[ ]:


print(test_predictions)


# As you can see you are able to make predictions, which means that the model has loaded correctly. You have now learned how save to and load the model from the WML repository.
# 
# ### Delete a model from the WML repository
# The code in the following cell deletes a published model from the WML repository. The code is commented out at this stage because you still need the model for deployment.

# Deploy and score data in the IBM Cloud
# Create the online deployment for the published model

# In[ ]:


# client.repository.delete(published_model_uid)


# In[ ]:


#created_deployment = client.deployments.create(published_model_uid, "Health")


# In[ ]:


scoring_endpoint = client.deployments.get_scoring_url(created_deployment)

print(scoring_endpoint)


# In[ ]:


sc.mean_


# In[ ]:


sc.var_


# In[ ]:


x_sc

