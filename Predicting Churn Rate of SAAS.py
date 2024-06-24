#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score



# In[11]:


dataset = pd.read_csv("C:\\Users\\msaad\\Downloads\\analytics\\Project\\SAAP dataset\\train.csv")


# In[12]:


dataset

Dataset Preprocessing
# In[13]:


dataset.info()


# In[14]:


dataset.isnull().sum()

There are no null values in the dataset
# In[15]:


dataset.describe()


# In[36]:


from sklearn.preprocessing import LabelEncoder
# Create a LabelEncoder object
encoder = LabelEncoder()

# Encode the categorical variable
dataset['SubscriptionType_en'] = encoder.fit_transform(dataset['SubscriptionType'])
dataset['Gender_en'] = encoder.fit_transform(dataset['Gender'])
dataset['PaymentMethod_en'] = encoder.fit_transform(dataset['PaymentMethod'])
dataset['ContentType_en'] = encoder.fit_transform(dataset['ContentType'])
dataset['SubtitlesEnabled_en'] = encoder.fit_transform(dataset['SubtitlesEnabled'])
dataset['ParentalControl_en'] = encoder.fit_transform(dataset['ParentalControl'])
dataset['MultiDeviceAccess_en'] = encoder.fit_transform(dataset['MultiDeviceAccess'])
dataset['DeviceRegistered_en'] = encoder.fit_transform(dataset['DeviceRegistered'])


print(dataset)


# In[48]:


dataset.info()


# # Calculating Churn rate

# In[37]:


churn=dataset['Churn'].value_counts() 
churn


# In[38]:


# Extract labels (0 for not churn, 1 for churn) and counts (number of customers)
churn_labels = churn.index.to_numpy()  # Convert index to NumPy array for plotting
counts = churn.to_numpy()

# Create bar plot
plt.bar(churn_labels, counts)  # Use churn labels (0 or 1) for x-axis

# Set labels and title
plt.xlabel("Churn (0: No Churn, 1: Churn)")
plt.ylabel("Number of Customers")
plt.title("Distribution of Churned and Non-Churned Customers")

# Display plot
plt.show()


# In[39]:


churned=dataset[dataset['Churn']==1]
not_churn=dataset[dataset['Churn']==0]
churned


# In[81]:


def visualize_feature(column_name):
  # Histogram
  plt.hist(dataset[column_name])
  plt.xlabel(column_name)
  plt.ylabel("Frequency")
  plt.title(f"Histogram of {column_name}")
  plt.show()

  # Box Plot
  plt.boxplot(dataset[column_name])
  plt.xlabel(column_name)
  plt.ylabel("Value")
  plt.title(f"Box Plot of {column_name}")
  plt.show()
visualize_feature('TotalCharges')
visualize_feature('ViewingHoursPerWeek')
visualize_feature('AverageViewingDuration')
visualize_feature('ContentDownloadsPerMonth')
visualize_feature('MonthlyCharges')



# In[41]:


churnrate= len(dataset[dataset['Churn']==1])/dataset['CustomerID'].count()
customer_churnrate=churnrate*100
print(f'customer churnrate is {customer_churnrate} %')


# In[43]:


dataset


# In[46]:


import numpy as np
df= dataset.iloc[:, 0:21]
df
corr_data=df.corr()
corr_data


# In[47]:


plt.figure(figsize=(10, 10))
sns.heatmap(corr_data, annot=True, cmap='coolwarm', vmin=0, vmax=1)


# In[53]:


dataset.info()


# In[153]:


features = dataset.iloc[:,[0,1,9,10,11,13,14,16,21,22,23,24,25,26]].values    
features = features.reshape(-1,14)
features.shape


# In[154]:


target = dataset.iloc[:,20].values      # Taking churn as dependant variable 
#target = target.reshape(-1,1)
target.shape


# In[155]:


# 70% of th data is separated for training and 30% for testing

from sklearn.model_selection import train_test_split
features_train, features_test, target_train, target_test = train_test_split(features,target,test_size=0.3, random_state =0)


# In[156]:


model = LogisticRegression()
model.fit(features_train, target_train)


# In[157]:


prediction = model.predict(features_test)
prediction


# In[158]:


print('Confusion Matrix:',confusion_matrix(target_test, prediction))  


# In[159]:


print('The Accuracy Score of the model is:',accuracy_score(target_test, prediction) )


# In[160]:


import statsmodels.api as sm
from scipy import stats

logistic_regression = sm.Logit(target, features)       
logistic_regression = logistic_regression.fit()
logistic_regression.summary()


# In[162]:


from sklearn.model_selection import cross_validate
predicted = cross_validate(model, features,target, cv = 10) 
# CV is the number of folds we want
test_scoree=predicted['test_score']
print(predicted['test_score'])
np.std(predicted['test_score']) 
                                                           


# In[ ]:


# Extract coefficients
coefficients = model.coef_.ravel()

# Absolute value of coefficients (for heatmap)
abs_coef = np.abs(coefficients)

# Create heatmap
fig, ax = plt.subplots()
heatmap = ax.pcolor(abs_coef.reshape(1, -1), cmap='RdBu', vmin=0, vmax=abs_coef.max())
fig.colorbar(heatmap)

# Set labels and title
ax.set_xticks(np.arange(len(features)))
ax.set_yticks([0])
ax.set_yticklabels(features)
ax.set_title('Logistic Regression Feature Importance (Absolute Coefficients)')
plt.tight_layout()

# Optional: Rotate feature labels for better readability with many features
plt.xticks(rotation=90)
plt.show()

# Churn predictions on the test data
# In[163]:


test = pd.read_csv("C:\\Users\\msaad\\Downloads\\analytics\\Project\\SAAP dataset\\test.csv")
test.info()
test.isnull().sum()


# In[164]:


# Create a LabelEncoder object
encoder = LabelEncoder()

# Encode the categorical variable
test['SubscriptionType_en'] = encoder.fit_transform(test['SubscriptionType'])
test['Gender_en'] = encoder.fit_transform(test['Gender'])
test['PaymentMethod_en'] = encoder.fit_transform(test['PaymentMethod'])
test['ContentType_en'] = encoder.fit_transform(test['ContentType'])
test['SubtitlesEnabled_en'] = encoder.fit_transform(test['SubtitlesEnabled'])
test['ParentalControl_en'] = encoder.fit_transform(test['ParentalControl'])
test['MultiDeviceAccess_en'] = encoder.fit_transform(test['MultiDeviceAccess'])
test['DeviceRegistered_en'] = encoder.fit_transform(test['DeviceRegistered'])


# In[167]:


# Assuming you have a trained model (`model`) and preprocessed test data (`X_test`)
X_test = test.iloc[:,[0,1,9,10,11,13,14,16,21,22,23,24,25,26]].values    
X_test = X_test.reshape(-1,14)
X_test.shape
predictions = model.predict_proba(X_test)[:,1] # Assuming churn is class 1
predictions.shape


# In[168]:


threshold = 0.7  # Assumed probability threshold
customerid=test.iloc[:,[19]].values
customerid.shape

Cutomer Ids Highly likely to churn on the basis of model predictions 
# In[169]:


filtered_customer_ids = customerid[predictions >= threshold]
high_risk_customers = filtered_customer_ids

print("Customer IDs Likely to Churn:")
print(high_risk_customers)


# In[170]:


r=len(high_risk_customers)
print(f'There are about {r} customers who have high probability to leave the subscription services') 


# In[ ]:




