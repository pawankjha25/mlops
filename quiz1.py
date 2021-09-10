#!/usr/bin/env python
# coding: utf-8

# In[45]:


# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage.transform import rescale, resize, downscale_local_mean


# In[24]:


digits = datasets.load_digits()
print(f"Total rows/images  in the dataset :{len(digits.data)}")
print(f"Total rows and columns in the dataset :{digits.data.shape}")
print(f"Total rows and columns(with each column size) in the dataset :{digits.images.shape}")
print(f"Shape of each image in the the dataset :{digits.images[-1].shape}")


# In[49]:


image_rescaled_1 = rescale(digits.images, 0.5, anti_aliasing=False)
image_rescaled_2 = rescale(digits.images, 1.5, anti_aliasing=False)
image_rescaled_3 = rescale(digits.images, 2.5, anti_aliasing=False)
print(f"Shape of each scaled-1(0.5 times) image in the the dataset :{image_rescaled_1[-1].shape}")
print(f"Shape of each scaled-2(1.5 times) image in the the dataset-2 :{image_rescaled_2[-1].shape}")
print(f"Shape of scaled image-3(2.5 times) in the the dataset-3 :{image_rescaled_3[-1].shape}")


# In[38]:


print("print Original image")
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)


# In[34]:


print("print scaled scaled-1(0.5 times)")
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, image_rescaled_1, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)


# In[36]:


print("print scaled scaled-1(1.5 times)")
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, image_rescaled_2, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)


# In[37]:


print("print scaled scaled-1(2.5 times)")
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, image_rescaled_3, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)


# In[68]:


# flatten the images
n_samples = len(image_rescaled_1)
data = image_rescaled_1.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target[0:898], test_size=0.5, shuffle=False)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

    
print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")


# In[79]:


# flatten the images
n_samples = len(image_rescaled_1)
data = image_rescaled_1.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)


X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target[0:898], test_size=0.8, shuffle=False)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)
print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")


# In[70]:


# flatten the images
n_samples = len(image_rescaled_1)
data = image_rescaled_1.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)


X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target[0:898], test_size=0.9, shuffle=False)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)
print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")


# In[72]:


# flatten the images
n_samples = len(image_rescaled_2)
data = image_rescaled_2.reshape((n_samples, -1))[0:1797]

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)


X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)
print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")


# In[73]:


# flatten the images
n_samples = len(image_rescaled_2)
data = image_rescaled_2.reshape((n_samples, -1))[0:1797]

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Split data into 80% train and 20% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.8, shuffle=False)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)
print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")


# In[74]:


# flatten the images
n_samples = len(image_rescaled_2)
data = image_rescaled_2.reshape((n_samples, -1))[0:1797]

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Split data into 80% train and 20% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.9, shuffle=False)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)
print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")


# In[83]:


# flatten the images
n_samples = len(image_rescaled_3)
data = image_rescaled_3.reshape((n_samples, -1))[0:1797]

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)
print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")


# In[87]:


# flatten the images
n_samples = len(image_rescaled_3)
data = image_rescaled_3.reshape((n_samples, -1))[0:1797]

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)


X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.8, shuffle=False)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

    
print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")


# In[88]:


# flatten the images
n_samples = len(image_rescaled_3)
data = image_rescaled_3.reshape((n_samples, -1))[0:1797]

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)


X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.9, shuffle=False)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

    
print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




