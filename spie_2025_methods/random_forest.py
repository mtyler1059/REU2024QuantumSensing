import argparse

# Argparse architecture
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('mp', type=int)
parser.add_argument('tp', type=int)

# Parse arguments from command line
args = parser.parse_args()



# Standard imports
import numpy as np
import os
import pandas as pd

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

from datetime import datetime

# In[6]:


training_directory_w_lagging_slash = '/u/project/andreabertozzi/mfloridi/no_noise_testing_suite/count_100_per/'
testing_directory_w_lagging_slash = '/u/project/andreabertozzi/mfloridi/no_noise_testing_suite/count_50_per/'
results_directory = f"/u/project/andreabertozzi/mfloridi/no_noise_testing_suite/results/m{args.mp}_t{args.tp}.txt"

# Obtain numpy array, containing individual units of unknown format
x = np.linspace(500, 3700, 33)

# Convert all unknown objects to integers
center_integers = []
for _ in x:
  center_integers.append(int(_))

# Convert all integers to strings
center_str = []
for _ in center_integers:
  center_str.append(str(_))

# In[7]:

for m in [args.mp]:
  for t in [args.tp]:
    training_directory = f"{training_directory_w_lagging_slash}m{m}_t{t}/"
    testing_directory = f"{testing_directory_w_lagging_slash}m{m}_t{t}/"

    for w in ['10', '30', '50']:
      for tpwn in ['1', '4', '10']:
        for c in center_str:
          name = f"m{m}_t{t}_c{c}_w{w}_tpwn{tpwn}.csv"
          training_file = f'{training_directory}{name}'
          testing_file = f'{testing_directory}{name}'

          training_data = pd.read_csv(training_file)
          testing_data = pd.read_csv(testing_file)

          features_train = training_data.drop('name', axis = 1)
          target_train = training_data['name']
        
          features_test = testing_data.drop('name', axis = 1)
          target_test = testing_data['name']

          rf = RandomForestClassifier()
          rf.fit(features_train, target_train)
        
          y_pred = rf.predict(features_test)
        
          accuracy = accuracy_score(target_test, y_pred)
        
          writing = open(results_directory, "a")
          writing.write(f"m{m} t{t} c{c} w{w} tpwn{tpwn} A{accuracy} \n")
          writing.close()
