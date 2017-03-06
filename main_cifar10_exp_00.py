## Make necessary imports
import numpy as np
np.set_printoptions(precision=3, linewidth=200, suppress=True)
from object_recog_datasets import cifar
import pandas as pd

## Step 0, 1: Make and load the CIFAR 10 dataset
# Not needed for random label generation

## Step 2: Find the number of unique class labels (object categories)
# We know it is 10 for CIFAR 10

## Step 3: Make the predictions for test dataset
num_test_images = 300000
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
          'horse', 'ship', 'truck']
predictions_numbers = np.random.randint(0, 9, size=(num_test_images,1)).flatten()
predictions_classes = []
indices = np.arange(1, num_test_images+1)
print('Making random predictions on test data')
for i in range(num_test_images):
    predictions_classes.append(classes[predictions_numbers[i]])
predictions_classes = np.array(predictions_classes)
predictions = np.column_stack((indices, predictions_classes))

## Step 4: Write the generated predictions to csv file
output_file = './data/cifar10/kaggle/submissions/submission_00.csv'
print('Writing the test data to file: %s' %output_file)
column_names = ['id', 'label']
predict_test_df = pd.DataFrame(data=predictions, columns=column_names)
predict_test_df.to_csv(output_file, index=False)

