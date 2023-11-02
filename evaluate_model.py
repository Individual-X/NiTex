import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.metrics import confusion_matrix
from io import StringIO

#Command Line parser
parser = argparse.ArgumentParser(description='Benchmark the model against test data')
parser.add_argument('--test-data', default='test.csv', help='Testing Default MNIST Test Data')
args = parser.parse_args()
test_path=args.test_data
test_data= pd.read_csv(test_path)
#droping extra labels
test_X = test_data.drop("label",axis=1)
test_Y=test_data['label']
##Molding pixel value in range of 0-1
test_X=test_X/255.0
#Converting to numpy array to feed CNN
test_X=np.array(test_X)
#Reshaping
test_X = test_X.reshape(-1, 28, 28, 1)
onehot_test_Y = to_categorical(test_Y)
#loading the CNN model
model = keras.models.load_model("model.h5")
print(model.summary())
test_eval = model.evaluate(test_X, onehot_test_Y, verbose=1)
#Plotting the results
accuracy = train_dropout.history['accuracy']
val_accuracy = train_dropout.history['val_accuracy']
loss = train_dropout.history['loss']
val_loss = train_dropout.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show() 

model_summary=model.summary()
sio = StringIO()
model.summary(print_fn=lambda x: sio.write(x + '\n'))
model_summary = sio.getvalue()
sio.close()
# Save the classification report to a text file
with open('output.txt', 'w') as file:
    file.write(test_eval)
    file.write('\n\n')
    file.write(model_summary)







