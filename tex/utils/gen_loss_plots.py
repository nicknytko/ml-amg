import numpy as np
import numpy.linalg as la
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.use('Agg')

def load_data(f):
    df = pd.read_csv(f)
    return df.values[:, 1:]

csv_files = list(filter(lambda x: x.endswith('.csv'), os.listdir('.')))
for filename in csv_files:
    if 'Train' in filename:
        if 'Lloyd' in filename:
            train_baseline_fname = filename
        elif 'ML' in filename:
            train_ml_fname = filename
    elif 'Test' in filename:
        if 'Lloyd' in filename:
            test_baseline_fname = filename
        elif 'ML' in filename:
            test_ml_fname = filename

train_baseline = load_data(train_baseline_fname)
train_ml = load_data(train_ml_fname)
test_baseline = load_data(test_baseline_fname)
test_ml = load_data(test_ml_fname)

plt.figure(figsize=(7,4))
plt.plot(train_baseline[:,0], train_baseline[:,1], label='Lloyd + SA Baseline Loss')
plt.plot(train_ml[:,0], train_ml[:,1], label='ML Loss')
plt.grid()
plt.title('Training Loss')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend()
plt.savefig('train_loss.pdf')

plt.figure(figsize=(7,4))
plt.plot(test_baseline[:,0], test_baseline[:,1], label='Lloyd + SA Baseline Loss')
plt.plot(test_ml[:,0], test_ml[:,1], label='ML Loss')
plt.grid()
plt.title('Testing Loss')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend()
plt.savefig('test_loss.pdf')
