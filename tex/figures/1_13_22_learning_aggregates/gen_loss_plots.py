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

date = 'Jan12_16-58-34'
train_baseline = load_data(f'{date}_nick-arch_Loss_Train_Lloyd_SA.csv')
train_ml = load_data(f'{date}_nick-arch_Loss_Train_ML.csv')
test_baseline = load_data(f'{date}_nick-arch_Loss_Test_Lloyd_SA.csv')
test_ml = load_data(f'{date}_nick-arch_Loss_Test_ML.csv')

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
