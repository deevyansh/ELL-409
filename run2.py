import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser()

# Add arguments for each parameter
parser.add_argument('--datapath', type=str, required=True)
parser.add_argument('--num_epochs', type=int, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--lr', type=float, required=True)

# Parse the command-line arguments
args = parser.parse_args()

# Access the values2
datapath = args.datapath
epoch = args.num_epochs
batch_size = args.batch_size
learning_rate = args.lr
epsilon = 0.00001

## Importing the data
df_temp = pd.read_csv(datapath)
training_loss = []
k = 5

## Z-Score Parameters
mean = df_temp['X'].mean()
std = df_temp['X'].std()
df = pd.DataFrame(columns=['X', 'y'])
mean2 = df_temp['y'].mean()
std2 = df_temp['y'].std()

## Finding the outliers
for i in range(len(df_temp)):
    if ((abs((df_temp.loc[i, 'X'] - mean) / std)) < 3 and (abs((df_temp.loc[i, 'y'] - mean2) / std2) < 3)):
        df.loc[len(df)] = df_temp.iloc[i]


## Tranining one epoch
def train(a, b, df):
    total_loss = 0

    arr = np.random.permutation(len(df))
    df = df.iloc[arr].reset_index(drop=True)

    for i in range(0, (int)(len(df) / batch_size + ((len(df) % batch_size) > 0))):
        sum1 = 0
        sum2 = 0
        loss = 0
        counter = 0
        for j in range(i * batch_size, min((i + 1) * batch_size, len(df))):
            x = df.loc[j, 'X']
            y = df.loc[j, 'y']
            counter += 1
            ## Calculating the Loss
            loss = loss + ((((a * x) + b) - y) ** 2)
            sum1 = sum1 + ((((a * x) + b) - y) * x)
            sum2 = sum2 + ((a * x) + b - y)

        ## Gradient Descent
        a = a - ((learning_rate) * (sum1) / counter)
        b = b - ((learning_rate) * (sum2) / counter)

        total_loss = total_loss + (loss)
    training_loss.append(total_loss / (2 * len(df)))
    return a, b, (total_loss / (2 * len(df)))


## Running the Algorithm
a = 0
b = 0

for _ in range(epoch):
    (a, b, loss) = train(a, b, df)
    ## Convergence Criteria
    if (len(training_loss) > k):
        total_loss = 0
        for i in range(1, k + 1):
            total_loss += abs(training_loss[-i] - training_loss[-(i + 1)])
        if (((total_loss) / k) < epsilon):
            break

print("Number of the Epochs used:", len(training_loss))
print("Parameters theta0 and theta1", a, b)
