import pandas as pd
import argparse
import numpy as np
parser = argparse.ArgumentParser(description="Run the model with specified parameters")

# Add arguments for each parameter
parser.add_argument('--datapath', type=str, required=True, help='Path to the dataset')
parser.add_argument('--num_epochs', type=int, required=True, help='Number of epochs')
parser.add_argument('--lr', type=float, required=True, help='Learning rate')


# Parse the command-line arguments
args = parser.parse_args()

# Access the values
datapath = args.datapath
epoch = args.num_epochs
learning_rate = args.lr
epsilon=0.00001

## Parameters for Z-score
df_temp=pd.read_csv(datapath)
df=pd.DataFrame(columns=['X','y'])
mean=df_temp['X'].mean()
std=df_temp['X'].std()
mean2=df_temp['y'].mean()
std2=df_temp['y'].std()

## Removing the outliers
for i in range (len(df_temp)):
    if((abs((df_temp.loc[i,'X']-mean)/std))<3  and (abs((df_temp.loc[i,'y']-mean2)/std2)<3)):
        df.loc[len(df)]=df_temp.iloc[i]

#print("Final Length",len(df))

def doit(x,y,a,b):
    return np.subtract(np.add((np.multiply(x,a)),b),y)

def train(a,b):
    loss=0
    sum1=0
    sum2=0
    x=np.array(df["X"])
    y=np.array(df["y"])

    ## Loss Calculation
    loss=np.sum(np.pow(doit(x,y,a,b),2))
    sum1=np.sum(np.multiply(doit(x,y,a,b),x))
    sum2=np.sum(doit(x,y,a,b))
    ## Gradient Descent
    a=a-((learning_rate)*(sum1/len(df)))
    b=b-((learning_rate)*(sum2/len(df)))

    return a,b,(loss/(2*len(df)))

prevloss=0
a=0
b=0
for i in range (epoch):
    (a,b,loss)=train(a,b)
    ## Checking the convergence criteria
    if(prevloss!=0 and abs(loss-prevloss)<epsilon):
        epoch=i
        break
    prevloss=loss

print("The final Paramters are ",a,b)
print("Final Epoch Taken", epoch)
