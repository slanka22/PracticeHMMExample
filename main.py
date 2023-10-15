import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm

data = pd.read_csv("https://github.com/natsunoyuki/Data_Science/blob/master/gold/gold/gold_price_usd.csv?raw=True")
print(data)
#Get and Print Data

data["datetime"] = pd.to_datetime(data["datetime"])
data["gold_price_change"] = data["gold_price_usd"].diff()
#Using full Data

data = data.drop(0, axis=0)
#Dropping first row or else an error will be caused

print(data)

#Rest of everything was just using the code of the article to try and fiture out what was happening,
#will try to read further into hmmlearn
X = data[["gold_price_change"]].values
model = hmm.GaussianHMM(n_components = 3, covariance_type = "diag", n_iter = 50, random_state = 42)
model.fit(X)
Z = model.predict(X)
states = pd.unique(Z)


print("Model Start Probabilities")
print(model.startprob_)
print("\nModel Transtion Probabilities")
print(model.transmat_)
print("\nModel Means")
print(model.means_)
print("\nModel Covariance")
print(model.covars_)

plt.figure(figsize = (15, 10))
plt.subplot(2,1,1)
for i in states:
    want = (Z == i)
    x = data["datetime"].iloc[want]
    y = data["gold_price_usd"].iloc[want]
    plt.plot(x, y, '.')
plt.legend(states, fontsize=16)
plt.grid(True)
plt.xlabel("datetime", fontsize=16)
plt.ylabel("gold price (usd)", fontsize=16)
plt.subplot(2,1,2)
for i in states:
    want = (Z == i)
    x = data["datetime"].iloc[want]
    y = data["gold_price_change"].iloc[want]
    plt.plot(x, y, '.')
plt.legend(states, fontsize=16)
plt.grid(True)
plt.xlabel("datetime", fontsize=16)
plt.ylabel("gold price change (usd)", fontsize=16)
plt.show()