import pandas as pd

df = pd.read_csv("trainingdataset.csv")
print("Training Data:\n")
print(df)

hypothesis = ['0', '0', '0', '0']

for i in range(len(df)):
    if df['EnjoySport'][i] == 'Yes':
        for j in range(4):
            if hypothesis[j] == '0':
                hypothesis[j] = df.iloc[i, j]
            elif hypothesis[j] != df.iloc[i, j]:
                hypothesis[j] = '?'
print("\nFinal Hypothesis:")
print(hypothesis)
