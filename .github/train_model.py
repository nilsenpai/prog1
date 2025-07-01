from sklearn.linear_model import LogisticRegression
x=[
    [10,7],
    [9,8],
    [2,5],
    [1,6],
]
y=[1,1,0,0]
model=LogisticRegression()
model.fit(x,y)
prediction=model.predict([[4,7]])
print("Prediction(1=pass, 0=fail):",prediction[0])