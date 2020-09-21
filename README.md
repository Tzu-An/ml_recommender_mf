# ml_recommender_mf
Simple implementation of matrix factorization for recommender system

## Declaration

  - Since mf module is just a simple implementation, there is a lot of room for improvements before applying to larger dataset, including:
    - Get better initial guess with SVD
    - Applying better optimization method, like rmsprop
    - Implement optimization part with C to get better performance

## Models

  - MF: An implementation of matrix factorization with gradient descent

  - MFWithNull: With different way to deal with null data from GDMF
    - This model treats 0 as null. Better to shift the ratings before applying this model.
    - This model will ignore nulls and make prediction based on information from real rating history.
    - Most of the results will be converge to some number between upper-bond and lower-bond, and
      the predictions will not seem like adding a random decimal to given record.
    - The predictions would not go too far from the recorded rating for those user with
      nearly no rating history or no preferences.

## Attributes

  - Common:
    - latents (int): dimension of latent features
    - lr (float): learning rate
    - max_iter (int): maximum iteration of gradient descent
    - tol (float): tolerance for early stopping
    - user: user-latent matrix, default = None
    - item: latent-item matrix, default = None

  - Only for MFWithNull:
    - maxn (int or float): desired upper-bond of prediction
    - minn (int or float): desired lower-bond of prediction

# Methods

  - Common:
    - loss(data): returns MSE loss on given data
    - predict(user, item): make prediction of a particular item for a given user
    - predict_user(user): make predictions for a given user
    - predict_all(): make predictions on all users and items
    - fit(data): train model on given data

## When to Use

  - Factorizing rating matrix to get more insights and predictions

## How to Use

### Train model

```python
import mf

data = [[4, 0, 0, 1], [0, 3, 1, 5], [2, 3, 4, 0]]
latent_size = 2
model = mf.MFWithNull(latent_size, maxn=5, minn=1, lr=0.01, max_iter=1000)
model.fit(data)
```

### Predict

```python
model.predict_all()           # To get all predictions of ratings
model.predict_user(1)         # To get all predictions of ratings for user with index=1
model.predict(user=1, item=3) # To get prediction of user number 1 on item number 3 
```
