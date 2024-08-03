import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('housing.csv')

data_numeric = data.apply(pd.to_numeric, errors='coerce')
data_filled = data_numeric.fillna(data_numeric.mean())

X = data_filled.drop(columns=['ocean_proximity', 'median_house_value']).values
Y = data['median_house_value'].values.reshape(-1, 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
scaler_Y = StandardScaler()  
Y_scaled = scaler_Y.fit_transform(Y)

decomp_rank = 3  # Decomposition rank
poly_order = 3  # Polynomial order
net_dims = (32, 32)  # Network layer dimensions
reg_lambda_w = 0.01  # Regularization strength for W
reg_lambda_c = 0.01  # Regularization strength for C

model = PolynomialTensorRegression(decomp_rank = 3,
                                   poly_order = 3,
                                   method='cp',
                                   reg_lambda_w = 0.01, 
                                   reg_lambda_c = 0.01) 
model.train_model(X_scaled, Y_scaled)

polynomial = model.get_significant_polynomial()