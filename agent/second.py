import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
# 1. 데이터 로드
df = pd.read_csv('Historical_Product_Demand.csv')

# 2. 전처리 (날짜 및 숫자 변환)
df['Order_Demand'] = df['Order_Demand'].str.replace('(', '').str.replace(')', '').astype(float)
df['Date'] = pd.to_datetime(df['Date'])
df.dropna(inplace=True)
df['Month'] = df['Date'].dt.month

# 3. 데이터 인코딩 (문자를 숫자로)
le = LabelEncoder()
df['Warehouse_n'] = le.fit_transform(df['Warehouse'])
df['Category_n'] = le.fit_transform(df['Product_Category'])
df['Product_n'] = le.fit_transform(df['Product_Code'])

# 4. 특징(Feature) 선택
features = ['Month', 'Warehouse_n', 'Category_n', 'Product_n']
X = df[features]
y = df['Order_Demand']

# 6. 알고리즘 적용: Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, 'logistics_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
