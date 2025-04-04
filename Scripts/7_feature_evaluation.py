from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd

df = pd.read_csv('../Data/data_for_feature_importance.csv')
features = df[['rating_count', 'cost_for_two', 'popularity_score']] 
target = df['rating']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)

importances = rf.feature_importances_ 
feature_names = features.columns 
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)

print("Feature Importances:") 
print(importance_df)

plt.figure(figsize=(8, 6)) 
sns.barplot(x='Importance', y='Feature', data=importance_df) 
plt.title('Feature Importances from RandomForestRegressor') 
plt.xlabel('Importance') 
plt.ylabel('Feature') 
plt.tight_layout() 
plt.savefig('feature_importance.png') 
plt.close()

