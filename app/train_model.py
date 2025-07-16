import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

df = pandas.read_csv(url, names=columns)
# removes the "Outcome" column in dataframe
X = df.drop("Outcome", axis=1).values
y = df["Outcome"]

# test_size=0.2 - 20% of the data goes to testing, 80% to training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {acc:.2f}")

joblib.dump(model, "model.pkl")
print("✅ Model saved as model.pkl")

model = joblib.load("model.pkl")

# Example input - manual data (higg risk patient):
sample = [[5, 170, 95, 35, 200, 42.0, 1.0, 45]]

prediction = model.predict(sample)
print(f"Prediction: {prediction[0]}")  # 0 = No diabetes, 1 = Diabetes