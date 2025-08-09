import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import joblib
import os

def load_and_prepare_data():
    """Load and prepare the Iris dataset"""
    print("Loading Iris dataset from sklearn...")
    
    # Always use sklearn for reliability
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {iris.target_names}")
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_track_models():
    """Train multiple models and track with MLflow"""
    
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    # Set MLflow experiment
    mlflow.set_experiment("iris_classification")
    
    models_performance = {}
    trained_models = {}
    
    print("\n🚀 Starting model training...")
    
    # Model 1: Logistic Regression
    print("\n1. Training Logistic Regression...")
    with mlflow.start_run(run_name="logistic_regression"):
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log parameters and metrics
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 200)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("test_samples", len(X_test))
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        models_performance["LogisticRegression"] = accuracy
        trained_models["LogisticRegression"] = model
        print(f"   ✅ Accuracy: {accuracy:.4f}")
    
    # Model 2: Random Forest
    print("\n2. Training Random Forest...")
    with mlflow.start_run(run_name="random_forest"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log parameters and metrics
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("test_samples", len(X_test))
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        models_performance["RandomForest"] = accuracy
        trained_models["RandomForest"] = model
        print(f"   ✅ Accuracy: {accuracy:.4f}")
    
    # Model 3: SVM
    print("\n3. Training SVM...")
    with mlflow.start_run(run_name="svm"):
        model = SVC(kernel='rbf', random_state=42, probability=True)  # probability=True for confidence scores
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log parameters and metrics
        mlflow.log_param("model_type", "SVM")
        mlflow.log_param("kernel", "rbf")
        mlflow.log_param("probability", True)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("test_samples", len(X_test))
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        models_performance["SVM"] = accuracy
        trained_models["SVM"] = model
        print(f"   ✅ Accuracy: {accuracy:.4f}")
    
    # Find and save best model
    best_model_name = max(models_performance, key=models_performance.get)
    best_accuracy = models_performance[best_model_name]
    best_model = trained_models[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name} with accuracy: {best_accuracy:.4f}")
    
    # Save best model for production
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.joblib')
    print(f"💾 Best model saved to: models/best_model.joblib")
    
    return models_performance, best_model_name

def save_dataset_for_reference():
    """Save the dataset as CSV for reference"""
    try:
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/iris.csv', index=False)
        print(f"📊 Dataset saved to: data/iris.csv")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
    except Exception as e:
        print(f"⚠️  Warning: Could not save dataset: {e}")

if __name__ == "__main__":
    print("🌸 MLOps Iris Project - Model Training")
    print("=" * 50)
    
    # Save dataset for reference
    save_dataset_for_reference()
    
    # Train and track models
    performance, best_model = train_and_track_models()
    
    print("\n" + "=" * 50)
    print("✅ Training completed successfully!")
    print(f"\nModel Performance Summary:")
    for model_name, accuracy in performance.items():
        print(f"  {model_name}: {accuracy:.4f}")
    
    print(f"\n🎯 Best Model: {best_model}")
    print("\nNext steps:")
    print("1. Run: python predict_api/app.py")
    print("2. Run: mlflow ui (to view experiments)")
    print("3. Test API with: python test_api.py")