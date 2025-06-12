import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Literal
from sklearn.preprocessing import LabelEncoder
SEED = 42


def clean_data(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    # Remove rows where target is NaN or empty string
    cleaned_df = df[df[target_col].notna() & (df[target_col] != "")]
    return cleaned_df


def find_and_convert_cat_cols(df: pd.DataFrame) -> pd.DataFrame:
    #Identify categorical columns in the DataFrame and convert them to numerical codes.

    # Identify categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Replace categorical columns with numerical codes
    for col in cat_cols:
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes

    return df


def split_data(
        df: pd.DataFrame,
        feature_cols: list,
        target_col: str,
        test_size: float = 0.3,
        random_state: int = SEED
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    # Split the DataFrame into train/test sets.
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


def evaluate_class_model(
        model: object,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> pd.DataFrame:

    from sklearn.metrics import classification_report

    # Generate predictions
    y_pred = model.predict(X_test)

    # Get classification report as a dictionary
    report = classification_report(y_test, y_pred, output_dict=True)

    # Filter out only the classes (exclude 'accuracy', 'macro avg', etc.)
    class_metrics = {
        label: values for label, values in report.items()
        if label not in ['accuracy', 'macro avg', 'weighted avg']
    }

    # Convert to DataFrame
    metrics_df = pd.DataFrame.from_dict(
        class_metrics,
        orient='index'
    )[['precision', 'recall', 'f1-score']]

    return metrics_df


def evaluate_regr_model(
        model: object,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> pd.DataFrame:
    
    from sklearn.metrics import mean_squared_error, r2_score

    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate RMSE and R-squared
    mse = round(mean_squared_error(y_test, y_pred), 2)
    r2 = round(r2_score(y_test, y_pred), 2)

    # Create a DataFrame to hold the metrics
    metrics_df = pd.DataFrame({
        'MSE': [mse],
        'R-squared': [r2]
    })

    return metrics_df


def train_and_evaluate_model(
        task_type: Literal["Clasificación", "Regresión"],
        model_name: str,
        dataset: pd.DataFrame,
        feature_cols: list,
        target_col: str,
        params: dict = None,
    ) -> pd.DataFrame:

    # Clean data
    cleaned_data = clean_data(dataset, target_col)

    # Convert categorical columns to numerical codes
    if task_type == "Regresión":
        dataset = find_and_convert_cat_cols(cleaned_data)

    # Split data
    X_train, X_test, y_train, y_test = split_data(cleaned_data, feature_cols, target_col)

    import streamlit as st
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test

    metrics_df = pd.DataFrame()

    match task_type:
        case "Clasificación":
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)
            match model_name:
                case "Regresión Logística":
                    from models.classification.models import (
                        train_logistic_regression_model
                    )
                    model = train_logistic_regression_model(X_train, y_train)
                case "Random Forest Classifier":
                    from models.classification.models import (
                        train_random_forest_model
                    )
                    model = train_random_forest_model(X_train, y_train)
                case "Máquinas de Vectores de Soporte (SVM)":
                    from models.classification.models import (
                        train_support_vector_machine
                    )
                    model = train_support_vector_machine(X_train, y_train)
                case "K-Nearest Neighbors Classifier":
                    from models.classification.models import (
                        train_knn_classifier
                    )
                    model = train_knn_classifier(X_train, y_train, params.get('n_neighbors'))
                case "Árbol de Decisión":
                    from models.classification.models import (
                        train_decision_tree_classifier
                    )
                    model = train_decision_tree_classifier(X_train, y_train)
                case "Gradient Boosting Classifier":
                    from models.classification.models import (
                        train_gradient_boosting_classifier
                    )
                    model = train_gradient_boosting_classifier(X_train, y_train)
                case "AdaBoost Classifier":
                    from models.classification.models import (
                        train_adaboost_classifier
                    )
                    model = train_adaboost_classifier(X_train, y_train)
                case "Naive Bayes":
                    from models.classification.models import (
                        train_naive_bayes_classifier
                    )
                    model = train_naive_bayes_classifier(X_train, y_train)
                case "XGBoost Classifier":
                    from models.classification.models import (
                        train_xgboost_classifier
                    )
                    model = train_xgboost_classifier(X_train, y_train)
                case "LightGBM Classifier":
                    from models.classification.models import (
                        train_lightgbm_classifier
                    )
                    model = train_lightgbm_classifier(X_train, y_train)
                case "CatBoost Classifier":
                    from models.classification.models import (
                        train_catboost_classifier
                    )
                    model = train_catboost_classifier(X_train, y_train)
                case _:
                    raise ValueError(f"Modelo no soportado: {model_name}")

            y_pred = model.predict(X_test)
            st.session_state.trained_models = st.session_state.get("trained_models", {})
            st.session_state.trained_models[model_name] = model
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            return {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
                "Recall": recall_score(y_test, y_pred, average='macro', zero_division=0),
                "F1-Score": f1_score(y_test, y_pred, average='macro', zero_division=0)
            }, model

        case "Regresión":
            match model_name:
                case "Regresión Lineal":
                    from models.regression.models import (
                        train_linear_regression
                    )
                    model = train_linear_regression(X_train, y_train)
                case "Random Forest Regressor":
                    from models.regression.models import (
                        train_random_forest
                    )
                    model = train_random_forest(X_train, y_train)
                case "Regresión de vectores de soporte (SVR)":
                    from models.regression.models import (
                        train_support_vector_regression
                    )
                    model = train_support_vector_regression(X_train, y_train)
                case "Regresión Ridge":
                    from models.regression.models import (
                        train_ridge_regression
                    )
                    model = train_ridge_regression(X_train, y_train)
                case "Regresión Lasso":
                    from models.regression.models import (
                        train_lasso_regression
                    )
                    model = train_lasso_regression(X_train, y_train)
                case "Elastic Net":
                    from models.regression.models import (
                        train_elastic_net
                    )
                    model = train_elastic_net(X_train, y_train)
                case "Gradient Boosting Regressor":
                    from models.regression.models import (
                        train_gradient_boosting_regressor
                    )
                    model = train_gradient_boosting_regressor(X_train, y_train)
                case "K-Nearest Neighbors Regressor":
                    from models.regression.models import (
                        train_knn_regressor
                    )
                    model = train_knn_regressor(X_train, y_train, params.get('n_neighbors'))
                case "Árbol de Decisión Regressor":
                    from models.regression.models import (
                        train_decision_tree_regressor
                    )
                    model = train_decision_tree_regressor(X_train, y_train)
                case "AdaBoost Regressor":
                    from models.regression.models import (
                        train_adaboost_regressor
                    )
                    model = train_adaboost_regressor(X_train, y_train)
                case "Bayesian Ridge Regression":
                    from models.regression.models import (
                        train_bayesian_ridge_regression
                    )
                    model = train_bayesian_ridge_regression(X_train, y_train)
                case "XGBoost Regressor":
                    from models.regression.models import (
                        train_xgboost_regressor
                    )
                    model = train_xgboost_regressor(X_train, y_train)
                case "LightGBM Regressor":
                    from models.regression.models import (
                        train_lightgbm_regressor
                    )
                    model = train_lightgbm_regressor(X_train, y_train)
                case "CatBoost Regressor":
                    from models.regression.models import (
                        train_catboost_regressor
                    )
                    model = train_catboost_regressor(X_train, y_train)
                case _:
                    raise ValueError(f"Modelo no soportado: {model_name}")

            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            import numpy as np
            st.session_state.trained_models = st.session_state.get("trained_models", {})
            st.session_state.trained_models[model_name] = model
            y_pred = model.predict(X_test)
            return {
                "MAE": mean_absolute_error(y_test, y_pred),
                "MSE": mean_squared_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "R2": r2_score(y_test, y_pred),
                
            }, model

    return metrics_df