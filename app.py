import pandas as pd
from sklearn.utils.multiclass import type_of_target
import streamlit as st
import numpy as np
from utils.utils import train_and_evaluate_model
import joblib
import io
from pandas.io.formats.style import Styler

# Configuration
st.set_page_config(
    page_title="🔍 Recomendador de modelos de ML",
    page_icon="🚀",
    layout="wide",
)

# Available models
CLASSIFICATION_MODELS = [
    "Regresión Logística",
    "Random Forest Classifier",
    "Máquinas de Vectores de Soporte (SVM)",
    "K-Nearest Neighbors Classifier",
    "Árbol de Decisión",
    "Gradient Boosting Classifier",
    "AdaBoost Classifier",
    "Naive Bayes",
    "XGBoost Classifier",
    "LightGBM Classifier",
    "CatBoost Classifier"
]

REGRESSION_MODELS = [
    "Regresión Lineal",
    "Random Forest Regressor",
    "Regresión de vectores de soporte (SVR)",
    "Regresión Ridge",
    "Regresión Lasso",
    "Elastic Net",
    "Gradient Boosting Regressor",
    "K-Nearest Neighbors Regressor",
    "Árbol de Decisión Regressor",
    "AdaBoost Regressor",
    "Bayesian Ridge Regression",
    "XGBoost Regressor",
    "LightGBM Regressor",
    "CatBoost Regressor"
]

CLASSIFICATION_TARGETS = [
    "binary",
    "multiclass",
    "multiclass-multioutput",
    "multilabel-indicator"
]

for var in ["uploaded_file", "input_df", "features", "target", "metrics_df", "selected_models"]:
    if var not in st.session_state:
        st.session_state[var] = None

def highlight_best_and_worst_metrics(df: pd.DataFrame) -> pd.io.formats.style.Styler:

    # Metrics where the value high is better
    max_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'R2']
    # Metrics where the value low is better
    min_metrics = ['MSE', 'RMSE', 'MAE']

    styles = pd.DataFrame('', index=df.index, columns=df.columns)

    for col in df.columns:
        if col in max_metrics:
            max_val = df[col].max()
            styles[col] = df[col].apply(lambda x: 'background-color: lightgreen' if x == max_val else '')
        elif col in min_metrics:
            min_val = df[col].min()
            styles[col] = df[col].apply(lambda x: 'background-color: lightgreen' if x == min_val else '')

    return df.style.apply(lambda _: styles, axis=None)



def main():
    st.title("🔍 Recomendador de modelos de Machine Learning")
    st.markdown("""
    ### Sube tu dataset y recibe sugerencias automáticas de modelos de ML
    1. Sube un archivo CSV
    2. Selecciona las variables predictoras y la variable objetivo
    3. Recibe recomendaciones y entrena el modelo
    """)

    # File upload section
    st.session_state.uploaded_file = st.file_uploader(
        "Selecciona un archivo CSV",
        type=["csv"],
        help="Tamaño máximo: 200MB"
    )

    if st.session_state.uploaded_file is not None:
        try:
            st.session_state.input_df = pd.read_csv(st.session_state.uploaded_file)
            st.success("✅ ¡Dataset cargado correctamente!")
        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {str(e)}")


    if st.session_state.input_df is not None:
        with st.container(border=True, key="data_analysis_container"):
            st.subheader("📊 Análisis del Dataset y Selección de Modelo")
            # Data preview
            with st.expander("Vista previa (primeras 10 filas)"):
                st.dataframe(
                    st.session_state.input_df.head(10),
                    use_container_width=True
                )

            input_df_columns = st.session_state.input_df.columns.tolist()

            # Feature selection
            st.session_state.features = st.multiselect(
                "Selecciona las variables predictoras",
                options=input_df_columns
            )

            if len(st.session_state.features) > 0:
                # Target selection
                target_options = [
                    col for col in input_df_columns
                    if col not in st.session_state.features
                ]
                st.session_state.target = st.selectbox(
                    "Selecciona la variable objetivo",
                    options=target_options
                )
                if st.session_state.target:
                    # Task type detection
                    y = st.session_state.input_df[
                        st.session_state.target
                        ].dropna()
                    task_type = "Clasificación" if type_of_target(y) in CLASSIFICATION_TARGETS else "Regresión"

                    # Model recommendations
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(
                            f"**Tipo de algoritmo:** `{task_type.capitalize()}`"
                        )

                    with col2:
                        if task_type == "Clasificación":
                            models = CLASSIFICATION_MODELS
                        elif task_type == "Regresión":
                            models = REGRESSION_MODELS
                        else:
                            models = []

                        if models:
                            st.session_state.selected_models = st.multiselect(
                                "Modelos disponibles",
                                options=models
                            )

                            if "K-Nearest Neighbors Classifier" in st.session_state.selected_models or "K-Nearest Neighbors Regressor" in st.session_state.selected_models:
                                n_neighbors = st.number_input(
                                    "Número de vecinos para KNN",
                                    min_value=1,
                                    max_value=50,
                                    value=5,
                                    step=1,
                                    help="Selecciona el número de vecinos para el clasificador KNN"
                                )
                                param = n_neighbors

                        else:
                            st.warning("Tipo de algoritmo no soportado")

                # Launch model training
                if st.session_state.selected_models:
                    if st.button("Entrenar modelos"):
                        all_metrics = []
                        all_models = {}
                        for model_name in st.session_state.selected_models:
                            with st.spinner(f"Entrenando {model_name}..."):
                                
                                metrics, trained_model = train_and_evaluate_model(
                                    task_type,
                                    model_name,
                                    st.session_state.input_df,
                                    st.session_state.features,
                                    st.session_state.target,
                                    n_neighbors=param
                                )


                                metrics["Modelo"] = model_name
                                all_metrics.append(metrics)
                                all_models[model_name] = trained_model

                     
                        st.session_state.metrics_df = pd.DataFrame(all_metrics).set_index("Modelo")
                        st.session_state.trained_models = all_models


                if st.session_state.metrics_df is not None:
                    st.subheader("📈 Comparativa de modelos")
                    styled_df = highlight_best_and_worst_metrics(st.session_state.metrics_df)
                    st.dataframe(styled_df, use_container_width=True)

                    if task_type == "Clasificación":
                        st.subheader("📉 Curvas ROC por modelo (multiclase soportado)")
                        from sklearn.preprocessing import label_binarize
                        from sklearn.metrics import roc_curve, auc
                        import matplotlib.pyplot as plt
                        import numpy as np

                        y_test = st.session_state.y_test
                        X_test = st.session_state.X_test
                        class_names = list(set(y_test))

                        y_test_bin = label_binarize(y_test, classes=class_names)
                        n_classes = y_test_bin.shape[1]

                        # Crear columnas para mostrar 2 gráficos por fila
                        cols = st.columns(2)
                        col_idx = 0

                        for model_name, model in st.session_state.trained_models.items():
                            if hasattr(model, "predict_proba"):
                                y_score = model.predict_proba(X_test)

                                if y_score.shape[1] == 1:
                                    st.warning(f"⚠️ {model_name} devuelve solo una clase en predict_proba.")
                                    continue

                                fpr = dict()
                                tpr = dict()
                                roc_auc = dict()

                                for i in range(n_classes):
                                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                                    roc_auc[i] = auc(fpr[i], tpr[i])

                                all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
                                mean_tpr = np.zeros_like(all_fpr)
                                for i in range(n_classes):
                                    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                                mean_tpr /= n_classes

                                fig, ax = plt.subplots(figsize=(4, 3))
                                ax.plot(all_fpr, mean_tpr, color='navy',
                                        label=f"Macro promedio (AUC = {auc(all_fpr, mean_tpr):.2f})", lw=2)

                                for i in range(n_classes):
                                    ax.plot(fpr[i], tpr[i],
                                            lw=1, label=f"Clase {class_names[i]} (AUC = {roc_auc[i]:.2f})")

                                ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
                                ax.set_xlabel("Tasa de falsos positivos")
                                ax.set_ylabel("Tasa de verdaderos positivos")
                                ax.set_title(f"Curvas ROC - {model_name}")
                                ax.legend(loc="lower right", fontsize="small")

                                # Mostrar el gráfico en la columna correspondiente
                                with cols[col_idx]:
                                    st.pyplot(fig)

                                col_idx = (col_idx + 1) % 2  # Alternar entre columna 0 y 1

                            else:
                                st.warning(f"⚠️ {model_name} no tiene método `predict_proba()`.")

                            
                    st.subheader("📥 Descargar modelos entrenados")
                    for model_name, model in st.session_state.trained_models.items():
                        buffer = io.BytesIO()
                        joblib.dump(model, buffer)
                        buffer.seek(0)
                        st.download_button(
                            label=f"Descargar {model_name}",
                            data=buffer,
                            file_name=f"{model_name.replace(' ', '_').lower()}.pkl",
                            mime="application/octet-stream"
                        ) 



if __name__ == "__main__":
    main()