import pandas as pd
from sklearn.utils.multiclass import type_of_target
import streamlit as st
import time
from utils.utils import train_and_evaluate_model

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
                        else:
                            st.warning("Tipo de algoritmo no soportado")

                # Launch model training
                if st.session_state.selected_models:
                    if st.button("Entrenar modelos"):
                        all_metrics = []

                        for model_name in st.session_state.selected_models:
                            metrics = train_and_evaluate_model(
                                task_type,
                                model_name,
                                st.session_state.input_df,
                                st.session_state.features,
                                st.session_state.target
                            )

                            
                            metrics["Modelo"] = model_name
                            
                            all_metrics.append(metrics)
                            

                        # Concat all metrics into a single DataFrame
                        st.session_state.metrics_df = pd.concat(all_metrics).reset_index().rename(columns={"index": "Clase"})


                if st.session_state.metrics_df is not None:
                    st.subheader("📈 Métricas de evaluación del modelo/s")
                    st.dataframe(
                        st.session_state.metrics_df,
                        use_container_width=True
                    )

if __name__ == "__main__":
    main()