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

                            # Specific params for KNN
                            params = {}

                            if "K-Nearest Neighbors Classifier" in st.session_state.selected_models:
                                n_neighbors = st.number_input(
                                    "Número de vecinos para KNN",
                                    min_value=1,
                                    max_value=50,
                                    value=5,
                                    step=1,
                                    help="Selecciona el número de vecinos para el clasificador KNN"
                                )
                                params["K-Nearest Neighbors Classifier"] = {"n_neighbors": n_neighbors}

                        else:
                            st.warning("Tipo de algoritmo no soportado")

                # Launch model training
                if st.session_state.selected_models:
                    if st.button("Entrenar modelos"):
                        all_metrics = []

                        for model_name in st.session_state.selected_models:
                            with st.spinner(f"Entrenando {model_name}..."):
                                # Puedes pasar parámetros aquí si alguno lo requiere
                                metrics = train_and_evaluate_model(
                                    task_type,
                                    model_name,
                                    st.session_state.input_df,
                                    st.session_state.features,
                                    st.session_state.target,
                                    model_params=params.get(model_name)
                                )

                                # Asegúrate de que `metrics` sea un dict plano
                                metrics["Modelo"] = model_name
                                all_metrics.append(metrics)

                        # Convertimos todo en DataFrame comparativo
                        comparison_df = pd.DataFrame(all_metrics).set_index("Modelo")
                        st.session_state.metrics_df = comparison_df


                if st.session_state.metrics_df is not None:
                    st.subheader("📈 Comparativa de modelos")
                    styled_df = highlight_best_and_worst_metrics(st.session_state.metrics_df)
                    st.dataframe(styled_df, use_container_width=True)

                    # Gráfico opcional (ejemplo con F1-Score)
                    if "F1-Score" in st.session_state.metrics_df.columns:
                        st.subheader("🔎 Comparación visual (F1-Score)")
                        st.bar_chart(st.session_state.metrics_df["F1-Score"])

if __name__ == "__main__":
    main()