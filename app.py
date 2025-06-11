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

for var in ["uploaded_file", "input_df", "features", "target"]:
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

if __name__ == "__main__":
    main()