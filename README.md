# 🔎 ML Model Trainer: Clasificación y Regresión con Streamlit

[Accede a la app en producción](https://modelchooser.streamlit.app/)

Este proyecto permite cargar un conjunto de datos, seleccionar características, entrenar múltiples modelos de **clasificación** o **regresión**, y evaluar su rendimiento de forma interactiva mediante **Streamlit**.

---

## 🚀 Características

- ✅ Limpieza automática de datos (eliminación de valores vacíos en la columna objetivo).
- 🔁 Codificación automática de variables categóricas.
- 📊 División en conjuntos de entrenamiento y prueba.
- 🤖 Entrenamiento de múltiples modelos con diferentes algoritmos.
- 📈 Evaluación automática de los modelos con métricas clave.
- 💾 Almacenamiento de los modelos entrenados en `st.session_state` para su descarga.
- 🧪 Visualización interactiva con **Streamlit**.

---

## 🧠 Modelos Soportados

### Clasificación
- Regresión Logística
- Random Forest Classifier
- Máquinas de Vectores de Soporte (SVM)
- K-Nearest Neighbors Classifier
- Árbol de Decisión
- Gradient Boosting Classifier
- AdaBoost Classifier
- Naive Bayes
- XGBoost Classifier
- LightGBM Classifier
- CatBoost Classifier

### Regresión
- Regresión Lineal
- Ridge Regression
- Lasso Regression
- Elastic Net
- Soporte Vectorial (SVR)
- Random Forest Regressor
- Árbol de Decisión Regressor
- AdaBoost Regressor
- Gradient Boosting Regressor
- KNN Regressor
- Bayesian Ridge Regression
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor

## ▶️ Cómo Ejecutar Localmente

1. Clona el repositorio:

```bash
git clone https://github.com/eloydsdlh/ModelChooser.git
cd ModelChooser
```

2. Instala dependencias:

```bash
pip install -r requirements.txt
```

3. Ejecuta la aplicación:

```bash
streamlit run app.py
```

## 📊 Métricas Reportadas
### Clasificación
- Accuracy
- Precision
- Recall
- F1-Score

### Regresión
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² (R-squared)

## ✨ Contribuciones
¡Las contribuciones son bienvenidas! Puedes:
- Hacer un fork y enviar un pull request
- Reportar errores o sugerencias en los issues

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo [`LICENSE`](./LICENSE) para más detalles.
