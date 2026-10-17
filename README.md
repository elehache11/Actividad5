\# Actividad 5. Entrenamiento y seguimiento de modelos de Machine Learning



\## 1. Descripción de la actividad



Esta actividad tiene como objetivo entrenar, evaluar y comparar dos algoritmos de clasificación para predecir el abandono de clientes (Churn) en una empresa de telecomunicaciones.



Se utilizan regresión logística y Random Forest, aplicando técnicas de preparación de datos, validación cruzada y ajuste de hiperparámetros.



Los experimentos se registran mediante MLflow y el código se versiona utilizando Git y GitHub.



\## 2. Dataset



Se utiliza el conjunto de datos público Telco Customer Churn.



Fuente:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn



Características principales:



\- 7,043 registros de clientes.

\- 21 columnas originales.

\- Variables numéricas y categóricas.

\- Variable objetivo: Churn (Yes/No).



Distribución de la variable objetivo:



\- No abandonaron: 73.46 %.

\- Abandonaron: 26.54 %.



\## 3. Preparación de datos



La exploración y limpieza se realizaron mediante Python y pandas.



Las principales transformaciones fueron:



\- Conversión de TotalCharges a formato numérico.

\- Sustitución de 11 valores vacíos por cero, considerando que

&#x20; corresponden a clientes con tenure igual a cero.

\- Conversión de Churn a una variable binaria.

\- Exclusión de customerID del entrenamiento.

\- Conservación de los 7,043 registros originales.



Se identificaron 22 coincidencias completas después de eliminar customerID. Se conservaron porque no necesariamente representan clientes duplicados.



La codificación de variables categóricas y la estandarización de variables numéricas se realizan mediante pipelines de Scikit-learn.



\## 4. Modelos utilizados



Se entrenaron dos algoritmos:



\- Regresión logística.

\- Random Forest.



Ambos modelos utilizaron los mismos conjuntos de entrenamiento y prueba.



Se estableció random\_state = 42 para reproducibilidad.



\## 5. Metodología



El dataset se dividió de la siguiente manera:



\- 80 % para entrenamiento.

\- 20 % para prueba.



Se utilizó una división estratificada para conservar la proporción de las clases.



Posteriormente, se aplicó validación cruzada Stratified K-Fold con cinco particiones.



El ajuste de hiperparámetros se realizó mediante GridSearchCV, utilizando F1-score como criterio de optimización.



Las métricas utilizadas fueron:



\- Accuracy.

\- Precision.

\- Recall.

\- F1-score.

\- ROC-AUC.



\## 6. Resultados



Los modelos ajustados obtuvieron los siguientes resultados en el entorno local:



| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |

| Regresión logística | 0.7417 | 0.5087 | 0.7861 | 0.6176 | 0.8410 |

| Random Forest | 0.7573 | 0.5300 | 0.7567 | 0.6233 | 0.8374 |



Random Forest obtuvo un F1-score ligeramente superior, mientras que la regresión logística presentó mejores resultados en recall y ROC-AUC.



La selección del modelo dependerá de las prioridades del negocio, especialmente del equilibrio entre identificar posibles abandonos y evitar falsas alarmas.



\## 7. Seguimiento de experimentos con MLflow



Se utilizó MLflow 3.17.1 con una base de datos SQLite local.



El experimento Actividad5\_Modelos contiene registros de:



\- Hiperparámetros de los modelos.

\- Métricas de evaluación.

\- Modelos entrenados.

\- Archivos comparativos CSV y PNG.



Los archivos de almacenamiento local de MLflow están excluidos del repositorio mediante .gitignore.



Las evidencias visuales permiten consultar los resultados de los experimentos realizados.



\## 8. Instrucciones de reproducción



\### Requisitos



\- Python 3.11.

\- Bibliotecas incluidas en requirements.txt.

\- Acceso al dataset disponible en el repositorio.



\### Instalación



Clonar o descargar el repositorio.



Desde la carpeta principal del proyecto, instalar las dependencias:



python -m pip install -r requirements.txt



\### Preparación de datos



Ejecutar:



python fuentes/datos\_prep.py



\### Entrenamiento y seguimiento



Ejecutar:



python fuentes/train.py



El programa realiza el entrenamiento, la evaluación, la validación cruzada, el ajuste de hiperparámetros y el registro de resultados en MLflow.



\### Google Colab



El archivo fuentes/entrena.ipynb permite ejecutar los experimentos en Google Colab.



El notebook descarga el dataset limpio directamente desde GitHub y genera los resultados comparativos.



\### Consideraciones de reproducibilidad



El desarrollo local se realizó con Scikit-learn 1.9.0.



Durante las pruebas en Google Colab se utilizó Scikit-learn 1.6.1.



Se identificaron pequeñas diferencias en los resultados de Random Forest, aunque se conservaron los mismos hiperparámetros óptimos.



Estas diferencias deben considerarse al reproducir los experimentos en distintos entornos.



\## 9. Control de versiones



Los cambios relevantes del proyecto se documentan en CHANGELOG.md.



El código fuente y los datasets se encuentran versionados mediante Git y GitHub.

