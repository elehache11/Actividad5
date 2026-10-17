import mlflow
import mlflow.sklearn

import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_validate, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#Identificar la carpeta principal del proyecto
proyecto = Path(__file__).resolve().parent.parent

#Cargar el dataset limpio
archivo = proyecto / "datos" / "datos_limp" / "telco_churn_limpio.csv"
df = pd.read_csv(archivo)

#Separar variables predictoras y variable objetivo
X = df.drop(columns = ["Churn"])
y = df["Churn"]

#Dividir los datos: 80% entrenamiento y 20% prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.20,
    random_state = 42,
    stratify = y
)

#Mostrar resultados
print("DIMENSIONES DE LOS CONJUNTOS")
print("Entrenamiento:", X_train.shape)
print("Prueba:", X_test.shape)

print("\nDISTRIBUCIÓN DE CHURN")
print("Dataset original:")
print(y.value_counts(normalize = True).round(4))

print("\nEntrenamiento:")
print(y_train.value_counts(normalize = True).round(4))

print("\nPrueba:")
print(y_test.value_counts(normalize = True).round(4))

#Identificar las variables numéricas y categóricas
numericas = X.select_dtypes(include = ["number"]).columns.tolist()
categoricas = X.select_dtypes(include = ["object", "str"]).columns.tolist()

print("\nVARIABLES NUMÉRICAS")
print(numericas)

print("\nVARIABLES CATEGÓRICAS")
print(categoricas)

#Definir las transformaciones
preprocesamiento = ColumnTransformer(
    transformers = [
        ("num", StandardScaler(), numericas),
        ("cat", OneHotEncoder(handle_unknown = "ignore"), categoricas)
    ]
)

print("\nPreprocesamiento configurado correctamente.")

#Crear modelo de regresión logística
modelo_logistico = Pipeline([
    ("preprocesamiento", preprocesamiento),
    ("modelo", LogisticRegression(max_iter = 1000, random_state = 42))
])

#Crear modelo Random Forest
modelo_forest = Pipeline([
    ("preprocesamiento", preprocesamiento),
    ("modelo", RandomForestClassifier(n_estimators = 100, random_state = 42))
])

#Entrenar los modelos
print("\nENTRENAMIENTO DE MODELOS")

modelo_logistico.fit(X_train, y_train)
print("Regresión logística: entrenamiento completado.")

modelo_forest.fit(X_train, y_train)
print("Random Forest: entrenamiento completado.")

#Evaluar los modelos
print("\nEVALUACIÓN DE MODELOS")

modelos = {
    "Regresión logística": modelo_logistico,
    "Random Forest": modelo_forest
}

resultados = []

for nombre, modelo in modelos.items():

    #Generar predicciones
    predicciones = modelo.predict(X_test)

    #Calcular probabilidades de abandono
    probabilidades = modelo.predict_proba(X_test)[:, 1]

    #Calcular métricas
    accuracy = accuracy_score(y_test, predicciones)
    precision = precision_score(y_test, predicciones)
    recall = recall_score(y_test, predicciones)
    f1 = f1_score(y_test, predicciones)
    roc_auc = roc_auc_score(y_test, probabilidades)

    #Guardar resultados
    resultados.append({
        "Modelo": nombre,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc
    })

#Crear tabla comparativa
tabla = pd.DataFrame(resultados)

print("\nCOMPARACIÓN DE RESULTADOS")
print(tabla.round(4).to_string(index = False))

#Configurar validación cruzada de 5 particiones
kfold = StratifiedKFold(
    n_splits = 5,
    shuffle = True,
    random_state = 42
)

#Definir las métricas
metricas = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

#Ejecutar validación cruzada
print("\nVALIDACIÓN CRUZADA (5-FOLD)")

resultados_cv = []

for nombre, modelo in modelos.items():

    evaluacion = cross_validate(
        clone(modelo),
        X_train,
        y_train,
        cv = kfold,
        scoring=metricas
    )

    #Calcular promedio y desviación estándar
    fila = {"Modelo": nombre}

    for metrica in metricas:
        valores = evaluacion["test_" + metrica]

        fila[metrica + "_promedio"] = valores.mean()
        fila[metrica + "_desv"] = valores.std()

    resultados_cv.append(fila)

#Mostrar resultados
tabla_cv = pd.DataFrame(resultados_cv)

print("\nRESULTADOS DE VALIDACIÓN CRUZADA")
print(tabla_cv.round(4).to_string(index = False))

#Definir hiperparámetros para cada modelo
parametros = {
    "Regresión logística": {
        "modelo__C": [0.1, 1, 10],
        "modelo__class_weight": [None, "balanced"]
    },
    "Random Forest": {
        "modelo__n_estimators": [100, 200],
        "modelo__max_depth": [5, 10, None],
        "modelo__class_weight": [None, "balanced"]
    }
}

#Buscar las mejores configuraciones
print("\nAJUSTE DE HIPERPARÁMETROS")

modelos_ajustados = {}

for nombre, modelo in modelos.items():

    busqueda = GridSearchCV(
        estimator = clone(modelo),
        param_grid = parametros[nombre],
        scoring = "f1",
        cv = kfold,
        n_jobs = 1
    )

    busqueda.fit(X_train, y_train)

    modelos_ajustados[nombre] = busqueda.best_estimator_

    print("\nModelo:", nombre)
    print("Mejores parámetros:", busqueda.best_params_)
    print("Mejor F1 promedio:", round(busqueda.best_score_, 4))

#Evaluar los modelos ajustados
print("\nEVALUACIÓN DE MODELOS AJUSTADOS")

resultados_ajustados = []

for nombre, modelo in modelos_ajustados.items():

    #Generar predicciones
    predicciones = modelo.predict(X_test)

    #Calcular probabilidades
    probabilidades = modelo.predict_proba(X_test)[:, 1]

    #Calcular métricas
    resultados_ajustados.append({
        "Modelo": nombre,
        "Accuracy": accuracy_score(y_test, predicciones),
        "Precision": precision_score(y_test, predicciones),
        "Recall": recall_score(y_test, predicciones),
        "F1": f1_score(y_test, predicciones),
        "ROC-AUC": roc_auc_score(y_test, probabilidades)
    })

#Crear tabla comparativa
tabla_ajustada = pd.DataFrame(resultados_ajustados)

print("\nCOMPARACIÓN DE MODELOS AJUSTADOS")
print(tabla_ajustada.round(4).to_string(index = False))

#Configurar MLflow
base_datos = proyecto / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{base_datos.as_posix()}")

#Seleccionar experimento
mlflow.set_experiment("Actividad5_Modelos")

print("\nREGISTRO DE EXPERIMENTOS EN MLFLOW")

#Registrar cada modelo ajustado
for nombre, modelo in modelos_ajustados.items():

    with mlflow.start_run(run_name = nombre.replace(" ", "_")):

        #Registrar el nombre del modelo
        mlflow.log_param("modelo", nombre)

        #Registrar los hiperparámetros
        parametros_modelo = modelo.named_steps["modelo"].get_params()

        for parametro in parametros[nombre]:
            nombre_parametro = parametro.replace("modelo__", "")
            mlflow.log_param(
                nombre_parametro,
                parametros_modelo[nombre_parametro]
            )

        #Obtener los resultados de evaluación
        resultado = tabla_ajustada[
            tabla_ajustada["Modelo"] == nombre
        ].iloc[0]

        #Registrar métricas
        for metrica in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
            mlflow.log_metric(
                metrica.lower().replace("-", "_"),
                float(resultado[metrica])
            )

        #Registrar el modelo entrenado
        mlflow.sklearn.log_model(
            sk_model = modelo,
            name = "modelo_entrenado",
            skops_trusted_types = ["sklearn.tree._tree.Tree"]
        )

        print("Modelo registrado:", nombre)

print("\nRegistro en MLflow completado.")

#Comparar resultados iniciales y ajustados
tabla_inicial = tabla.copy()
tabla_inicial["Versión"] = "Inicial"

tabla_final = tabla_ajustada.copy()
tabla_final["Versión"] = "Ajustado"

comparacion = pd.concat(
    [tabla_inicial, tabla_final],
    ignore_index = True
)

#Guardar resultados
carpeta = proyecto / "resultados"
carpeta.mkdir(exist_ok = True)

comparacion.to_csv(
    carpeta / "comparacion_modelos.csv",
    index = False
)

#Crear gráfica comparativa
grafica = comparacion.set_index(
    ["Modelo", "Versión"]
)[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]

grafica.plot(kind = "bar", figsize = (11, 6))

plt.title("Comparación de modelos iniciales y ajustados")
plt.ylabel("Valor de la métrica")
plt.xlabel("Modelo y versión")
plt.ylim(0, 1)
plt.xticks(rotation = 20, ha = "right")
plt.legend(title = "Métricas")
plt.tight_layout()

plt.savefig(
    carpeta / "comparacion_modelos.png",
    dpi=300
)

plt.show()

print("\nResultados comparativos guardados correctamente.")

#Registrar los resultados comparativos en MLflow
with mlflow.start_run(run_name = "Comparacion_modelos"):

    mlflow.log_artifact(
        str(carpeta / "comparacion_modelos.csv")
    )

    mlflow.log_artifact(
        str(carpeta / "comparacion_modelos.png")
    )

print("Resultados comparativos registrados en MLflow.")
