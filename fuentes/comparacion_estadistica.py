import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#1. Ubicación de los archivos
base = Path(__file__).resolve().parent.parent

archivo = base / "datos" / "datos_limp" / "telco_churn_limpio.csv"

resultados = base / "resultados"
resultados.mkdir(exist_ok = True)

#2. Cargar datos
df = pd.read_csv(archivo)

X = df.drop(columns = ["Churn"])
y = df["Churn"]

#3. División de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.20,
    random_state = 42,
    stratify = y
)

#4. Preprocesamiento
numericas = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categoricas = [
    columna for columna in X.columns
    if columna not in numericas
]

preprocesamiento = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numericas),
        ("cat", OneHotEncoder(handle_unknown = "ignore"), categoricas)
    ]
)

#5. Modelos con hiperparámetros ajustados
logistica = Pipeline([
    ("preprocesamiento", preprocesamiento),
    ("modelo", LogisticRegression(
        C = 0.1,
        class_weight = "balanced",
        max_iter = 1000,
        random_state = 42
    ))
])

forest = Pipeline([
    ("preprocesamiento", preprocesamiento),
    ("modelo", RandomForestClassifier(
        n_estimators = 100,
        max_depth = 10,
        class_weight = "balanced",
        random_state = 42
    ))
])

#6. Entrenamiento
print("Entrenando regresión logística...")
logistica.fit(X_train, y_train)

print("Entrenando Random Forest...")
forest.fit(X_train, y_train)

#7. Predicciones
pred_logistica = logistica.predict(X_test)
pred_forest = forest.predict(X_test)

# 8. Comparar aciertos y errores
correcto_logistica = pred_logistica == y_test.to_numpy()
correcto_forest = pred_forest == y_test.to_numpy()

ambos_correctos = np.sum(correcto_logistica & correcto_forest)

solo_logistica = np.sum(
    correcto_logistica & ~correcto_forest
)

solo_forest = np.sum(
    ~correcto_logistica & correcto_forest
)

ambos_incorrectos = np.sum(
    ~correcto_logistica & ~correcto_forest
)

#9. Tabla de contingencia
tabla = pd.DataFrame(
    [
        [ambos_correctos, solo_logistica],
        [solo_forest, ambos_incorrectos]
    ],
    index = [
        "Logistica correcta",
        "Logistica incorrecta"
    ],
    columns = [
        "Forest correcto",
        "Forest incorrecto"
    ]
)

print("\nTabla de contingencia:")
print(tabla)

#10. Prueba exacta de McNemar
#Equivale a una prueba binomial bilateral
#sobre los pares discordantes.

from scipy.stats import binomtest

discordantes = solo_logistica + solo_forest

if discordantes > 0:
    prueba = binomtest(
        k = int(solo_logistica),
        n = int(discordantes),
        p = 0.5,
        alternative = "two-sided"
    )

    valor_p = prueba.pvalue
else:
    valor_p = 1.0

print("\nResultados de McNemar:")
print("Aciertos exclusivos de regresión logística:", solo_logistica)
print("Aciertos exclusivos de Random Forest:", solo_forest)
print("Pares discordantes:", discordantes)
print("Valor p:", round(valor_p, 6))

if valor_p < 0.05:
    print("Resultado: diferencia estadísticamente significativa.")
else:
    print("Resultado: no se encontró evidencia suficiente de diferencia.")

#11. Guardar resultados
tabla.to_csv(
    resultados / "tabla_mcnemar.csv",
    encoding = "utf-8-sig"
)

resumen = pd.DataFrame({
    "Indicador": [
        "Aciertos exclusivos de logística",
        "Aciertos exclusivos de Random Forest",
        "Pares discordantes",
        "Valor p",
        "Nivel de significancia"
    ],
    "Valor": [
        solo_logistica,
        solo_forest,
        discordantes,
        valor_p,
        0.05
    ]
})

resumen.to_csv(
    resultados / "resultado_mcnemar.csv",
    index = False,
    encoding = "utf-8-sig"
)

print("\nResultados guardados correctamente.")
