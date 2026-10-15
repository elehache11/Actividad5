import pandas as pd

from pathlib import Path

#Identificar la carpeta principal del proyecto
proyecto = Path(__file__).resolve().parent.parent

#Ubicación del dataset original
archivo = proyecto / "datos" / "datos_ini" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

#Cargar el dataset
df = pd.read_csv(archivo)

#1. Dimensiones
print("\nDIMENSIONES DEL DATASET")
print("Registros:", df.shape[0])
print("Columnas:", df.shape[1])

#2. Primeros registros
print("\nPRIMEROS 5 REGISTROS")
print(df.head().to_string())

#3. Variables y tipos de datos
print("\nVARIABLES Y TIPOS DE DATOS")
print(df.dtypes.to_string())

#4. Valores nulos
print("\nVALORES NULOS POR COLUMNA")
print(df.isnull().sum().to_string())

#5. Registros duplicados
print("\nREGISTROS DUPLICADOS")
print(df.duplicated().sum())

#6. Distribución de la variable objetivo
print("\nDISTRIBUCIÓN DE CHURN")
print(df["Churn"].value_counts())

print("\nPORCENTAJES DE CHURN")
print((df["Churn"].value_counts(normalize = True) * 100).round(2))

#7. Revisar valores no numéricos en TotalCharges
print("\nREVISIÓN DE TOTALCHARGES")

total_numerico = pd.to_numeric(df["TotalCharges"], errors = "coerce")

print("Valores no convertibles:", total_numerico.isna().sum())

print("\nREGISTROS CON VALORES NO CONVERTIBLES")
print(df.loc[total_numerico.isna(),
             ["tenure", "MonthlyCharges", "TotalCharges"]].to_string(index = False))

#8. Revisar identificadores duplicados
print("\nCUSTOMERID DUPLICADOS")
print(df["customerID"].duplicated().sum())
