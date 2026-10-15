import pandas as pd

from pathlib import Path

#Identificar la carpeta principal del proyecto
proyecto = Path(__file__).resolve().parent.parent

#Definir las rutas de los archivos
origen = proyecto / "datos" / "datos_ini" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
destino = proyecto / "datos" / "datos_limp" / "telco_churn_limpio.csv"

#Cargar el dataset original
df = pd.read_csv(origen)

#Convertir TotalCharges a formato numérico
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors = "coerce")

#Sustituir por cero los valores faltantes con tenure = 0
condicion = df["TotalCharges"].isna() & (df["tenure"] == 0)
df.loc[condicion, "TotalCharges"] = 0

#Convertir la variable objetivo a formato binario
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

#Eliminar el identificador del cliente
df = df.drop(columns = ["customerID"])

#Verificar valores faltantes y duplicados
print("Registros:", df.shape[0])
print("Columnas:", df.shape[1])
print("Valores nulos:", df.isnull().sum().sum())
print("Registros duplicados:", df.duplicated().sum())

#Guardar el dataset limpio
destino.parent.mkdir(parents = True, exist_ok = True)
df.to_csv(destino, index = False)

print("\nDataset limpio guardado correctamente.")
print("Ubicación:", destino)
