import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

#1. Definir las rutas del proyecto
base = Path(__file__).resolve().parent.parent

archivo = base / "datos" / "datos_limp" / "telco_churn_limpio.csv"
resultados = base / "resultados"

resultados.mkdir(exist_ok = True)

#2. Cargar el dataset limpio
df = pd.read_csv(archivo)

#Preparar etiquetas para las gráficas
df["Churn_etiqueta"] = df["Churn"].map({
    0: "Permaneció",
    1: "Abandonó"
})

#3. Gráfica 1: distribución de Churn
conteo = df["Churn_etiqueta"].value_counts()
porcentajes = conteo / len(df) * 100

fig, ax = plt.subplots(figsize = (8, 5))
barras = ax.bar(conteo.index, conteo.values)

ax.set_title("Distribución de clientes según abandono")
ax.set_ylabel("Número de clientes")

for barra, porcentaje in zip(barras, porcentajes):
    ax.text(
        barra.get_x() + barra.get_width() / 2,
        barra.get_height(),
        f"{porcentaje:.1f}%",
        ha = "center",
        va = "bottom"
    )

plt.tight_layout()
plt.savefig(resultados / "01_distribucion_churn.png", dpi=300)
plt.close()

#4. Gráfica 2: antigüedad según Churn
grupos = [
    df.loc[df["Churn"] == 0, "tenure"],
    df.loc[df["Churn"] == 1, "tenure"]
]

fig, ax = plt.subplots(figsize = (8, 5))
ax.boxplot(grupos, tick_labels = ["Permaneció", "Abandonó"])

ax.set_title("Antigüedad de clientes según abandono")
ax.set_ylabel("Antigüedad (meses)")

plt.tight_layout()
plt.savefig(resultados / "02_antiguedad_churn.png", dpi = 300)
plt.close()

#5. Gráfica 3: abandono según tipo de contrato
tasas = (
    df.groupby("Contract")["Churn"]
    .mean()
    .mul(100)
    .sort_values(ascending = False)
)

fig, ax = plt.subplots(figsize = (9, 5))
barras = ax.bar(tasas.index, tasas.values)

ax.set_title("Tasa de abandono por tipo de contrato")
ax.set_ylabel("Clientes que abandonaron (%)")
ax.set_ylim(0, max(tasas.values) * 1.2)

for barra, valor in zip(barras, tasas.values):
    ax.text(
        barra.get_x() + barra.get_width() / 2,
        barra.get_height(),
        f"{valor:.1f}%",
        ha = "center",
        va = "bottom"
    )

plt.tight_layout()
plt.savefig(resultados / "03_contrato_churn.png", dpi = 300)
plt.close()

#6. Gráfica 4: cargos mensuales según Churn
grupos = [
    df.loc[df["Churn"] == 0, "MonthlyCharges"],
    df.loc[df["Churn"] == 1, "MonthlyCharges"]
]

fig, ax = plt.subplots(figsize = (8, 5))
ax.boxplot(grupos, tick_labels = ["Permaneció", "Abandonó"])

ax.set_title("Cargos mensuales según abandono")
ax.set_ylabel("Cargo mensual")

plt.tight_layout()
plt.savefig(resultados / "04_cargos_mensuales_churn.png", dpi = 300)
plt.close()

#7. Mostrar estadísticas para interpretar las gráficas
print("\nDistribución de Churn:")
print(df["Churn_etiqueta"].value_counts())

print("\nAntigüedad mediana por grupo:")
print(df.groupby("Churn_etiqueta")["tenure"].median())

print("\nTasa de abandono por contrato (%):")
print(tasas.round(2))

print("\nCargo mensual mediano por grupo:")
print(df.groupby("Churn_etiqueta")["MonthlyCharges"].median())

print("\nSe generaron correctamente las cuatro gráficas.")
print("Ubicación:", resultados)
