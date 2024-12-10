# %%
import pandas as pd
import numpy as np
from keras.api.models import Sequential
from keras.api.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# %%
# Carregar os dados
data = pd.read_csv("sinais_vitais.txt", header=None)

# %%
# Seleção de variáveis preditoras e alvos
X = data.iloc[
    :, [3, 4, 5]
].values  # si3 (qualidade da pressão), si4 (pulso), si5 (respiração)
y_gravity = data.iloc[:, -2].values  # Gravidade contínua
y_labels = data.iloc[:, -1].values  # Rótulo de saída

# %%
# Normalização
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# %%
# Divisão em treino e teste
X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = train_test_split(
    X_normalized, y_gravity, y_labels, test_size=0.2, random_state=42
)

# %%
# Modelo de Regressão para prever gravidade contínua
model = Sequential(
    [
        Input(shape=(X.shape[1],)),
        Dense(256, activation="relu"),
        Dense(128, activation="relu"),
        Dense(1),  # Saída contínua
    ]
)

# %%
model.compile(loss="mse", metrics=["mae"])

# %%
# Treinamento
history = model.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1,
)

# %%
# Previsões
y_pred = model.predict(X_test).flatten()


# %%
# Classificação com base nos intervalos fixos
def classify_gravity(value):
    if 0 <= value <= 25:
        return 1
    elif 25 < value <= 50:
        return 2
    elif 50 < value <= 75:
        return 3
    elif 75 < value <= 100:
        return 4


# %%
# Prever gravidade contínua no conjunto de teste
y_pred_continuous = model.predict(X_test).flatten()

# Mapear previsões contínuas para classes
y_pred_classes = np.array([classify_gravity(v) for v in y_pred_continuous])

# Mapear os valores reais de gravidade para classes
y_test_classes = np.array([classify_gravity(v) for v in y_test])

# %%
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Acurácia da classificação
accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f"Acurácia da classificação: {accuracy:.4f}")

# Matriz de confusão
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
print("Matriz de Confusão:\n", conf_matrix)

# Relatório de classificação
print(
    "Relatório de Classificação:\n",
    classification_report(y_test_classes, y_pred_classes),
)

# Número de acertos (diagonal da matriz de confusão)
correct_predictions = np.trace(conf_matrix)

# Número total de amostras
total_predictions = np.sum(conf_matrix)

# Porcentagem de acertos
accuracy_percentage = (correct_predictions / total_predictions) * 100
print(f"Porcentagem de Acertos: {accuracy_percentage:.2f}%")

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Função para suavizar curvas com médias móveis
def smooth_curve(values, weight=0.90):
    smoothed = []
    last = values[0]
    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


# Gráfico de Perda
plt.figure(figsize=(8, 6))
train_loss = smooth_curve(history.history["loss"])
val_loss = smooth_curve(history.history["val_loss"])
plt.plot(train_loss, label="Treino", linewidth=2, color="blue")
plt.plot(val_loss, label="Validação", linewidth=2, linestyle="--", color="orange")
plt.title("Evolução da Perda", fontsize=16)
plt.xlabel("Épocas", fontsize=14)
plt.ylabel("Perda", fontsize=14)
plt.legend(fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Adicionar anotação dos valores finais
plt.annotate(
    f"{train_loss[-1]:.4f}",
    xy=(len(train_loss) - 1, train_loss[-1]),
    xytext=(-40, -15),
    textcoords="offset points",
    fontsize=12,
    color="blue",
)
plt.annotate(
    f"{val_loss[-1]:.4f}",
    xy=(len(val_loss) - 1, val_loss[-1]),
    xytext=(-40, 15),
    textcoords="offset points",
    fontsize=12,
    color="orange",
)

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_continuous, alpha=0.6, label="Predições Contínuas")
plt.plot([0, 100], [0, 100], color="red", linestyle="--", label="Ideal")
plt.xlabel("Gravidade Real")
plt.ylabel("Gravidade Prevista")
plt.title("Gravidade Real vs Prevista")
plt.legend()
plt.grid()
plt.show()

# Calcular a matriz de confusão
cm = confusion_matrix(y_test_classes, y_pred_classes)

# Exibir a matriz de confusão com o display
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3, 4])

# Plotar a matriz de confusão
disp.plot(cmap="Blues", values_format="d")

# Título e exibição
plt.title("Matriz de Confusão", fontsize=14)
plt.show()
