import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. CARGA DE DATOS
try:
    df = pd.read_csv('CancerDeMama.csv')
    
    # Traducción de columnas al español
    traducciones = {
        'diagnosis': 'diagnostico',
        'radius_mean': 'radio_medio', 'texture_mean': 'textura_media', 'perimeter_mean': 'perimetro_medio',
        'area_mean': 'area_media', 'smoothness_mean': 'suavidad_media', 'compactness_mean': 'compacidad_media',
        'concavity_mean': 'concavidad_media', 'concave points_mean': 'puntos_concavos_medio',
        'symmetry_mean': 'simetria_media', 'fractal_dimension_mean': 'dimension_fractal_media',
        'radius_se': 'radio_se', 'texture_se': 'textura_se', 'perimeter_se': 'perimetro_se',
        'area_se': 'area_se', 'smoothness_se': 'suavidad_se', 'compactness_se': 'compacidad_se',
        'concavity_se': 'concavidad_se', 'concave points_se': 'puntos_concavos_se',
        'symmetry_se': 'simetria_se', 'fractal_dimension_se': 'dimension_fractal_se',
        'radius_worst': 'radio_peor', 'texture_worst': 'textura_peor', 'perimeter_worst': 'perimetro_peor',
        'area_worst': 'area_peor', 'smoothness_worst': 'suavidad_peor', 'compactness_worst': 'compacidad_peor',
        'concavity_worst': 'concavidad_peor', 'concave points_worst': 'puntos_concavos_peor',
        'symmetry_worst': 'simetria_peor', 'fractal_dimension_worst': 'dimension_fractal_peor'
    }
    df = df.rename(columns=traducciones)
    
    print("Archivo cargado y columnas traducidas correctamente.")
except FileNotFoundError:
    print("Error: No se encontró 'CancerDeMama.csv'.")
    exit()

# 2. PREPARACIÓN ESPECÍFICA (Limpieza)
# Eliminamos la columna 'id' porque no sirve para predecir
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# En el dataset, la columna objetivo a predecir es 'diagnostico' (M=Maligno, B=Benigno)
# X (Características/Pistas) = Todo el dataset menos la columna diagnostico
# y (Target/Respuesta) = Solo la columna diagnostico
X = df.drop('diagnostico', axis=1) # axis=1 indica que se debe eliminar la columna entera.
y = df['diagnostico'] #se asigna la columna diagnostico a la variable Y.

# Se reemplaza M (maligno) por 1 y B (benigno) por 0.
y = y.map({'M': 1, 'B': 0})

#Se eliminan columnas vacías en caso que hubiera.
X = X.loc[:, ~X.columns.str.contains('^Unnamed')]

# 3. DIVISIÓN DEL DATASET
# Dividimos los datos en dos grupos: Entrenamiento (80%) y Prueba (20%)
#test_size=0.2 indica que el 20% de los datos se van a usar para prueba.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. ENTRENAMIENTO
# max_depth=3 indica que el arbol puede tener hasta 3 niveles de profundidad.
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# 5. PROTOTIPOS Y PRUEBAS
# Hacemos predicciones sobre el conjunto de prueba que no se usó para entrenar al modelo.
y_pred = clf.predict(X_test)

print("\n" + "="*30)
print("   RESULTADOS DEL MODELO")
print("="*30)
print(f"Precisión (Accuracy): {accuracy_score(y_test, y_pred):.2f}")
print("\nMatriz de Confusión (Maligno=1, Benigno=0):")
print(confusion_matrix(y_test, y_pred))
print("\nInforme detallado:")
reporte_dict = classification_report(y_test, y_pred, target_names=['Benigno (0)', 'Maligno (1)'], output_dict=True)

print(f"{'':<20} {'Precisión':<12} {'Sensibilidad':<15} {'Puntaje F1':<12} {'Casos Totales':<15}")
print("-" * 75)
for clase in ['Benigno (0)', 'Maligno (1)']:
    print(f"{clase:<20} {reporte_dict[clase]['precision']:<12.2f} {reporte_dict[clase]['recall']:<15.2f} {reporte_dict[clase]['f1-score']:<12.2f} {int(reporte_dict[clase]['support']):<15}")
print("-" * 75)
print(f"{'Accuracy':<20} {'':<12} {'':<15} {reporte_dict['accuracy']:<12.2f} {int(reporte_dict['macro avg']['support']):<15}")
print(f"{'Promedio macro':<20} {reporte_dict['macro avg']['precision']:<12.2f} {reporte_dict['macro avg']['recall']:<15.2f} {reporte_dict['macro avg']['f1-score']:<12.2f} {int(reporte_dict['macro avg']['support']):<15}")
print(f"{'Promedio ponderado':<20} {reporte_dict['weighted avg']['precision']:<12.2f} {reporte_dict['weighted avg']['recall']:<15.2f} {reporte_dict['weighted avg']['f1-score']:<12.2f} {int(reporte_dict['weighted avg']['support']):<15}")

# 6. VISUALIZACIÓN
plt.figure(figsize=(18, 10))
plot_tree(clf, 
          filled=True, 
          feature_names=list(X.columns), 
          class_names=['Benigno', 'Maligno'], 
          rounded=True,
          fontsize=10)

# Traducción de los textos internos del gráfico generados por sklearn
for texto in plt.gca().texts:
    nuevo_texto = texto.get_text()
    nuevo_texto = nuevo_texto.replace("gini", "Gini")
    nuevo_texto = nuevo_texto.replace("samples", "muestras")
    nuevo_texto = nuevo_texto.replace("value", "valor")
    nuevo_texto = nuevo_texto.replace("class", "clase")
    nuevo_texto = nuevo_texto.replace("True", "Verdadero")
    nuevo_texto = nuevo_texto.replace("False", "Falso")
    texto.set_text(nuevo_texto)

plt.title("Árbol de Decisión - Clasificación de Cáncer de Mama")
plt.show()