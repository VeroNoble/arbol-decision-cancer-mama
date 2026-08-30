# Árbol de decisión para diagnóstico de cáncer de mama

Trabajo práctico de la Tecnicatura Superior en Ciencia de Datos e IA.

## Objetivo

Clasificar tumores de mama como benignos o malignos a partir de características geométricas y de textura de las células, usando un árbol de decisión, y visualizar e interpretar el modelo resultante.

## Datos

Dataset **Breast Cancer Wisconsin (Diagnostic)**, 569 registros con 30 características numéricas de cada tumor (radio, textura, perímetro, área, concavidad, simetría, etc., en sus valores medio, error estándar y peor caso). Las columnas fueron traducidas al español para facilitar la interpretación del árbol.

## Enfoque

1. Carga y traducción de columnas al español, eliminación de la columna `id` (no aporta valor predictivo).
2. Conversión de la variable objetivo (`diagnostico`: M/B) a valores binarios (1/0).
3. Partición 80/20 en entrenamiento y prueba.
4. Entrenamiento de un `DecisionTreeClassifier` (criterio Gini, profundidad máxima de 3 niveles) para mantener el árbol interpretable.
5. Evaluación con accuracy, matriz de confusión y reporte de clasificación.
6. Visualización del árbol completo, con los términos traducidos al español (Gini, muestras, valor, clase, verdadero/falso).

## Resultados

- **Precisión (accuracy): 95%**
- La variable más determinante para la primera división es `puntos_concavos_medio`, seguida de `radio_peor`.
- El árbol, limitado a 3 niveles de profundidad, logra separar la mayoría de los casos benignos y malignos manteniendo una estructura simple e interpretable — algo valioso en un contexto de soporte al diagnóstico médico, donde poder explicar el porqué de una predicción es tan importante como la predicción en sí.

## Herramientas

Python · pandas · scikit-learn · matplotlib

## Cómo correrlo

```bash
pip install -r requirements.txt
python arbol_decision_cancer_mama.py
```
