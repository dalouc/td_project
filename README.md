# Identificación de Publicaciones Desinformativas y Polarización Ideológica

## Descripción general
Este proyecto se enfoca en detectar publicaciones potencialmente desinformativas (por ejemplo, **noticias falsas / *fake news***). Para ello, se entrenaron y compararon distintos modelos de **clasificación de texto**, desde enfoques de *machine learning* tradicional hasta métodos de *deep learning*.  
El objetivo es **identificar automáticamente** contenidos posiblemente falsos o engañosos, y comparar qué combinación de **representación del texto + clasificador** funciona mejor.

---

## Objetivos del proyecto

- **Clasificación de textos con diferentes representaciones vectoriales**
  - Se emplean varias formas de representar el texto numéricamente:
    - **TF-IDF** (frecuencias de término ponderadas por el inverso de frecuencia en documentos).
    - **Word2Vec** (embeddings entrenados para capturar similitudes semánticas).
    - **Embeddings de BERT** preentrenados (representación contextual).
  - Sobre estas representaciones se prueban modelos supervisados clásicos:
    - **Random Forest** (Scikit-learn).
    - **MLP** simple (PyTorch).

- **Fine-tuning de un modelo Transformer**
  - Se ajusta finamente un modelo tipo **BERT** para clasificación binaria (**real vs. fake**), entrenando de extremo a extremo con el texto.

- **Evaluación rigurosa y comparación de modelos**
  - Se evalúan todos los enfoques usando:
    - **Accuracy**
    - **F1-score**
    - **ROC-AUC**
  - Se busca determinar qué combinación ofrece el mejor rendimiento en detección de desinformación.

---

## Datos utilizados
El conjunto de datos proviene de una fuente pública de noticias etiquetadas como reales o falsas. Incluye artículos en inglés de agencias confiables (por ejemplo, Reuters) y artículos de sitios web de dudosa credibilidad o conocidos por difundir desinformación.

Cada ejemplo contiene:
- `title`: título
- `text`: cuerpo completo
- `label`: etiqueta binaria (**1 = real**, **0 = potencialmente falsa**)

En total se recopilaron alrededor de **40.000** noticias, aproximadamente balanceadas entre ambas clases.

### Preprocesamiento y limpieza (sesgos del dataset)
Durante el análisis exploratorio:
1. Se eliminaron noticias repetidas.
2. Se eliminaron noticias con longitudes de palabras muy breves. 
3. Se recortaron las noticias con longitudes de palabras muy largas.
4. Se observó que muchas noticias reales incluían la palabra **"Reuters"** en el texto, lo cual se convertía en una “trampa” para el modelo (alto desempeño, baja generalización).  
Para mitigarlo, se **eliminaron menciones explícitas a la fuente** en el texto, obligando a los modelos a aprender señales lingüísticas reales y no metadatos accidentales.

> **Nota sobre idioma**: el dataset referenciado está en **inglés**. Si se usa un BERT “en español”, se recomienda asegurar consistencia (por ejemplo, usando un modelo multilingüe o una versión traducida/alternativa del corpus).

---

## Técnicas de modelado

### 1) TF-IDF + clasificadores
- Vectorización: **TF-IDF**
- Clasificadores evaluados:
  - Random Forest
  - MLP (PyTorch) con:
    - capas densas + ReLU
    - dropout
    - salida sigmoidal (probabilidad de clase positiva)
- Ventajas:
  - Maneja bien alta dimensionalidad
  - Permite inspeccionar importancia de características (palabras clave)

### 2) Word2Vec + clasificadores
- Entrenamiento de Word2Vec (Gensim) con **embeddings de 300 dimensiones**
- Representación de documento: promedio de embeddings de palabras
- Clasificadores evaluados:
  - Random Forest
  - MLP (PyTorch) con las mismas capas anteriores.

### 3) Embeddings de BERT + clasificadores
- Se extrae el embedding del token **[CLS]** (vector de 768 dimensiones) por documento
- Clasificadores evaluados:
  - Random Forest
  - MLP (PyTorch) con las mismas capas anteriores.

### 4) Fine-tuning de BERT (Transformers)
- Ajuste fino de BERT para clasificación binaria
- Entrenamiento:
  - **3 epochs**
  - optimizador **AdamW**
  - *learning rate* bajo (**2e-5**)
  - validación durante entrenamiento y selección del mejor modelo según accuracy de validación

---

## Evaluación de resultados
Se utilizó un conjunto de **validación** (para ajuste de hiperparámetros) y un conjunto **test** separado para la comparación final.

**Métricas reportadas**:
- Accuracy (Acc.)
- F1-score (F1)
- ROC-AUC

### Resultados (test)
| Modelo (Representación)                    | Acc.  | F1    | ROC-AUC |
|-------------------------------------------|-------|-------|--------|
| Random Forest (TF-IDF)                     | ~96%  | ~96%  | ~0.99  |
| MLP (TF-IDF)                               | ~97%  | ~97%  | ~0.99  |
| Random Forest (Word2Vec promedio)          | ~95%  | ~95%  | ~0.99  |
| MLP (Word2Vec promedio)                    | ~95%  | ~95%  | ~0.99  |
| Random Forest (BERT embedding [CLS])       | ~91%  | ~91%  | ~0.91  |
| MLP (BERT embedding [CLS])                 | ~95%  | ~95%  | ~0.99  |
| BERT Transformer fine-tune (HuggingFace)   | 96–97%| 96–97%| 0.99   |

---

## Observaciones
- Los enfoques basados en **TF-IDF** obtuvieron resultados muy altos (96–97%). Inicialmente, parte del rendimiento se explicaba por señales obvias (por ejemplo, “Reuters”), motivo por el cual se eliminaron dichas menciones.
- **Word2Vec** logró buen desempeño (~95% F1), pero ligeramente inferior a TF-IDF, posiblemente porque el promedio de vectores pierde información de orden y contexto.
- Los **embeddings BERT sin fine-tuning** mostraron un comportamiento mixto:
  - con Random Forest el rendimiento fue menor (~0.91 F1),
  - con MLP mejoró notablemente (~95%).
- El **BERT fine-tune** alcanzó el mejor rendimiento global (96–97%), con mayor potencial de generalización al aprender señales más complejas del lenguaje.

> Estos resultados deben interpretarse con cautela: el dataset contiene sesgos (fuentes diferenciadas, temas específicos, patrones de estilo) que pueden inflar métricas en escenarios controlados.

---

## Conclusiones

El proyecto demuestra que la detección automática de noticias desinformativas es viable con alta precisión usando tanto técnicas clásicas como modelos avanzados de aprendizaje automático. Métodos simples como Random Forest con TF-IDF ya identifican diferencias claras entre noticias reales y falsas, mientras que modelos neuronales y Transformers captan matices lingüísticos más sutiles.

La elección de cómo representar el texto influye en el rendimiento. TF-IDF, a pesar de su simplicidad, funcionó sorprendentemente bien en este caso
(debido a términos reveladores presentes en las noticias reales). Por otro lado, los embeddings densos (Word2Vec, BERT) ofrecen una representación más general; cuando se complementaron con modelos adecuados, alcanzaron desempeños similares. Esto sugiere que en tareas de fake news detection, las palabras clave y el estilo de redacción pueden ser indicadores tan fuertes como las características semánticas profundas.

Finalmente, se identifica como línea futura de investigación la incorporación del análisis de polarización ideológica mediante técnicas de stance detection y análisis de sentimiento, ya que muchas fake news presentan un fuerte sesgo político y lenguaje emotivo, lo que podría ayudar a comprender mejor su papel en la polarización social.


## Requisitos
Se recomienda el siguiente entorno:

- Python 3
- Jupyter Notebook (opcional)
- Librerías:
  - `pandas`, `numpy`, `matplotlib`
  - `scikit-learn`
  - `gensim`
  - `torch`
  - `transformers`
  - `sklearn.metrics` (u otras utilidades para métricas)

---

## Ejecución (pasos generales)

1. **Preparar los datos**
   - Descargar el dataset público y colocarlo en el directorio del proyecto.
   - Ajustar la ruta del archivo en el código de carga si es necesario.

2. **Preprocesar y vectorizar**
   - Ejecutar scripts/celdas para:
     - limpieza del texto
     - TF-IDF
     - entrenamiento de Word2Vec
     - extracción de embeddings de BERT

3. **Entrenar modelos**
   - Entrenar secuencialmente:
     - Random Forest
     - MLP (PyTorch)
     - Fine-tuning de BERT (Transformers)
   - Se recomienda GPU para acelerar PyTorch/Transformers si está disponible.

4. **Evaluar resultados**
   - Ejecutar evaluación en test:
     - accuracy, F1, ROC-AUC
   - Revisar salidas cualitativas:
     - importancias de variables (RF + TF-IDF)
     - matrices de confusión (si se incluyen)
     - curvas ROC (opcional)

---

## Fuente de datos
Dataset público (Kaggle):  
`https://www.kaggle.com/datasets/aadyasingh55/fake-news-classification?resource=download&select=train+%282%29.csv`

---

## Nota
Los datos utilizados provienen de fuentes públicas y abiertas, disponibles para fines de investigación.
