import os
import gzip
import json
import pickle
import zipfile
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
#*from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
# Paso 1: Limpieza de los datos
def clean_data(df):
    df = df.copy()                                                      #? Se crea una copia del DataFrame para evitar modificar el original
    df = df.dropna()                                                    #? Elimina las filas con valores nulos
    df['Age'] = 2021 - df['Year']
    df = df.drop(columns=['Year', 'Car_Name'])
    
    return df

# Paso 3: Crear el pipeline del modelo
def model():
    categorical_columns = ["Fuel_Type", "Selling_type", "Transmission"]              #? Columnas categóricas
    numeric_columns = ["Selling_Price", "Driven_kms", "Age", "Owner"]                #? Columnas numéricas

    preprocessor = ColumnTransformer(                                                     #? Transformación de columnas categóricas y numéricas
                    transformers = [
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns), #? OneHotEncoding para las columnas categóricas
                    ('scaler', MinMaxScaler(), numeric_columns)                         #? Estandarización de las variables numéricas
                    ],
                    remainder ='passthrough'                                              #? Las columnas restantes se mantienen sin cambios
                    )
    selectkbest  = SelectKBest(score_func = f_classif)                           #? Selección de las K mejores característica
    pipeline = Pipeline(steps = [                                             #? Modelo de clasificador (SVM)
                        ('preprocessor', preprocessor),                       #? Primero, se aplica la preprocesamiento
                        ("selectkbest", selectkbest ),                           #? Selección de las K mejores característica
                        ('classifier', LinearRegression())
                                ]
                        )

    return pipeline                                                           #? Devuelve el pipeline completo


def optimize_hyperparameters(model, n_splits, x_train, y_train, scoring):     #? Paso 4: Optimización de hiperparámetros
    estimator = GridSearchCV(
        estimator = model,
        param_grid = {
            'selectkbest__k': range(1, 13),                                           #? Selección de las mejores 20 características
        },
        cv = n_splits,                                                          #? Validación cruzada con 10 splits
        refit = True,
        #*verbose=0,                                                            #? No mostrar detalles adicionales del proceso
        scoring = scoring,                                                      #? Métrica de precisión balanceada
        #*return_train_score = False                                            #? No devolver las métricas de entrenamiento
    )
    estimator.fit(x_train, y_train)                                           #? Ajusta el modelo con los datos de entrenamiento

    return estimator

def metrics(model, x_train, y_train, x_test, y_test):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_metrics = {
        'type': 'metrics',
        'dataset': 'train',
        'r2': r2_score(y_train, y_train_pred),
        'mse': mean_squared_error(y_train, y_train_pred),
        'mad': median_absolute_error(y_train, y_train_pred)
    }

    test_metrics = {
        'type': 'metrics',
        'dataset': 'test',
        'r2': r2_score(y_test, y_test_pred),
        'mse': mean_squared_error(y_test, y_test_pred),
        'mad': median_absolute_error(y_test, y_test_pred)
    }

    return train_metrics, test_metrics

def save_model(model):                                      #? Paso 5: Guardar el modelo
    os.makedirs('files/models', exist_ok = True)

    with gzip.open('files/models/model.pkl.gz', 'wb') as f:
        pickle.dump(model, f)

def save_metrics(metrics):
    os.makedirs('files/output', exist_ok = True)

    with open("files/output/metrics.json", "w") as f:
        for metric in metrics:
            json_line = json.dumps(metric)
            f.write(json_line + "\n")


def bootstrap_experiments(model, x_data, y_data, iterations=100, random_state=42):
    rng = np.random.default_rng(random_state)
    x_data = x_data.reset_index(drop=True)
    y_data = y_data.reset_index(drop=True)
    n_samples = len(x_data)

    experiments = []
    for idx in range(iterations):
        sample_indices = rng.integers(0, n_samples, n_samples)
        x_sample = x_data.iloc[sample_indices]
        y_sample = y_data.iloc[sample_indices]
        y_pred = model.predict(x_sample)

        if np.unique(y_sample).size > 1:
            r2_value = r2_score(y_sample, y_pred)
        else:
            r2_value = float('nan')

        experiments.append({
            'experiment': idx + 1,
            'r2': r2_value,
            'mse': mean_squared_error(y_sample, y_pred),
            'mad': median_absolute_error(y_sample, y_pred)
        })

    return pd.DataFrame(experiments)


def summarize_experiments(experiments_df):
    summary = experiments_df[['r2', 'mse', 'mad']].agg(['mean', 'std', 'min', 'max'])
    summary = summary.rename_axis('statistic').reset_index()
    return summary


def _format_stat(value):
    if pd.isna(value):
        return 'nan'
    return f"{value:.6f}"


def save_experiment_outputs(experiments_df, summary_df):
    os.makedirs('files/results', exist_ok=True)
    experiments_df.to_csv('files/results/experiments.csv', index=False)
    summary_df.to_csv('files/results/stats.csv', index=False)

    lines = []
    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['statistic']}: r2={_format_stat(row['r2'])}, mse={_format_stat(row['mse'])}, mad={_format_stat(row['mad'])}"
        )

    with open('files/results/stats.txt', 'w') as stats_file:
        stats_file.write("\n".join(lines))



file_Test = 'files/input/test_data.csv.zip'     #? Limpia los datos de prueba
file_Train = 'files/input/train_data.csv.zip'   #? Limpia los datos de entrenamient


with zipfile.ZipFile(file_Test, 'r') as zip:
    with zip.open("test_data.csv") as f:
        df_Test = pd.read_csv(f)


with zipfile.ZipFile(file_Train, 'r') as zip:
    with zip.open('train_data.csv') as f:
        df_Train = pd.read_csv(f)


df_Test = clean_data(df_Test)
df_Train = clean_data(df_Train)


x_train, y_train = df_Train.drop('Present_Price', axis = 1), df_Train['Present_Price']
x_test, y_test = df_Test.drop('Present_Price', axis = 1), df_Test['Present_Price']

model_pipeline = model()                                                                                 #? Crea el pipeline
model_pipeline = optimize_hyperparameters(model_pipeline, 10, x_train, y_train, 'neg_mean_absolute_error')    #? Optimiza los hiperparámetros


save_model(model_pipeline)

train_metrics, test_metrics = metrics(model_pipeline, x_train, y_train, x_test, y_test)                  #? Calcular y guardar las métricas

save_metrics([train_metrics, test_metrics])                                   #? Guardar las métricas y matrices de confusión

experiment_results = bootstrap_experiments(model_pipeline, x_test, y_test, iterations=100, random_state=42)
experiment_summary = summarize_experiments(experiment_results)
save_experiment_outputs(experiment_results, experiment_summary)
