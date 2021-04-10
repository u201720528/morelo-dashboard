import itertools

from flask import Flask, request, render_template
from sklearn import preprocessing, model_selection, linear_model
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score
from werkzeug.utils import secure_filename
import os
import io
import pandas as pd
import matplotlib.pyplot as plt
import base64
from matplotlib import pyplot
import requests
import numpy as np

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('home.html')

@app.route('/modelo', methods=['GET'])
def modelo():
    ModelData = pd.read_csv("MainData.csv")
    print(ModelData.head())
    ModelData['clasificacion'] = ModelData['clasificacion'].astype('int')

    X = np.array(ModelData.drop(['clasificacion'], 1))
    y = np.array(ModelData['clasificacion'])
    #f = X.shape
    # print(f)

    X = preprocessing.StandardScaler().fit(X).transform(X)

    validation_size = 0.2
    seed = 4

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=validation_size,
                                                                        random_state=seed)

    print('Train set:', X_train.shape, Y_train.shape)
    print('Test set:', X_test.shape, Y_test.shape)

    # C=1.0
    model = linear_model.LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
                                            intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                                            penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                                            verbose=0, warm_start=False)
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)

    prescision = round(precision_score(Y_test, predictions)*100,2)
    recall = round(recall_score(Y_test, predictions)*100,2)
    accuracy = round(accuracy_score(Y_test, predictions)*100,2)
    cantidadMuestra = X_train.shape[0] + X_test.shape[0]
    cantidadEntrenamiento = X_train.shape[0]
    cantidadTest = X_test.shape[0]

    cnf_matrix = confusion_matrix(Y_test, predictions, labels=[1, 0])
    plt.figure()
    plot_url1 = plot_confusion_matrix(cnf_matrix, classes=['churn=1', 'churn=0'], normalize=False, title='Matriz de confusión')
    return render_template('prediccion.html', imagen={'imagen1': plot_url1,'prescision':prescision,'recall':recall,'accuracy':accuracy,'cantidadMuestra':cantidadMuestra
                                                      ,"cantidadEntrenamiento":cantidadEntrenamiento,"cantidadTest":cantidadTest})

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de confusión normalizada")
    else:
        print('Matriz de confusión sin normalización')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Real')
    plt.xlabel('Predicción')
    img = io.BytesIO()
    pyplot.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url


@app.route('/grafica', methods=['GET'])
def grafica():
    df = pd.read_csv("hurtos.csv")
    headers = ["numero_cliente", "fecha_inspeccion", "comuna", "distrito", "actividad", "actividad_descripcion",
               "categoria", "giro_suministro", "tarifa", "clave_tarifa", "latitud", "longitud", "tipo_causal",
               "inf_disponible", "sucursal", "fecha_inicio", "fecha_fin", "fecha_creacion", "meses", "f1", "f2", "f3",
               "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19",
               "f20", "f21", "f22", "f23", "f24"
        , "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
               "d18", "d19", "d20", "d21", "d22", "d23", "d24"
        , "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11", "c12", "c13", "c14", "c15", "c16", "c17",
               "c18", "c19", "c20", "c21", "c22", "c23", "c24"]

    df.columns = headers
    df.dropna(subset=["comuna"], axis=0, inplace=True)
    moda_actividad = df["actividad"].mode()[0]
    df["actividad"].replace(0, moda_actividad, inplace=True)
    moda_actividad_desc = df["actividad_descripcion"].mode()[0]
    df["actividad_descripcion"].replace("", moda_actividad_desc, inplace=True)
    plot_url1 = grafica1(df)
    plot_url2 = grafica2(df)
    plot_url3 = grafica3(df)
    return render_template('grafica.html', imagen={ 'imagen1': plot_url1, 'imagen2': plot_url2, 'imagen3': plot_url3 })

def grafica1(df):
    fig, ax = pyplot.subplots(figsize =(17, 20))
    ax.barh(df['distrito'].unique().tolist(), df["distrito"].value_counts(), align='center')
    ax.grid(b = True, color ='grey',linestyle ='-.', linewidth = 0.5, alpha = 0.2)
    for i in ax.patches:
        pyplot.text(i.get_width()+0.2, i.get_y()+0.25, str(round((i.get_width()), 2)),fontsize = 10, fontweight ='bold',color ='grey')
    pyplot.xlabel("Distrito")
    pyplot.ylabel("Cantidad")
    pyplot.title("Hurto por Distrito")
    img = io.BytesIO()
    pyplot.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()  
    return  plot_url

def grafica2(df):
    fig, ax = pyplot.subplots(figsize =(11, 10))
    ax.barh(df['sucursal'].unique().tolist(), df["sucursal"].value_counts(), align='center')
    ax.grid(b = True, color ='grey',linestyle ='-.', linewidth = 0.5, alpha = 0.2)
    for i in ax.patches:
        pyplot.text(i.get_width()+0.2, i.get_y()+0.25, str(round((i.get_width()), 2)),fontsize = 10, fontweight ='bold',color ='grey')
    pyplot.xlabel("Sucursal")
    pyplot.ylabel("Cantidad")
    pyplot.title("Hurto por Sucursal")
    img = io.BytesIO()
    pyplot.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()  
    return  plot_url

def grafica3(df):
    fig, ax = pyplot.subplots(figsize =(11, 5))
    ax.barh(df['tarifa'].unique().tolist(), df["tarifa"].value_counts(), align='center')
    ax.grid(b = True, color ='grey',linestyle ='-.', linewidth = 0.5, alpha = 0.2)
    for i in ax.patches:
        pyplot.text(i.get_width()+0.2, i.get_y()+0.25, str(round((i.get_width()), 2)),fontsize = 10, fontweight ='bold',color ='grey')
    pyplot.xlabel("Distrito")
    pyplot.ylabel("Cantidad")
    pyplot.title("Hurto por Tarifa")
    
    img = io.BytesIO()
    pyplot.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()  
    return  plot_url

#if __name__ == "__main__":
#    app.run(port = 80, debug = True)

