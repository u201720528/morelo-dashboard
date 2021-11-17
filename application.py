import itertools
import pickle

from flask import Flask, request, render_template
from sklearn import preprocessing, model_selection, linear_model
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score
from services.service import login,programar
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from scipy import stats
from werkzeug.utils import secure_filename
import os
import io
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import base64
from matplotlib import pyplot
import numpy as np

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('home.html')


@app.route('/prediccion', methods=['GET'])
def prediccion():
    return render_template('nuevoscasos.html')

@app.route('/archivo', methods=['POST'])
def archivo():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(filename)

    url = "septiembre_evalua.csv"
    ModelData = ProcesarInput(1, url, True)

    ModelData = CalcularPendiente(ModelData)
    # dfDistrito = pd.get_dummies(ModelData["distrito"])
    # ModelData = pd.concat([ModelData, dfDistrito], axis=1, join="inner")
    distrito = ["Cañete", "Chacarilla", "Chorrillos", "Chosica", "La República", "Miraflores", "Pedro Miotta",
                    "San Bartolo", "San Juan", "Santa Anita"]
    possibilites = {"sucursal":distrito}

    giro = ["00000", "A0111", "A0112", "A0113", "A0121", "A0122", "A0130", "A0140", "C1320", "D1511", "D1520", "D1531",
            "D1533", "D1541", "D1543", "D1549", "D1600", "D1711", "D1712", "D1721", "D1729", "D1730", "D1810", "D1820",
            "D1920", "D2000", "D2010", "D2029", "D2222", "D2413", "D2423", "D2424", "D2500", "D2519", "D2520", "D2691",
            "D2692", "D2693", "D2720", "D2811", "D2899", "D3599", "D3610", "D3699", "D3710", "E4000", "E4010", "E4100",
            "F4500", "F4510", "F4520", "F4530", "F4550", "G5000", "G5010", "G5020", "G5030", "G5040", "G5050", "G5121",
            "G5122", "G5131", "G5139", "G5141", "G5143", "G5150", "G5190", "G5200", "G5211", "G5219", "G5220", "G5231",
            "G5252", "G5259", "G5260", "G5270", "H5500", "H5510", "H5520", "I6010", "I6021", "I6110", "I6300", "I6302",
            "I6303", "I6400", "I6420", "J6511", "J6519", "J6601", "J6720", "K7000", "K7010", "K7020", "K7129", "K7400",
            "K7411", "K7414", "K7430", "K7499", "L7500", "L7511", "L7522", "L7523", "M8000", "M8010", "M8020", "M8021",
            "M8022", "M8030", "M8090", "N8510", "N8511", "N8512", "N8519", "N8520", "N8532", "O9000", "O9100", "O9111",
            "O9112", "O9120", "O9191", "O9199", "O9200", "O9213", "O9219", "O9241", "O9249", "O9300", "O9301", "O9302",
            "O9309", "P9500", "Q9900"]
    possibilitesGiro = {"giro_suministro": giro}
    data = {'Name': ['Jai', 'Princi', 'Gaurav', 'Anuj']}

    # Convert the dictionary into DataFrame
    df = pd.DataFrame(data)

    dfSuc = pd.DataFrame(possibilites)
    dfSucursal = pd.get_dummies(dfSuc["sucursal"])

    for f in distrito:
        dfSucursal[f] = dfSucursal[f].astype('int')

    dfSucursalFinal = pd.concat([dfSuc, dfSucursal], axis=1, join="inner")
    #Prueba = pd.merge(ModelData, dfSucursalFinal, how='left', on=['sucursal'])
    ModelData = pd.merge(ModelData, dfSucursalFinal, how='left', on=['sucursal'])
    for f in distrito:
        ModelData[f] = ModelData[f].replace(np.nan, 0)

    dfGir = pd.DataFrame(possibilitesGiro)
    dfGiro = pd.get_dummies(dfGir["giro_suministro"])

    for f in giro:
        dfGiro[f] = dfGiro[f].astype('int')

    dfGiroFinal = pd.concat([dfGir, dfGiro], axis=1, join="inner")
    ModelData = pd.merge(ModelData, dfGiroFinal, how='left', on=['giro_suministro'])
    # ModelData['DataFrame Column'] = df['DataFrame Column'].fillna(0)
    for f in giro:
        ModelData[f] = ModelData[f].replace(np.nan, 0)

    LimpiarColumnas(ModelData)

    loaded_model = pickle.load(open('lds_model2.sav', 'rb'))

    ModelData['clasificacion'] = ModelData['clasificacion'].astype('int')

    X = np.array(ModelData.drop(['clasificacion'], 1))
    y = np.array(ModelData['clasificacion'])

    predictions = loaded_model.predict(X)

    token = Token()
    linea = 0
    with open('septiembre_evalua.csv', 'r') as f:
        for line in f:
            CargaWS(line, token, y[linea])
            linea = linea + 1


    return render_template('cargado.html')

def Token():
    data = {
        'email': "cesartaira86@gmail.com",
        'password': "123456"
    }
    token_response = login(data)
    if token_response is not None:
        token_split = str(token_response.content).split("'")
        token = 'Bearer ' + token_split[1]
        return token
    return ""

def CargaWS(contents, token, prediccion):
    data = {
        'email': "cesartaira86@gmail.com",
        'password': "123456"
    }
    headers = {
        'Authorization': token
    }
    contentsArreglo = contents.split(",")
    data2 = {"agregarProgramacionDto": {},
             "codReferencia": contentsArreglo[0],
             "fecha": "2021-11-17",
             "email": "cesartaira86@gmail.com",
             "latitud": contentsArreglo[10],
             "longitud": contentsArreglo[11],
             "tipo": "24",
             "actor": "1",
             "persona": "19",
             "prediccion": str(prediccion)
             }
    programar(headers, data2)
    return token

def ProcesarInput(valorHurto,archivo,evalua):
    df = pd.read_csv(archivo, header=None)
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
    if not evalua:
        df.dropna(subset=["comuna"], axis=0, inplace=True)
    df['comuna'] = df['comuna'].astype('int')


    missing_data = df.isnull()
    missing_data.head(5)

    if not evalua:
        moda_actividad = df["actividad"].mode()[0]
        df["actividad"].replace(0, moda_actividad, inplace=True)

        moda_actividad_desc = df["actividad_descripcion"].mode()[0]
        df["actividad_descripcion"].replace("", moda_actividad_desc, inplace=True)

    # for column in missing_data.columns.values.tolist():
    #     print (missing_data[column].value_counts())
    #     print("")

    df['pendiente'] = 0
    df['pendiente12'] = 0
    df['pendiente6'] = 0
    df['cantfN'] = 0
    df['cantfT'] = 0
    df['cantfU'] = 0
    df['cantf'] = 0
    #df['max12'] = 0
    #df['min12'] = 0
    #df['min12'] = 0
    df['var'] = 0

    classification = []
    for index in df.iterrows():
        classification.append(valorHurto)
    df['clasificacion'] = classification

    # df.head()
    for index, row in df.iterrows():
        cantfN = 0
        cantfT = 0
        cantfU = 0
        cantf = 0
        for i in range(24):
            if row[8 + i] == 'N':
                df.loc[index, 'f' + str(i + 1)] = 0
                cantfN = cantfN + 1
            elif row[8 + i] == '*':
                df.loc[index, 'f' + str(i + 1)] = 1
                cantfT = cantfT + 1
            elif row[8 + i] == 'U':
                df.loc[index, 'f' + str(i + 1)] = 2
                cantfU = cantfU + 1
            else:
                df.loc[index, 'f' + str(i + 1)] = 3
                cantf = cantf + 1
        df.loc[index, 'cantfN'] = cantfN
        df.loc[index, 'cantfT'] = cantfT
        df.loc[index, 'cantfU'] = cantfU
        df.loc[index, 'cantf'] = cantf

    for i in range(24):
        df[["f" + str(i+1)]] = df[["f" + str(i+1)]].astype("int32")

    return df


def CalcularPendiente(df):
    mes = []
    for index in range(24):
        mes.append(index + 1)

    dfc = df.copy()

    print("Fila: " + str(len(dfc.index)))
    for index, row in dfc.iterrows():
        if (index % 1000 == 0):
            print("Van " + str(index))
        consumo = []
        for i in range(24):
            consumo.append(row['c' + str(i + 1)])
        if max(consumo) > 0:
            maximoConsumo = max(consumo)
        for i in range(24):
            dfc.loc[index, 'c' + str(i + 1)] = row['c' + str(i + 1)] / maximoConsumo

        X = []
        Y = []
        for i in range(24):
            X.append(i + 1)
            Y.append(row['c' + str(i + 1)])

        slope, intercept, r, p, std_err = stats.linregress(X, Y)
        if (max(consumo) > 0):
            dfc.loc[index, 'pendiente'] = slope
        if max(consumo) > 0:
            dfc.loc[index, 'var'] = (max(consumo) - min(consumo))/max(consumo)

        X = []
        Y = []
        for i in range(12):
            X.append(i + 1)
            Y.append(row['c' + str(i + 11 + 1)])

        slope, intercept, r, p, std_err = stats.linregress(X, Y)
        if (max(consumo) > 0):
            dfc.loc[index, 'pendiente12'] = slope

        X = []
        Y = []
        for i in range(6):
            X.append(i + 1)
            Y.append(row['c' + str(i + 17 + 1)])

        slope, intercept, r, p, std_err = stats.linregress(X, Y)
        if (max(consumo) > 0):
            dfc.loc[index, 'pendiente6'] = slope
    return dfc



def LimpiarColumnas(df):
    df.drop("distrito", axis="columns", inplace=True)
    df.drop("comuna", axis="columns", inplace=True)
    df.drop("categoria", axis="columns", inplace=True)
    df.drop("actividad", axis="columns", inplace=True)
    df.drop("actividad_descripcion", axis="columns", inplace=True)
    df.drop("giro_suministro", axis="columns", inplace=True)
    df.drop("latitud", axis="columns", inplace=True)
    df.drop("longitud", axis="columns", inplace=True)
    df.drop("clave_tarifa", axis="columns", inplace=True)
    df.drop("tipo_causal", axis="columns", inplace=True)
    df.drop("inf_disponible", axis="columns", inplace=True)
    df.drop("sucursal", axis="columns", inplace=True)
    df.drop("fecha_creacion", axis="columns", inplace=True)
    df.drop("fecha_fin", axis="columns", inplace=True)
    df.drop("fecha_inicio", axis="columns", inplace=True)
    df.drop("fecha_inspeccion", axis="columns", inplace=True)
    df.drop("numero_cliente", axis="columns", inplace=True)
    df.drop("tarifa", axis="columns", inplace=True)
    df.drop("meses", axis="columns", inplace=True)

    for i in range(24):
        df.drop("c" + str(i + 1), axis="columns", inplace=True)

    for i in range(24):
        df.drop("f" + str(i + 1), axis="columns", inplace=True)

    for i in range(24):
        df.drop("d" + str(i + 1), axis="columns", inplace=True)

@app.route('/modelo', methods=['GET'])
def modelo():
    ModelData = pd.read_csv("MainData.csv")
    print(ModelData.head())
    ModelData['clasificacion'] = ModelData['clasificacion'].astype('int')

    X = np.array(ModelData.drop(['clasificacion'], 1))
    y = np.array(ModelData['clasificacion'])
    f = X.shape
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

    prescision = round(precision_score(Y_test, predictions) * 100, 2)
    recall = round(recall_score(Y_test, predictions) * 100, 2)
    accuracy = round(accuracy_score(Y_test, predictions) * 100, 2)
    cantidadMuestra = X_train.shape[0] + X_test.shape[0]
    cantidadEntrenamiento = X_train.shape[0]
    cantidadTest = X_test.shape[0]

    pickle.dump(model, open("lds_model.sav", "wb"))

    cnf_matrix = confusion_matrix(Y_test, predictions, labels=[1, 0])
    plt.figure()
    plot_url1 = plot_confusion_matrix(cnf_matrix, classes=['churn=1', 'churn=0'], normalize=False,
                                      title='Matriz de confusión')
    return render_template('prediccion.html', imagen={'imagen1': plot_url1, 'prescision': prescision, 'recall': recall,
                                                      'accuracy': accuracy, 'cantidadMuestra': cantidadMuestra
        , "cantidadEntrenamiento": cantidadEntrenamiento, "cantidadTest": cantidadTest})


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Esta función muestra y dibuja la matriz de confusión.
    La normalización se puede aplicar estableciendo el valor `normalize=True`.
    """
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
    # if request.method == 'POST':
    # da = request.files['file']
    # filename = secure_filename(da.filename)
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

    moda_actividad_desc = df["actividad_descripcion"].mode()[0]
    df["actividad_descripcion"].replace(np.nan, moda_actividad_desc, inplace=True)

    plot_url1 = grafica1(df)
    plot_url2 = grafica2(df)
    plot_url3 = grafica3(df)
    plot_url4 = grafica4(df)
    return render_template('grafica.html', imagen={'imagen1': plot_url1, 'imagen2': plot_url2, 'imagen3': plot_url3,
                                                   'imagen4': plot_url4})


def grafica1(df):
    # Lo guarda en la carpeta info
    # da.save(os.path.join(app.config["data"], filename))
    # df = pd.read_csv('./info/{}'.format(filename))

    fig, ax = pyplot.subplots(figsize=(17, 20))
    ax.barh(df['distrito'].unique().tolist(), df["distrito"].value_counts(), align='center')
    ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)
    for i in ax.patches:
        pyplot.text(i.get_width() + 0.2, i.get_y() + 0.25, str(round((i.get_width()), 2)), fontsize=10,
                    fontweight='bold', color='grey')
        # establece las etiquetas x/y y muestra el título 
    pyplot.xlabel("Distrito")
    pyplot.ylabel("Cantidad")
    pyplot.title("Hurto por Distrito")

    img = io.BytesIO()
    pyplot.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url


def grafica2(df):
    fig, ax = pyplot.subplots(figsize=(11, 10))
    ax.barh(df['sucursal'].unique().tolist(), df["sucursal"].value_counts(), align='center')
    ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)
    for i in ax.patches:
        pyplot.text(i.get_width() + 0.2, i.get_y() + 0.25, str(round((i.get_width()), 2)), fontsize=10,
                    fontweight='bold', color='grey')
    # establece las etiquetas x/y y muestra el título 
    pyplot.xlabel("Sucursal")
    pyplot.ylabel("Cantidad")
    pyplot.title("Hurto por Sucursal")

    img = io.BytesIO()
    pyplot.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url


def grafica3(df):
    fig, ax = pyplot.subplots(figsize=(11, 5))
    ax.barh(df['tarifa'].unique().tolist(), df["tarifa"].value_counts(), align='center')
    ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)
    for i in ax.patches:
        pyplot.text(i.get_width() + 0.2, i.get_y() + 0.25, str(round((i.get_width()), 2)), fontsize=10,
                    fontweight='bold', color='grey')
    # establece las etiquetas x/y y muestra el título 
    pyplot.xlabel("Distrito")
    pyplot.ylabel("Cantidad")
    pyplot.title("Hurto por Tarifa")

    img = io.BytesIO()
    pyplot.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url


def grafica4(df):
    fig, ax = pyplot.subplots(figsize=(17, 20))
    ax.barh(df['actividad_descripcion'].unique().tolist(), df["actividad_descripcion"].value_counts(), align='center')
    ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)
    for i in ax.patches:
        pyplot.text(i.get_width() + 0.2, i.get_y() + 0.25, str(round((i.get_width()), 2)), fontsize=10,
                    fontweight='bold', color='grey')
    # establece las etiquetas x/y y muestra el título
    pyplot.xlabel("Actividad")
    pyplot.ylabel("Cantidad")
    pyplot.title("Hurto por Actividad")

    img = io.BytesIO()
    pyplot.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url


if __name__ == "__main__":
    app.run(port=80, debug=True)
