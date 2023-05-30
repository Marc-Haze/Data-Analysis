import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import base64

model_eight = load_model('./models/eighthr_model/keras_ozone.h5')
model_one = joblib.load('./models/onehr_model/Ozone_Forest_Model.pkl')
scaler = joblib.load('./scaler/StandardScaler.pkl')

# Lista de columnas
columnas = ['Date', 'WSR0', 'WSR1', 'WSR2', 'WSR3', 'WSR4', 'WSR5', 'WSR6', 'WSR7', 'WSR8', 'WSR9', 'WSR10', 'WSR11',
            'WSR12', 'WSR13', 'WSR14', 'WSR15', 'WSR16', 'WSR17', 'WSR18', 'WSR19', 'WSR20', 'WSR21', 'WSR22', 'WSR23',
            'WSR_PK', 'WSR_AV', 'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13',
            'T14', 'T15', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T_PK', 'T_AV', 'T85', 'RH85', 'U85',
            'V85', 'HT85', 'T70', 'RH70', 'U70', 'V70', 'HT70', 'T50', 'RH50', 'U50', 'V50', 'HT50', 'KI', 'TT', 'SLP',
            'SLP_', 'Precp']

def process_data(data, data_source):
    # Leer archivo .data
    df = pd.read_csv(data, names=columnas)
    
    # Eliminar columnas no necesarias
    cols_to_drop = ['Date'] + [f'WSR{i}' for i in range(24)] + [f'T{i}' for i in range(24)]
    df = df.drop(columns=cols_to_drop)

    # Reemplazar '?' con NaN
    df = df.replace('?', np.nan)

    # Convertir a tipo float
    df = df.astype(float)

    # Rellenar valores faltantes con la media
    df = df.fillna(df.mean())

    # Escalamiento de los datos
    df_scaled = scaler.transform(df)

    # PCA para reducir a 4 componentes (one_data) o 12 componentes (eight_data)
    if data_source == 'one_data':
        pca = PCA(n_components=4)
    else:
        pca = PCA(n_components=12)

    df_pca = pca.fit_transform(df_scaled)
    df_pca = pd.DataFrame(df_pca)
    return df_pca

def make_predictions(model,data,data_source):
    data = pd.DataFrame(data)
    if data_source == 'one_data':
        predictions = model.predict(data)
        df_predictions = pd.DataFrame(predictions, columns=['Predicciones'])
        df_predictions.to_csv('predictions_1h.csv', index=False)
        st.success('Predicciones realizadas y guardadas en predictions_1h.csv')
        download_link(pd.read_csv('predictions_1h.csv'), 'predictions_1h.csv')
        st.success('Estas son algunas de las predicciones')
        st.write(df_predictions.head(5)) 
    else:
        predictions = model.predict(data)
        predictions = np.round(predictions).flatten().astype(int)
        df_predictions = pd.DataFrame(predictions, columns=['Predicciones'])
        df_predictions.to_csv('predictions_8h.csv', index=False)
        st.success('Predicciones realizadas y guardadas en predictions_8h.csv')
        download_link(pd.read_csv('predictions_8h.csv'), 'predictions_8h.csv')
        st.success('Estas son algunas de las predicciones')
        st.write(df_predictions.head(5))

    return df_predictions

def download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Codificar en base64
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Descargar archivo CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

# Streamlit app
def main():
    st.title('Predicción de Datos')

    # Botón para cargar archivo one_data
    one_data = st.file_uploader('Cargar Registros - Muestreo de 1h', type='.data')

    # Botón para cargar archivo eight_data
    eight_data = st.file_uploader('Cargar Registros - Muestreo de 8h', type='.data')

    # Procesar eight_data y hacer predicciones
    if eight_data is not None:
        eight_df = process_data(eight_data,'eight_data')
        make_predictions(model_eight, eight_df,'eight_data')

    # Procesar one_data y hacer predicciones
    if one_data is not None:
        one_df = process_data(one_data,'one_data')
        make_predictions(model_one, one_df,'one_data')


if __name__ == '__main__':
    main()