import os
import tkinter as tk
from tkinter import messagebox, Toplevel
from googleapiclient.discovery import build
from textblob import TextBlob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import Counter
from wordcloud import WordCloud
import re
from transformers import pipeline
import pandas as pd

# Desactivar operaciones específicas de oneDNN (opcional)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configuración de la API de YouTube
api_key = 'TU_TOKEN_AQUI'
youtube = build('youtube', 'v3', developerKey=api_key)

# Cargar un pipeline de análisis de sentimientos de transformers con PyTorch
sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english', framework='pt')

# Función para obtener el título del video y los comentarios
def obtener_comentarios_y_titulo(video_id):
    comentarios = []
    fechas = []
    try:
        video_response = youtube.videos().list(part="snippet", id=video_id).execute()
        titulo = video_response['items'][0]['snippet']['title']
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo obtener el título del video: {e}")
        return "", [], []

    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText"
        )
        response = request.execute()

        while request is not None:
            for item in response['items']:
                comentario = item['snippet']['topLevelComment']['snippet']['textDisplay']
                fecha = item['snippet']['topLevelComment']['snippet']['publishedAt']
                comentarios.append(comentario)
                fechas.append(fecha)

            if 'nextPageToken' in response:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    pageToken=response['nextPageToken'],
                    maxResults=100,
                    textFormat="plainText"
                )
                response = request.execute()
            else:
                break
    except Exception as e:
        messagebox.showerror("Error", f"No se pudieron obtener los comentarios: {e}")
        return titulo, [], []

    return titulo, comentarios, fechas

# Función para analizar los sentimientos usando TextBlob
def analizar_sentimientos_textblob(comentarios):
    sentimientos = {'positivo': 0, 'neutral': 0, 'negativo': 0}
    for comentario in comentarios:
        analisis = TextBlob(comentario)
        if analisis.sentiment.polarity > 0:
            sentimientos['positivo'] += 1
        elif analisis.sentiment.polarity == 0:
            sentimientos['neutral'] += 1
        else:
            sentimientos['negativo'] += 1
    return sentimientos

# Función para analizar los sentimientos usando Transformers
def analizar_sentimientos_transformers(comentarios):
    sentimientos = {'positivo': 0, 'neutral': 0, 'negativo': 0}
    for comentario in comentarios:
        resultado = sentiment_analyzer(comentario)[0]
        if resultado['label'] == 'POSITIVE':
            sentimientos['positivo'] += 1
        elif resultado['label'] == 'NEGATIVE':
            sentimientos['negativo'] += 1
        else:
            sentimientos['neutral'] += 1
    return sentimientos

# Función para graficar resultados
def graficar_resultados(resultados, titulo="Análisis de Sentimientos de Comentarios de YouTube"):
    labels = list(resultados.keys())
    sizes = list(resultados.values())
    colors = ['green', 'grey', 'red']
    explode = (0.1, 0, 0)  # Solo "explota" el primer segmento
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', shadow=True, startangle=140)
    ax.axis('equal')  # Para que el gráfico sea un círculo
    ax.set_title(titulo)
    return fig

# Función para el análisis temporal de los sentimientos
def analizar_sentimientos_temporal(comentarios, fechas):
    if len(comentarios) != len(fechas):
        messagebox.showwarning("Advertencia", "Las listas de comentarios y fechas tienen longitudes diferentes. Se recortarán para igualarlas.")
        min_length = min(len(comentarios), len(fechas))
        comentarios = comentarios[:min_length]
        fechas = fechas[:min_length]

    # Crear el DataFrame
    df = pd.DataFrame({'comentario': comentarios, 'fecha': pd.to_datetime(fechas)})
    df['sentimiento'] = df['comentario'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['sentimiento'] = df['sentimiento'].apply(lambda x: 'positivo' if x > 0 else ('negativo' if x < 0 else 'neutral'))
    df.set_index('fecha', inplace=True)

    # Crear tabla dinámica
    df_pivot = df.pivot_table(index=df.index.date, columns='sentimiento', aggfunc='size', fill_value=0)

    fig, ax = plt.subplots()
    df_pivot.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title("Evolución de Sentimientos a lo largo del tiempo")
    return fig

# Función para el análisis de palabras clave
def analizar_palabras_clave(comentarios):
    palabras = ' '.join(comentarios).lower().split()
    conteo_palabras = Counter(palabras)
    return conteo_palabras

# Función para generar una nube de palabras
def generar_nube_palabras(conteo_palabras):
    wordcloud = WordCloud(width=800, height=400, max_words=100).generate_from_frequencies(conteo_palabras)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# Función para filtrar comentarios irrelevantes o spam
def filtrar_comentarios(comentarios):
    comentarios_filtrados = []
    for comentario in comentarios:
        # Eliminar URLs y menciones
        comentario = re.sub(r"http\S+|www\S+|@\S+", "", comentario)
        # Eliminar caracteres especiales
        comentario = re.sub(r"[^a-zA-Z0-9\s]", "", comentario)
        if len(comentario.split()) > 3:  # Excluir comentarios muy cortos
            comentarios_filtrados.append(comentario)
    return comentarios_filtrados

# Función para ejecutar el análisis desde la GUI
def ejecutar_analisis():
    video_id = entry_video_id.get()
    if not video_id:
        messagebox.showwarning("Error", "Por favor ingresa un ID de video.")
        return

    titulo, comentarios, fechas = obtener_comentarios_y_titulo(video_id)
    if not comentarios:
        return

    comentarios_filtrados = filtrar_comentarios(comentarios)

    # Análisis con TextBlob
    resultados_textblob = analizar_sentimientos_textblob(comentarios_filtrados)

    # Análisis con Transformers
    resultados_transformers = analizar_sentimientos_transformers(comentarios_filtrados)

    # Análisis temporal
    fig_temporal = analizar_sentimientos_temporal(comentarios_filtrados, fechas)

    conteo_palabras = analizar_palabras_clave(comentarios_filtrados)

    mostrar_resultados(titulo, resultados_textblob, resultados_transformers, len(comentarios), len(comentarios_filtrados))
    fig_textblob = graficar_resultados(resultados_textblob, titulo="Análisis de Sentimientos con TextBlob")
    fig_transformers = graficar_resultados(resultados_transformers, titulo="Análisis de Sentimientos con Transformers")
    fig_wordcloud = generar_nube_palabras(conteo_palabras)

    mostrar_graficos(fig_textblob, fig_transformers, fig_temporal, fig_wordcloud)

# Función para mostrar los resultados en la GUI
def mostrar_resultados(titulo, resultados_textblob, resultados_transformers, total_comentarios, comentarios_filtrados):
    resultado_text = (f"Título del video: {titulo}\n"
                      f"Total de comentarios: {total_comentarios}\n"
                      f"Comentarios filtrados: {comentarios_filtrados}\n"
                      f"\n--- Análisis con TextBlob ---\n"
                      f"Comentarios positivos: {resultados_textblob['positivo']}\n"
                      f"Comentarios neutrales: {resultados_textblob['neutral']}\n"
                      f"Comentarios negativos: {resultados_textblob['negativo']}\n"
                      f"\n--- Análisis con Transformers ---\n"
                      f"Comentarios positivos: {resultados_transformers['positivo']}\n"
                      f"Comentarios neutrales: {resultados_transformers['neutral']}\n"
                      f"Comentarios negativos: {resultados_transformers['negativo']}")
    lbl_resultados.config(text=resultado_text)

# Función para mostrar los gráficos en una nueva ventana
def mostrar_graficos(fig_textblob, fig_transformers, fig_temporal, fig_wordcloud):
    graficos_window = Toplevel(root)
    graficos_window.title("Gráficos de Análisis de Sentimientos")
    graficos_window.geometry("1200x800")
    graficos_window.configure(bg='#1b1b1b')

    # Crear un frame para los gráficos
    frame_graficos = tk.Frame(graficos_window, bg='#1b1b1b')
    frame_graficos.pack(expand=True, fill='both')

    # Configurar columnas y filas
    frame_graficos.columnconfigure(0, weight=1)
    frame_graficos.columnconfigure(1, weight=1)
    frame_graficos.rowconfigure(0, weight=1)
    frame_graficos.rowconfigure(1, weight=1)
    frame_graficos.rowconfigure(2, weight=1)

    # Mostrar los gráficos en una cuadrícula
    display_chart(fig_textblob, frame_graficos, row=0, column=0)
    display_chart(fig_transformers, frame_graficos, row=0, column=1)
    display_chart(fig_temporal, frame_graficos, row=1, column=0, columnspan=2)
    display_chart(fig_wordcloud, frame_graficos, row=2, column=0, columnspan=2)

# Función para mostrar los gráficos en la GUI usando FigureCanvasTkAgg
def display_chart(fig, window, row, column, columnspan=1):
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().grid(row=row, column=column, columnspan=columnspan, padx=10, pady=10)

# Configuración de la interfaz gráfica (GUI)
root = tk.Tk()
root.title("Análisis de Sentimientos en YouTube")
root.geometry("600x600")
root.configure(bg='#1b1b1b')

# Estilo de los widgets
font_label = ('Helvetica', 14, 'bold')
font_entry = ('Helvetica', 12)
font_button = ('Helvetica', 12, 'bold')
bg_color = '#1b1b1b'
fg_color = '#ffffff'
button_color = '#ff6600'
button_fg_color = '#ffffff'

tk.Label(root, text="ID del video de Youtube", font=font_label, bg=bg_color, fg=fg_color).pack(pady=20)
entry_video_id = tk.Entry(root, width=30, font=font_entry, bd=1, relief='solid')
entry_video_id.pack(pady=10)

btn_analizar = tk.Button(root, text="Analizar emociones", font=font_button, bg=button_color, fg=button_fg_color, command=ejecutar_analisis)
btn_analizar.pack(pady=20)

lbl_resultados = tk.Label(root, text="", font=font_entry, bg=bg_color, fg=fg_color, justify="left")
lbl_resultados.pack(pady=20)

root.mainloop()
