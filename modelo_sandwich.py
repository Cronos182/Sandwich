import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('sanguchez.csv', sep=';', header=0, encoding='latin-1')#Lee csv con cambio de codificacion
df = df[['Ingredientes', 'nota']]#Filtra columnas de interes
df.dropna(inplace = True)#Elimina valores N/A
df['aprobacion'] = [1 if x > 2 else 0 for x in df['nota']]#Agrega columna con evaluacion buena o mala (1 o 0) a partir de las notas
print(df)
#df['seq'] = df.groupby('nota').cumcount()
#df.sort_values(['seq', 'nota'], inplace= True)
#df.reset_index(drop= True, inplace = True)
sentences = df['Ingredientes'].values#Asigna recetas
y = df['aprobacion'].values#Asigna evaluacion de recetas
#Divide dataset, 75% datos entrenamiento y 25% datos test
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state = 1000)
print(df.shape, sentences_train.shape, sentences_test.shape, y_train.shape, y_test.shape)

vectorizer = CountVectorizer(strip_accents='ascii')#Asigna funcion vectorizar con preprocesado sin acento y minuscula
#vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)#Crea indices diccionario de palabras a partir de recetas de entrenamiento
print(vectorizer.vocabulary_)#Imprime diccionario
print(len(vectorizer.vocabulary_))
X_train = vectorizer.transform(sentences_train)#Crea vectores caracteristicos de recetas de entrenamiento
X_test  = vectorizer.transform(sentences_test)#Crea vectores caracteristicos de recetas de prueba
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(type(y_test))

classifier = LogisticRegression()#Asigna modelo de ajuste binario
classifier.fit(X_train, y_train)#Ajusta modelo binario con vetores caracteristicos y evaluacion de recetas de entrenamiento 
y_pred = classifier.predict(X_test)#Con modelo ajustado se predice evaluacion de recetas de pruebas
score = classifier.score(X_test, y_test)#Se evalua la predicion de evaluacion del modelo con la evaluacion real de las recetas de prueba

print("Precisión:", score)#Se imprime indice de aciertos del modelo respecto a la evaluacion real