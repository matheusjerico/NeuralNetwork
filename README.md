# Tarefa 1: Aprendizado Supervisionado

**Autor**: Matheus Jericó Palhares <br>
**LinkedIn**: https://linkedin.com/in/matheusjerico <br>
**Github**: https://github.com/matheusjerico

## 1) Tarefa:
**1. Utilizar apenas as colunas age, chol, thalach e target;** <br>
**2. Separar aleatoriamente e de forma equilibrada o dataset, utilizando 70% como conjunto de treinamento;**<br>
**3. Utilizar, também de forma equilibrada, 15% para validação e 15% para teste;**  <br>
**4. Implementar o processo de treinamento e validação do modelo, plotando as curvas com as evoluções dos erros de treinamento e de validação;**<br>
**5. Pesquisar a respeito de matrizes de confusão para classificação binária, gere uma relativa ao seu conjunto de testes e calcule o máximo de métricas que você julgar interessantes para avaliar seu modelo.**

### 1. Utilizar apenas as colunas *age*, *chol*, *thalach* e *target*

```python
df = dataset[['age', 'chol', 'thalach', 'target']].copy()
df.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>chol</th>
      <th>thalach</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>233</td>
      <td>150</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
      <td>250</td>
      <td>187</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>204</td>
      <td>172</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56</td>
      <td>236</td>
      <td>178</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57</td>
      <td>354</td>
      <td>163</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### 2. Separando dados de treino (70%), teste (15%) e validação (15%)


```python
X = df.drop(columns='target')
y = df['target'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, stratify=y, random_state = 43)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5, stratify= y_test,random_state = 43)
```

    Quantidade de registros e dimensão X: (303, 3)
    Quantidade de registros e dimensão y: (303,)
    
    Quantidade de registros e dimensão X_train: (212, 3)
    Quantidade de registros e dimensão y_train: (212,)
    
    Quantidade de registros e dimensão X_test: (45, 3)
    Quantidade de registros e dimensão y_test: (45,)
    
    Quantidade de registros e dimensão X_val: (46, 3)
    Quantidade de registros e dimensão y_val: (46,)
    


### 3. Criando Rede Neural com tensorflow 2.0 


```python
def create_model(input_dim = 3):
    model = Sequential([
               tf.keras.layers.Dense(units=16, 
                                     activation = 'relu', 
                                     kernel_initializer = 'normal',
                                     input_dim=input_dim),
               tf.keras.layers.Dropout(0.2),
               tf.keras.layers.Dense(units=16,
                                     activation = 'relu',
                                     kernel_initializer = 'normal'),
               tf.keras.layers.Dropout(0.2),
               tf.keras.layers.Dense(units=1, activation = 'sigmoid')])

    return model
```


```python
model = create_model(input_dim=3)
otimizador = tf.keras.optimizers.Adam(lr = 0.01, decay = 0.0001, clipvalue = 0.5)
```

### 4. Normalizando os dados


```python
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)
```

### 5. Treinamento da Rede Neural


```python
EPOCHS = 50
BS = 12

model.compile(optimizer = otimizador, loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])

H = model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs = EPOCHS, 
              batch_size = BS,      
              shuffle=True)

model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense (Dense)                (None, 16)                64        
    _________________________________________________________________
    dropout (Dropout)            (None, 16)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 16)                272       
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 16)                0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 17        
    =================================================================
    Total params: 353
    Trainable params: 353
    Non-trainable params: 0
    _________________________________________________________________

### 6. Gráfico de evolução das métricas de treinamento e validação

![png](imagens/output_32_1.png)

![png](imagens/output_33_1.png)


### 7. Predição
```python
y_pred = model.predict_classes(X_test)
```

### 8. Métricas

#### 8.1. Matriz de Confusão

![png](imagens/output_36_0.png)


#### 8.2. Precision & Accuracy

    Precision: 0.6666666666666666

    Accuracy: 0.5777777777777777


#### 8.3. ROC AUC

    ROC AUC Score: 0.59



#### 8.4. F1-Score

    F1-Score: 0.5581395348837209


#### 8.5. Tudo junto & Misturado (Precision, Recall e F1-score)

                  precision    recall  f1-score   support
    
               0       0.52      0.70      0.60        20
               1       0.67      0.48      0.56        25
    
        accuracy                           0.58        45
       macro avg       0.59      0.59      0.58        45
    weighted avg       0.60      0.58      0.57        45
    
---
---

# DESAFIO
**Utilizando todos os dados do dataset**

### 1. Separação dos dados


```python
df = dataset.copy()
X = df.drop(columns='target')
y = df['target'].copy()

X.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>1</td>
      <td>3</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>1</td>
      <td>187</td>
      <td>0</td>
      <td>3.5</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>0</td>
      <td>1</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>0</td>
      <td>172</td>
      <td>0</td>
      <td>1.4</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56</td>
      <td>1</td>
      <td>1</td>
      <td>120</td>
      <td>236</td>
      <td>0</td>
      <td>1</td>
      <td>178</td>
      <td>0</td>
      <td>0.8</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>354</td>
      <td>0</td>
      <td>1</td>
      <td>163</td>
      <td>1</td>
      <td>0.6</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>


### 2. Separando dados de treino (70%), teste (15%) e validação (15%)

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, stratify=y, random_state = 43)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5, stratify= y_test,random_state = 43)
```

### 3. Criação do modelo

```python
model = create_model(input_dim=13)
otimizador = tf.keras.optimizers.Adam(lr = 0.01, decay = 0.0001, clipvalue = 0.5)
```

### 3. Normalizando os dados

```python
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)
```

### 4. Treinamento da Rede Neural


```python
EPOCHS = 50
BS = 12

model.compile(optimizer = otimizador, loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])

H = model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs = EPOCHS, 
              batch_size = BS,      
              shuffle=True)
```

### 5. Predição

```python
y_pred = model.predict_classes(X_test)
```

### 6. Gráfico de evolução de métricas de treinamento e validação

![png](imagens/output_62_1.png)

![png](imagens/output_63_1.png)


### 7. Métricas

#### 7.1. Matriz de Confusão
![png](imagens/output_65_0.png)

#### 7.2. Precision, Recall e F1-Score
                  precision    recall  f1-score   support
    
               0       0.75      0.75      0.75        20
               1       0.80      0.80      0.80        25
    
        accuracy                           0.78        45
       macro avg       0.78      0.78      0.78        45
    weighted avg       0.78      0.78      0.78        45

#### 7.3. Accuracy

    Accuracy: 0.7777777777777778


**Análise:** 
- O desempenho melhorou consideravelmente analisando a matriz de confusão, acurácia, recall e f1-score.
- O incremento na quantidade de features mostrou melhorar o desempenho do modelo de redes neurais.  
