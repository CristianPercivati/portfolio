# About me

# Education

- Advanced technician diploma in Data Science and AI - IFTS 24
- Computer Science engineering student - University of Buenos Aires

## Skills

### Global skills

- Data Analysis
- Model development and testing
- Models fine-tuning
- NLP and transformers applications
- API development
- MLOps basics
- OLTP relational databases
- OLAP reporting
- Data Visualization
- Scrapping and automation

### Platforms
- Azure (Azure Data Studio)
- Kaggle
- Databricks
- Google Colab
- Streamlit

### Languages and frameworks
- Python
- SQL
- Pandas and Numpy
- PySpark and SparkSQL
- Tensorflow and Pytorch
- Huggingface libraries
- FastAPI and Pydantic
- Github
- Github Actions
- DVC
- Selenium and Scrapy

I also have some knowledge on backend technologies for web and API development:

- Typescript
- Node and NextJS
- MongoDB
- Sequelize
- GraphQL

# Projects

## Notebooks

### Credit card fraud detection
[![image](/assets/img/banner_credit_card.png)](https://www.kaggle.com/code/cristianpercivati/detecci-n-de-fraudes-en-tarjetas-de-cr-dito)

*Tipo de problema*: Clasificación binaria 

El objetivo es encontrar un modelo que, dada la información brindada, sea capaz de predecir si una transacción futura será fraudulenta o no. En este análisis lo que hago es el tratamiento de un típico dataset con un problema de **desbalanceo de datos**, donde el **recall** es la métrica más relevante para evaluar la utilidad del modelo que se utilice. 

Pruebo técnicas de **reducción de dimensionalidad**, y de balanceo de datos como resampling o **SMOTE**. Se utiliza una **regresión logística** como modelo viable.

### IBM attrition analysis
[![image](/assets/img/banner_ibm_attrition.png)](https://www.kaggle.com/code/cristianpercivati/rotaci-n-de-empleados-de-ibm)

*Tipo de problema*: Clasificación binaria 

La idea del trabajo es explicar las causas del attrition y encontrar algún modelo predictivo que permita interceptar futuros casos de attrition (desgaste que provocan el posible renunciamiento de un empleado) para evitar la **rotación excesiva**. En este análisis también se nos presenta un problema de desbalanceo de datos, pero en este caso es menos permisibile los falsos negativos respecto al ejemplo de fraude de tarjetas, por lo cual se necesitaba mantener un balance de estas métricas. 

Se hizo un EDA de las características, y se seleccionó en función de las correlaciones teniendo en cuenta tests de hipótesis como **chi cuadrado** y dándole importancia a la segmentación de los datos, que nos permitió ver mejores correlaciones y elegir las características útiles en función de la variable objetivo.

Para mejorar los resultados, se realizó un **SMOTE** que permitió mejorar los resultados del modelo, en este caso se utilizó **XGBoost** dado que las relaciones son poco lineales.

### Spaceship Titanic Competition
[![image](/assets/img/banner_spaceship_titanic.png)](https://www.kaggle.com/code/cristianpercivati/spaceship-titanic-around-80-precision)

*Tipo de problema*: Clasificación binaria 

### Buenos Aires Properati Price Prediction
[![image](/assets/img/banner_properati.png)](https://www.kaggle.com/code/cristianpercivati/buenos-aires-properties-price-prediction)

*Tipo de problema*: Regresión 

### Twitter dataset NLP analysis
[![image](/assets/img/banner_twitter_analysis.png)](https://www.kaggle.com/code/cristianpercivati/transformers-on-twitter-dataset)

*Tipo de problema*: Análisis de sentimientos / Clasificación multiclase

Se utiliza un **transformer ROBERTA** para la clasificación de twits de un dataset de Twitter de la India.

### Uber NY NLP analysis
[![image](/assets/img/banner_uber.png)]{ width="800" height="120" style="display: block; margin: 0 auto" }
*Tipo de problema*: Análisis de sentimientos / Nube de palabras

### YOLO object detection
![image](/assets/img/banner_yolo.png)
*Tipo de problema*: Detección de imágenes

## Apps

### EstudIA
### Bot-to-Bot app
### Phonetics corrector
### DQN applied for transport problems

## Models

### Whisper fine-tuned for CMU
![Whisper Model](./assets/img/model_1.png)

Realicé un **ajuste fino** a la versión base de Whisper de OpenAI. La idea era poder utilizarlo en mi app (compartida más arriba) que permite corregir errores fonéticos en la pronunciación.

---

### Llama 3B fine-tuned for Uber dataset
![Llama Model](./assets/img/model_2.png)

El ajuste fino (realizado con **QLoRA**) se utilizó para adaptar la versión de 3B de Llama 3 para que simule ser un pasajero según un dataset de viajes propio que se le brindó. En función de los datos de los viajes, generó comentarios y calificaciones **sintéticas**.

---

### Deep Reinforcement Learning DQN for transport problems
![DQN Model](./assets/img/model_3.png)

Este modelo fue parte de un trabajo práctico integrador para mi tecnicatura. La idea fue usar una DQN utilizando **Deep Reinforcement Learning** que me permitiera desarrollar un conjunto de **datos sintéticos** predictivos que demuestren el efecto que puede tener la **toma de decisiones simulada** sobre la tarifa del servicio y los beneficios de la dinamización de la misma.

## Dashboards

## Others

### Personal blog
### Youtube's channel
