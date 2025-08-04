# Hello there!👋 

I’m Cristian Percivati, a passionate Computer Science and Artificial Intelligence student pursuing my first role as an **AI Engineer** or **Data Scientist**.

### Academic Background
- Associate Degree in Data Science & AI (completed).
- Currently studying Computer Science Engineering (BSc).

### Core Technical Skills
- Languages: Python, SQL, JS
- Cloud/ML Tools: Azure
- Reporting Tools: Power BI
- Frameworks: (PyTorch, TensorFlow, LangChain, HuggingFace, etc)

### What I Bring  
A blend of theoretical knowledge and practical project experience in AI/ML, with a focus on building scalable solutions.

### Get In Touch  
📧 **Email**: [cpercivatif@gmail.com](mailto:cpercivatif@gmail.com)  
📱 **Phone**: [+54 11 4063-3321](tel:+541140633321)  

### Featured Projects  
Next you can explore my technical exercises and implementations:  

<details markdown="block">
  <summary>Apps</summary>
  
--- 
  
## Apps

### EstudIA

The app helps students understand and progress in academic subjects. Built on *Langchain*, it handles chained prompts and **document retrieval**, with content stored in a vector database (*ChromaDB*). An LLM (*GPT's API*) then uses this retrieved data as context to deliver more accurate answers and questions.

The app operates in two modes:

- Conversational Mode: A free-form discussion about the subject.

- Evaluation Mode: A QA exam-style format where an **agent** evluates the student’s answers, provides feedback, and determines correctness.

Based on the user’s performance, the system tracks correct and incorrect responses, dynamically adjusting focus to reinforce weaker areas. Additionally, the app can generate summaries of the subject matter.

<div align="center">  
  <a href="https://github.com/CristianPercivati/estudia-app" target="_blank">
    <img src="https://img.shields.io/badge/View_Code-GitHub-181717?style=for-the-badge&logo=github" alt="GitHub">
  </a>  
  <a href="https://www.youtube.com/watch?v=1NdnyeP0FbM" target="_blank">
    <img src="https://img.shields.io/badge/Watch_Demo-YouTube-FF0000?style=for-the-badge&logo=youtube" alt="YouTube">
  </a>  
  <a href="https://docker.com/" target="_blank">
    <img src="https://img.shields.io/badge/Download-🐳_Docker-2496ED?style=for-the-badge&logo=docker" alt="Docker">
  </a>  
</div>


### Bot-to-Bot app

This app is a fun and simple way to explore how two advanced AI chatbots (GPT and Deepseek) think and respond each other, allowing you to see the difference in **biases**.

You start by entering a prompt, and then the two bots take turns chatting with each other. As their conversation goes on, you’ll notice they often take different approaches: GPT usually focuses more on privacy and individual rights, while Deepseek tends to highlight the importance of social good and collective values.

To guide the conversation a bit, the starting prompt includes an example to set the tone. Watching the two bots interact gives you an interesting look at how their training shapes their opinions, and how different kinds of AI can “see” the same topic in very different ways.

- [Ver video 👀](https://youtu.be/z35HhIoJ5gY)
- [Descargar desde Docker 🐳](https://google.com)

### Phonetics corrector

This was a personal project aimed at exploring the capabilities of OpenAI's Whisper, a speech transcription model released at the time. The app’s goal was to identify pronunciation differences in spoken words (English only).

#### Model Retraining
Fine-tuned Whisper using a public audio transcription dataset that included open mic recordings and noisy audios.
Converted dataset labels to CMU (an open pronunciation dictionary) for compatibility.

#### Functionality
The retrained model transcribes speech into CMU phonetic representations.
The original Whisper model compares this output to the actual pronunciation.
The system then identifies correctly and incorrectly pronounced phonemes.

- [Ver video 👀](https://youtu.be/lnRcwrBtzmY)
- [Descargar desde Docker 🐳](https://google.com)


Enlace a video: :movie_camera:

### DQN applied for transport problems

This project demonstrates how **simulated environments** can be created from limited original data using **data augmentation**, **causal inference**, and expansion techniques. The goal was to model how **dynamic pricing adjustments** could significantly improve growth projections for the following year.

A Deep Q-Network (DQN) was trained via **reinforcement learning**, enabling an agent (the company itself) to learn optimal pricing strategies based on environmental feedback. The resulting model provided a (synthetic) data driven estimate of how flexible pricing decisions could enhance forecasted growth curves.

#### Technical Approach:
- Data Simulation: Augmented sparse datasets to build a robust synthetic environment.
- Causal Analysis: Identified key decision drivers through inference techniques.
- Agent Training: The DQN agent learned adaptive pricing policies by interacting with the simulated market.

- [Ver video 👀](https://youtu.be/aYyind5eH5w)
- [Descargar desde Docker 🐳](https://google.com)
</details>
<details markdown="block">
  <summary>Notebooks</summary>
  
## Notebooks

### Credit card fraud detection
[![image](../assets/img/banner_credit_card.png)](https://www.kaggle.com/code/cristianpercivati/detecci-n-de-fraudes-en-tarjetas-de-cr-dito)

*Tipo de problema*: Clasificación binaria 

El objetivo es encontrar un modelo que, dada la información brindada, sea capaz de predecir si una transacción futura será fraudulenta o no. En este análisis lo que hago es el tratamiento de un típico dataset con un problema de **desbalanceo de datos**, donde el **recall** es la métrica más relevante para evaluar la utilidad del modelo que se utilice. 

Pruebo técnicas de **reducción de dimensionalidad**, y de balanceo de datos como resampling o **SMOTE**. Se utiliza una **regresión logística** como modelo viable.

### IBM attrition analysis
[![image](../assets/img/banner_ibm_attrition.png)](https://www.kaggle.com/code/cristianpercivati/rotaci-n-de-empleados-de-ibm)

*Tipo de problema*: Clasificación binaria 

La idea del trabajo es explicar las causas del attrition y encontrar algún modelo predictivo que permita interceptar futuros casos de attrition (desgaste que provocan el posible renunciamiento de un empleado) para evitar la **rotación excesiva**. En este análisis también se nos presenta un problema de desbalanceo de datos, pero en este caso es menos permisibile los falsos negativos respecto al ejemplo de fraude de tarjetas, por lo cual se necesitaba mantener un balance de estas métricas. 

Se hizo un EDA de las características, y se seleccionó en función de las correlaciones teniendo en cuenta tests de hipótesis como **chi cuadrado** y dándole importancia a la segmentación de los datos, que nos permitió ver mejores correlaciones y elegir las características útiles en función de la variable objetivo.

Para mejorar los resultados, se realizó un **SMOTE** que permitió mejorar los resultados del modelo, en este caso se utilizó **XGBoost** dado que las relaciones son poco lineales.

### Spaceship Titanic Competition
[![image](../assets/img/banner_spaceship_titanic.png)](https://www.kaggle.com/code/cristianpercivati/spaceship-titanic-around-80-precision)

Este fue un trabajo que hice durante la materia de Cs. de Datos en el instituto. Tras un EDA profundo, se logró una buena explicabilidad de la correlación de las variables e **ingeniería de características**. Como modelo, el que mejor resultado dio fue una **red neuronal profunda**, pero para poder explicar los resultados recurrí a un análisis **SHAP**.

*Tipo de problema*: Clasificación binaria 

### Buenos Aires Properati Price Prediction
[![image](../assets/img/banner_properati.png)](https://www.kaggle.com/code/cristianpercivati/buenos-aires-properties-price-prediction)

Este trabajo es uno de mis primeros notebooks, basado en el clásico problema de regresión de los precios de propiedades, pero esta vez utilizando un dataset de Properati para el análisis descriptivo y predictivo
de precios de inmuebles de la Ciudad de Buenos Aires.

*Tipo de problema*: Regresión 

### Twitter dataset NLP analysis
[![image](../assets/img/banner_twitter_analysis.png)](https://www.kaggle.com/code/cristianpercivati/transformers-on-twitter-dataset)

*Tipo de problema*: Análisis de sentimientos / Clasificación multiclase

Se utiliza el **transformer BERT** para la clasificación de twits de un dataset de Twitter de la India.

### Uber NY NLP analysis
[![image](../assets/img/banner_uber.png)](https://www.kaggle.com/code/cristianpercivati/uber-espa-ol-an-lisis-de-sentimientos)

Este fue el análisis exploratorio previo a desarrollar el modelo que genere comentarios simulando ser un pasajero de un viaje. Se hizo un trabajo de preprocesamiento de NLP básico (**lemmatización** y eliminación de **stop words**) y luego se vectorizó el vocabulario con **CountVectorizer**. Esto me permitió realizar una nube de palabras de las palabras más positivas y más negativas realizadas por los pasajeros.

*Tipo de problema*: Análisis de sentimientos / Nube de palabras

### YOLO object detection
[![image](../assets/img/banner_yolo.png)](https://www.kaggle.com/code/cristianpercivati/yolo-demo)

En este ejercicio, lo que se hizo fue utilizar la librería de **YOLOv8** para la detección de objetos en una imagen.

*Tipo de problema*: Detección de objetos en computer vision

### SAM image segmentation
[![image](../assets/img/banner_sam.png)](https://www.kaggle.com/code/cristianpercivati/sam-demo)

En este ejercicio, lo que se hizo fue utilizar la librería de **SAM** para la segmentación de imágenes.

*Tipo de problema*: Segmentación de imágense en computer vision

</details>
<details markdown="block">
<summary>Models</summary>
  
## Modelos y ajuste fino
<table>
  <tr>
    <td style="vertical-align: top; width: 100px;">
      <img src="../assets/img/model_1.png" alt="Whisper Model" width="100"/>
    </td>
    <td>
      <h3>Whisper fine-tuned for CMU</h3>
      Realicé un <strong>ajuste fino</strong> a la versión base de Whisper de OpenAI. La idea era poder utilizarlo en mi app (compartida más arriba) que permite corregir errores fonéticos en la pronunciación.
    </td>
  </tr>
  <tr>
    <td style="vertical-align: top;">
      <img src="../assets/img/model_2.png" alt="Llama Model" width="100"/>
    </td>
    <td>
      <h3>Llama 3B fine-tuned for Uber dataset</h3>
      El ajuste fino (realizado con <strong>QLoRA</strong>) se utilizó para adaptar la versión de 3B de Llama 3 para que simule ser un pasajero según un dataset de viajes propio que se le brindó. En función de los datos de los viajes, generó comentarios y calificaciones <strong>sintéticas</strong>.
    </td>
  </tr>
  <tr>
    <td style="vertical-align: top;">
      <img src="../assets/img/model_3.png" alt="DQN Model" width="100"/>
    </td>
    <td>
      <h3>Deep Reinforcement Learning DQN for transport problems</h3>
      Este modelo fue parte de un trabajo práctico integrador para mi tecnicatura. La idea fue usar una DQN utilizando <strong>Deep Reinforcement Learning</strong> que me permitiera desarrollar un conjunto de <strong>datos sintéticos</strong> predictivos que demuestren el efecto que puede tener la <strong>toma de decisiones simulada</strong> sobre la tarifa del servicio y los beneficios de la dinamización de la misma.
    </td>
  </tr>
</table>

</details>
<details markdown="block">  
<summary>Dashboards</summary>
  
## Dashboards

### Reporte sobre el mercado de datos

Este dashboard fue un proyecto que realicé durante unas prácticas en el Instituto. En este reporte lo que hice fue analizar la oferta laboral de ai-jobs.net. Luego, usando técnicas de **scrapping**, obtuve las ofertas equivalentes en Linkedin Argentina.

![PBI - Data Jobs](../assets/img/pbi-1.png)

### Ejemplo de Data Warehousing

Este fue un ejercicio realizado para una capacitación en Quales. La idea era aplicar **ETL** con **SQL** para transformar archivos csv sueltos en un **Data Warehouse** listo para ser consumido en PBI.

![PBI - Data Warehousing](../assets/img/pbi-2.png)

### Ejercicio de PBI

Este es un simple ejercicio que realicé hace algunos años en un curso de Udemy.

![PBI - RRHH Exercise](../assets/img/pbi-3.png)

</details>
