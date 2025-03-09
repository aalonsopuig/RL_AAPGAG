# Práctica 2: Aprendizaje en entornos complejos
## Información
- **Alumnos:** García Meroño, Andrés; Guillén Marquina, Pablo; Alonso Puig, Alejandro
- **Asignatura:** Extensiones de Machine Learning
- **Curso:** 2024/2025
- **Grupo:** AAPGAG
## Descripción
Estudiar problemas donde no se conoce el modelo que rige el entorno ni se sabe qué recompensaas se pueden encontrar.
Se estudian técnicas que recurren a la experiencia que va adquiriendo el agente conforme interactúa con el entorno.
Se desarrollarán estudios comparativos de diferentes técnicas, como Monte Carlo (on-policy y off-policy) y Diferencias Temporales (SARSA, Q-Lerning) para métodos tabulares
y otras técnicas de control con aproximaciones como SARSA semi-gradiente y Deep Q-Learning.

## Estructura
[Explicación de la organización del repositorio]

## Instalación y Uso
El notebook **main.ipynb** es el punto de inicio del proyecto. Desde él, se proporciona acceso a los notebooks de los estudios.  

Para poner en marcha la ejecución del proyecto, simplemente sigue estos pasos:  

1. **Abrir main.ipynb** utilizando el siguiente enlace para Google Colab: [Open in Colab](https://colab.research.google.com/github/aalonsopuig/RL_AAPGAG/blob/main/main.ipynb) 

2. **Acceder a los notebooks de los experimentos**:  
   Al finalizar la ejecución, en la parte inferior del notebook principal, aparecerán enlaces directos a los notebooks individuales para cada estudio:  
     - **Primer Agente**  
     - **Estudio Monte Carlo**  
     - **Estudio Diferencias Temporales**
       
   Basta con hacer clic en cualquier enlace para abrir y ejecutar el estudio correspondiente.

3. **Ejecutar todas las celdas** en orden automático:  
   En la barra de menú de Colab, haz clic en **Entorno de ejecución > Ejecutar todas**.  
    

## Tecnologías Utilizadas  
- **Lenguaje:** Python 3.x  
- **Bibliotecas:** NumPy, Matplotlib, Pandas, SciPy, gymnasium, tqdm, seaborn  
- **Entorno de ejecución:** Jupyter Notebook, Google Colab 
