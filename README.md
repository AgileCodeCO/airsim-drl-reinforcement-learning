# Controlar un dron en el entorno de simulación Microsoft AirSim usando aprendizaje por refuerzo

<img src="https://github.com/AgileCodeCO/airsim-drl-reinforcement-learning/blob/main/img/drone-flying.gif?raw=true" width="275">

**Autor**: Jorge Armando González Pino

**Director**: Cesar Augusto Guzmán Álvarez


# Resumen

En los años recientes ha surgido un interés general por las carreras de drones con ligas de pilotos que compiten alrededor de mundo con dispositivos reales o en ambientes simulados.  Luego de ver varias de estas competencias en la televisión y en internet, nació la idea de aplicar los conocimientos adquiridos en el Máster para crear un modelo de inteligencia artificial que le permita a un agente volar uno de estos drones en un ambiente de carreras simulado.

Se realizó un análisis de las diferentes plataformas de simulación 3D que permiten interactuar con vehículos de conducción autónoma, en este caso con drones: Gazebo, Unity ML, Microsoft AirSim (Shah, Dey, Lovett, & Kapoor, 2018) entre otras.  Siendo esta última la plataforma elegida para este proyecto dado que existe un framework llamado AirSim Drone Racing Lab (Madaan, et al., 2020) basado en Microsoft AirSim, el cual facilitó el tiempo de prototipado al generar entornos de simulación con pistas de carreras en diferentes ambientes, compuertas, sensores, cámaras y otros elementos optimizados para investigaciones de Machine Learning.

Utilizando la API basada en Python provista por AirSim Drone Racing Lab, la cual permite acceder a la información actual del entorno, cámaras, sensores, estado de la carrera, además de permitir la ejecución de acciones sobre el dron. Se entrenó un agente inteligente usando el algoritmo de aprendizaje por refuerzo llamado optimización de política próxima o Proximal Policy Optimization (PPO) con un espacio de acciones continuo.

Para llevar a cabo dicho entrenamiento, se estableció cómo único estado del entorno la nube de puntos 3D generada por un sensor LiDAR configurado en el dron con el objetivo de hacer el modelo más generalizable. Se implementó en el modelo de aprendizaje por refuerzo una arquitectura de red neuronal diseñada específicamente para extraer características tridimensionales a partir de una nube de puntos 3D llamada PointNet.

Uno de los puntos importantes del modelo de aprendizaje por refuerzo implementado fue el cálculo de la recompensa obtenida por el agente o dron luego de cada acción. Se diseñaron diferentes tipos de recompensa teniendo en cuenta la posición del dron con respecto a las puertas de la pista y el orden de estas que le permitieron al agente aprender una estrategia óptima para cruzar las puertas de la pista siguiendo una trayectoria ideal.


# Arquitectura

 


<img src="https://github.com/AgileCodeCO/airsim-drl-reinforcement-learning/blob/main/img/model-diagram.png?raw=true">

 
 Se implementa un modelo de aprendizaje por refuerzo basado en PPO con un agente que obtiene del entorno de simulación **AirSim Drone Racing Lab** una observación a partir de la cual se define un estado usando la nube de puntos 3D generada por el sensor **LiDAR** configurado en el dron. Dicho estado sirve como entrada para una red neuronal de aprendizaje profundo basada en la arquitectura **PointNet** que extrae sus características tridimensionales para entrenar una policy que ejecute acciones en el entorno para volar el dron a traves de las puertas de la pista y recibir una recompensa.


# Pasos para ejecutar

 

## Requerimientos

* Computador o portatil con sistema operativo Windows 10 o superior
* NVIDIA GPU con controladores CUDA 11.3 o superior instalados
* Python 3.9.9
* Librerías de Python
    
    * Pytorch con soporte para CUDA 11.3 o superior
    * API del entorno de simulación **airsimdroneracinglab**
    * Numpy
    * Pandas
    * OpenCV
    * MatplotLib


## Instalación


### Código fuente

1. Clonar este repositorio https://github.com/AgileCodeCO/airsim-drl-reinforcement-learning.git en el equipo local

### Entorno de simulación

1. Descargar el ejecutable de AirSim Drone Racing Lab para windows: https://github.com/microsoft/AirSim-Drone-Racing-Lab/releases/download/v1.0-windows/ADRL.zip

2. Descomprimir el archivo ADRL.zip
3. Descargar el archivo de configuración del entorno de simulación para este proyecto en la carpeta C:\Users\[Usuario]\Documents\AirSim (crear carpeta si no existe): https://raw.githubusercontent.com/AgileCodeCO/airsim-drl-reinforcement-learning/main/environment/settings.json

### Librerias de python

1. pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
2. pip install pandas
3. pip install opencv-python
4. pip install airsimdroneracinglab

## Ejecución

 

### Para entrenamiento:

1. Iniciar el entorno de simulación por linea de comandos desde la carpeta de instalación: 

    **.\ADRL.exe -windowed -NoVSync**

2. Ejecutar el script de python por linea de comandos desde la carpeta donde se clonó el código fuente:

    **python .\main.py --mode=train**

### Para pruebas:

1. Iniciar el entorno de simulación por linea de comandos desde la carpeta de instalación: 

    **.\ADRL.exe -windowed -NoVSync**

2. Modificar la linea de código 225 del archivo **main.py** para definir con cual modelo pre-entrenado de la carpeta **models** se quiere probar. (5 o 7 son los mejores modelos)

3. Ejecutar el script de python por linea de comandos desde la carpeta donde se clonó el código fuente:

    **python .\main.py --mode=test**

# Referencias

 
Madaan, R., Gyde, N., Vemprala, S., Brown, M., Nagami, K., Taubner, T., . . . Kapoor, A. (2020). Airsim drone racing lab. In NeurIPS 2019 Competition and Demonstration Track, 177-191.

Li, Y., Ma, L., Zhong, Z., Liu, F., Chapman, M., Cao, D., & Li, J. (2020). Deep learning for lidar point clouds in autonomous driving: A review. IEEE Transactions on Neural Networks and Learning Systems, 3412-3432.

Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017). Pointnet: Deep learning on point sets for 3d classification and segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, 652-660.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv.

Shah, S., Dey, D., Lovett, C., & Kapoor, A. (2018). Airsim: High-fidelity visual and physical simulation for autonomous vehicles. In Field and service robotics, 621-635.

Barhate, N. (2021). Minimal PyTorch Implementation of Proximal Policy Optimization. Retrieved from GitHub: https://github.com/nikhilbarhate99/PPO-PyTorch