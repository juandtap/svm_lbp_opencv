## Visión por computador 
## Practica 3.1.2 Patrones Binarios Locales (LBP) para tareas de identificación y clasificación de imágenes 
### Autores:  
Astudillo Paul y Tapia Diego 

### Fase de entrenamiento
En el archivo ```training.cpp``` se cargan las imágenes de dos categorías de texturas: Rocas y Madera. Se realizan las conversiones necesarias, se entrena el modelo SVM y finalmente se exporta para ser usado en la siguiente fase.

### Fase de predicción/clasificación
En el archivo ```main.cpp``` se carga el modelo SVM, se carga una nueva imagen y se clasifica. Si el resultado es 0, la imagen pertenece a la categoría Roca; si el resultado es 1, es Madera.



