FERIA MEXICANA DE CIENCIAS E INGENIERÍAS
Reconocimiento del Lenguaje de Señas Mexicano mediante Inteligencia Artificial
NÚMERO DE FIPI : COA-SI-XXXX

Tipo de proyecto: Tecnológico
Nivel educativo: Educación media superior (preparatoria)
Categoría: Sistemas informáticos
Institución: CECyTEC Sol de Oriente

Nombre de los participantes:
[Nombre del participante 1]
[Nombre del participante 2]
[Nombre del participante 3]

Nombre del asesor:
[Nombre del asesor]

RESUMEN

Este proyecto construye un sistema que reconoce Lengua de Señas Mexicana con cámara y software libre, y la convierte en texto. La idea es directa: alguien hace una seña frente a la cámara, y el sistema escribe lo que dice.

Usamos MediaPipe para sacar 21 puntos de la mano en cada cuadro del video. Después esos puntos pasan por una red neuronal que aprende a distinguir letras o palabras. Hay dos modelos separados. El de abecedario funciona con 21 letras estáticas y cinco dinámicas, y se acerca al 99% de acierto. El de vocabulario es más ambicioso: 210 palabras en 12 categorías, con apenas 17 videos por palabra en promedio. Aquí el accuracy baja a 42%, y no por culpa del diseño sino por falta de datos.

La parte interesante es que todo el código, los datos y los modelos entrenados están disponibles para quien los quiera usar, modificar o adaptar a otra lengua de señas. La barrera de entrada es una computadora normal y una cámara web. Nada más.

A futuro, el sistema podría integrarse a videollamadas de Google Meet, Teams o WhatsApp, donde la gente ya pasa varias horas al día. Pero antes de eso hay que resolver el cuello de botella real: conseguir más datos etiquetados de señas completas.


1. INTRODUCCIÓN

Comunicarse es una necesidad humana básica. Para una persona sorda en México, hacerlo con alguien que no conoce la Lengua de Señas Mexicana casi nunca es sencillo. La salida más común sigue siendo escribir en un papel, esperar a un intérprete, o resignarse a no comunicarse.

El LSM no es español traducido a señas. Es una lengua completa, con gramática propia. El problema es que casi nadie fuera de la comunidad sorda la conoce, así que cuando alguien sordo llega a una consulta médica, a una tienda o a un salón de clases, lo más probable es que no haya nadie que lo entienda.

En los últimos años esto empezó a cambiar. MediaPipe, la librería de visión por computadora de Google, puede rastrear la posición de los dedos y la palma con una cámara normal, sin sensores ni guantes especiales. No fue pensada para lengua de señas, pero da justo lo que este problema necesita: las coordenadas de cada articulación de la mano, cuadro a cuadro. TensorFlow se encarga de la otra mitad: aprender a distinguir una seña de otra a partir de esos puntos.

La pregunta que guió el proyecto fue simple: ¿se puede entrenar un modelo que reconozca el abecedario del LSM en tiempo real, usando solo una computadora estándar? Construimos un sistema que lo hace. Después intentamos llevarlo un paso más allá con un segundo modelo entrenado con 210 palabras del LSM, no solo letras. Ese segundo modelo todavía no es confiable, y se explica por qué más adelante.

La idea de uso es directa: alguien hace una seña frente a la cámara de su computadora o teléfono, y el sistema la convierte en texto que la otra persona puede leer. Esto no sustituye a un intérprete humano, pero puede ayudar en una conversación breve donde no hay uno disponible. Farmacia, ventanilla, videollamada de trabajo, lo que sea.

Nada de esto depende de comprar hardware especial. El 96% de los usuarios de internet en México ya accede desde un celular, y ese teléfono ya trae cámara. Por eso diseñamos el modelo pensando en integrarse a las herramientas que la gente ya usa, no en competir con ellas ofreciendo una app más que nadie tendría que instalar.

Las redes sociales y las apps de mensajería ya cambiaron cómo se comunica la gente, y eso incluye a personas sordas. Una videollamada de WhatsApp, por ejemplo, ya es un canal que cualquiera puede usar sin depender de escribir. Los adolescentes pasan buena parte del día en el celular, lo que convierte a esas plataformas en el lugar más natural para llevar el reconocimiento de señas, en vez de pedirle a la gente que aprenda una herramienta nueva. Del lado empresarial, negocios que atienden clientes sordos por canales digitales se benefician de un sistema que funcione dentro de las apps que ya usan a diario.


2. ANTECEDENTES

Antes de escribir código revisamos qué se había hecho ya en reconocimiento de señas con IA. Casi toda esa investigación está enfocada en ASL (americano) y BSL (británico). Para LSM la información disponible era bastante más limitada.

Dentro de México sí hay trabajo previo que vale la pena mencionar. En la UABC y el CICESE, Irvin Hussein López Nava desarrolló Daktilos, una aplicación que reconoce las 30 letras del abecedario LSM usando visión por computadora. El IPN construyó un guante con sensores mioeléctricos que verbaliza señas y ganó el James Dyson Award México con eso. Ninguno de los dos cubre vocabulario de palabras completas en tiempo real con solo una computadora estándar.

En 2024 el CICESE lanzó Sordbo, pensado para tomar pedidos de cafetería en LSM. Por otro lado, EnSeñas AI, una startup mexicana, está construyendo un traductor bidireccional LSM-español. Su obstáculo principal, documentado por ellos mismos, es la falta de datos etiquetados de calidad. Es el mismo problema que enfrentamos nosotros.

A nivel internacional el panorama es bueno, pero desigual entre lenguas de señas. Las mejores arquitecturas para ASL llegan a 99.97%, para LIBRAS a 93%, para ISL a 89.5%. Para LSM, un estudio de 2025 basado en máquinas de soporte vectorial alcanzó 92%. Nosotros llegamos a 99.06% con el modelo de abecedario.

La diferencia entre estos resultados no está en qué tan sofisticada es la arquitectura. Conv1D+BiLSTM, SVM o una red convolucional simple pueden llegar a resultados parecido en condiciones comparables. La diferencia real está en cuántos datos etiquetados existen para cada lengua. El inglés y el portugués tienen datasets públicos enormes. Para LSM prácticamente no había ninguno cuando empezamos.

Las herramientas de base también son conocidas. MediaPipe se ha vuelto el estándar para detectar puntos clave de manos en tiempo real. TensorFlow y PyTorch dominan el entrenamiento de redes profundas. De esta revisión sacamos una conclusión clara: no existía un sistema completo para LSM que pudiéramos adaptar. Terminamos combinando MediaPipe para detectar la mano con una red entrenada desde cero para clasificar la seña. Cada pieza tuvo que ajustarse, desde el número de keypoints hasta cómo normalizar las coordenadas, antes de que funcionara con nuestros propios datos.

FUENTES EXTERNAS VALIDADAS

Durante la investigación encontramos y auditamos varias fuentes de datos públicas. La regla fue simple: nada se usa sin verificar licencia, consentimiento y separación por firmante. Las fuentes validadas son:

Mendeley (Espejel et al., 10.17632/6rj76z6y3n.1) bajo licencia CC BY 4.0. Contiene 249 clases de señas y carpetas por firmante. Después de auditar la cobertura completa, las clases con todos los firmantes son 210. Este es el benchmark principal del proyecto, dividido así: firmantes S01-S07 para entrenamiento (1,470 clips), S08 para validación (210 clips), S09 para prueba (210 clips). S09 se mantiene cerrado: no se usa para seleccionar candidatos ni comparar modelos.

Zenodo 14689869 contiene las cinco letras dinámicas J, K, Ñ, Q, X, Z, con 12 participantes y alrededor de 600 videos bajo licencia CC BY 4.0. Se incorpora como fuente separada del benchmark de palabras, sin mezclar datos.

ICKMejia (RGB-D, 30 señas, 3,000 secuencias de 20 frames, keypoints 3D precalculados). La validación externa con una TCN simple alcanzó macro-F1 de 0.919 en el conjunto de prueba publicado, lo que confirma que la ruta de extracción de keypoints funciona correctamente. Este corpus no se mezcla con el benchmark de palabras.

Mendeley CC BY-NC (dactilología y números, 3 participantes) y Mozilla Querétaro (corpus paralelo) están en cuarentena: se auditaron pero no se descargaron ni usaron por restricciones de licencia.

La sección de investigación del repositorio documenta el protocolo de auditoría para cada fuente, con puertas de DOI, hash, licencia, consentimiento, etiquetas y separación por firmante.


3. DEFINICIÓN DEL PROBLEMA, PREGUNTA Y META DEL PROYECTO

La pregunta que guió todo fue directa: ¿puede un sistema de IA reconocer señas del LSM en tiempo real, usando solo la cámara de una computadora estándar, sin depender de escritura ni de un intérprete humano presente?

De ahí se desprendió una meta concreta: construir un prototipo capaz de detectar las 26 letras entrenables del abecedario LSM con una precisión mínima del 80%, usando la cámara de una computadora estándar y herramientas de código abierto.

El contexto que le da peso a esta pregunta es la situación de accesibilidad en México. Según el INEGI (2020), solo el 21% de las personas sordas en edad escolar asiste a la escuela, y el 47% no llega más allá de cuarto grado de primaria. En lo laboral, apenas el 29.9% de la población sorda es económicamente activa.

Estos números no hablan de falta de capacidad. Hablan de falta de acceso. Y el acceso no se resuelve con buenas intenciones, se resuelve con herramientas que la gente pueda usar hoy.

No buscábamos la solución definitiva. Sería ingenuo pensar que un prototipo escolar puede resolver algo tan grande. Buscábamos demostrar que, con las herramientas que ya existen en cualquier computadora, se puede construir algo funcional para el LSM.

HIPÓTESIS

Si se extraen keypoints de la mano con MediaPipe en lugar de usar la imagen completa, y se entrena una red sobre esos keypoints con suficientes muestras por clase, entonces el sistema puede alcanzar un accuracy de al menos 80% en el reconocimiento de señas en tiempo real, sin necesidad de hardware especializado.

Variables que seguimos durante el desarrollo: la cantidad de muestras por clase como variable principal, la arquitectura del modelo como segunda variable, y como controladas mantuvimos la resolución de cámara, la versión de MediaPipe, la proporción de datos y los hiperparámetros de entrenamiento.

JUSTIFICACIÓN

El censo de 2020 del INEGI registró 2,576,213 personas en México con mucha dificultad para oír, y 1,417,126 personas con dificultad para hablar o comunicarse. La distribución geográfica se concentra en los estados más poblados y en zonas urbanas del centro del país.

La distribución por edad importa para pensar el diseño de la solución. Más de la mitad tiene más de 60 años, en su mayoría sordera adquirida con la edad. Un 34% adicional tiene entre 30 y 59 años. Los niños representan apenas el 2%. No es un problema de una sola generación.

Casi la mitad de los casos de discapacidad auditiva en el país se debe a la edad avanzada. Otro 28.9% proviene de alguna enfermedad. Solo el 9.3% es de nacimiento.

La barrera más concreta que enfrenta esta población no es la falta de intérpretes. Es que el 70% de las palabras del español no tienen una seña directa en LSM y deben deletrearse letra por letra. Eso vuelve lento cualquier trámite cotidiano. La ENADIS 2022 documentó que el 33.8% de las personas con discapacidad reportó haber sufrido discriminación, y el 80% de los trabajadores sordos no cuenta con ninguna tecnología de asistencia.

Para dimensionar el mercado, el software de accesibilidad digital a nivel global pasó de 721.1 millones de dólares en 2023 a una proyección de 1,300.3 millones para 2030, con una tasa de crecimiento anual del 9.2%.


5. OBJETIVOS

Objetivo general

Construir un sistema de reconocimiento de señas del LSM que traduzca gestos a texto en tiempo real, usando IA y herramientas de código abierto.

Objetivos específicos

1. Entrenar un modelo de reconocimiento de señas del LSM con MediaPipe, TensorFlow y OpenCV. La meta concreta: que el abecedario alcance al menos 80% de accuracy y funcione en tiempo real con cámara estándar.

2. Entrenar un segundo modelo para 210 palabras del LSM, organizado en 12 categorías semánticas, con meta de 85% de accuracy.

3. Construir una app móvil en Expo y React Native que ejecute los modelos en tiempo real, sin enviar datos a la nube, y deje trazas de auditoría de cada predicción.

4. Validar los modelos con separación estricta por firmante (Leave-One-Signer-Out). La meta: que los resultados en S08 confirmen lo observado en entrenamiento, y que S09 permanezca cerrado como prueba ciega.

5. Publicar el código, los datos y los protocolos de auditoría como código abierto, con puertas claras para verificar licencias, consentimientos y separación por firmante.

6. Documentar el proceso de construcción del sistema con detalle suficiente para que otros equipos puedan replicar el trabajo, auditar los datos o adaptarlo a otra lengua de señas.


6. METODOLOGÍA

El desarrollo se organizó en cinco etapas: recolección de datos, extracción de keypoints con MediaPipe, generación de features, augmentación y entrenamiento. La Figura 6 resume el flujo completo.

Lo que cambió respecto al primer borrador de este documento es el alcance del vocabulario. El documento original hablaba de 249 palabras, pero la auditoría del corpus Mendeley mostró que solo 210 tienen cobertura completa de los nueve firmantes. Esa diferencia obligó a reescribir la metodología. Las 210 palabras con cobertura completa son las que forman el benchmark sucesor del proyecto, con la división S01-S07 para entrenar, S08 para validar y S09 como prueba ciega.

6.0 Materiales y herramientas

Python 3, MediaPipe (tasks-vision con el modelo HandLandmarker), TensorFlow 2.21.0, OpenCV y NumPy del lado de Python. Rust para las rutinas de extracción de features que corren en móvil. Expo y React Native para la app. ONNX Runtime para ejecutar los modelos entrenados en el teléfono.

La captura usa una cámara web estándar, sin sensores ni hardware de profundidad. El entrenamiento corre en una máquina con GPU y 20 núcleos de CPU.

El modelo final se exporta a TFLite para la integración móvil, lo que reduce el archivo a 45 KB en el caso del abecedario.

6.1 Recolección de datos

Para el abecedario se tomaron 574,064 fotografías en 21 letras estáticas (de la A a la V, sin la Ñ, más la W y la Y), con un promedio de 27,358 fotos por letra; el rango va de 26,358 a 29,362. Las cinco letras dinámicas (J, K, Q, X, Z) se grabaron en video en lugar de foto fija, porque su forma cambia con el movimiento y una sola imagen no las representa bien. La Ñ se intentó capturar de la misma manera, pero no se obtuvo ninguna toma utilizable y terminó excluida del entrenamiento.

Para el vocabulario el camino fue distinto. En lugar de grabar nuestro propio dataset desde cero, auditamos el corpus Mendeley de 249 clases. La auditoría reveló que solo 210 tienen cobertura completa de los nueve firmantes. Esas 210 son las que forman el benchmark. La división es estricta: 1,470 clips para entrenamiento (S01-S07, siete firmantes con 7 clips por palabra), 210 clips para validación (S08) y 210 clips para prueba (S09).

S09 es prueba ciega. No se usa para seleccionar candidatos ni para comparar modelos. Solo se abre cuando un candidato pasa las puertas de S08 y se quiere confirmar que el resultado se sostiene.


6.2 Cómo ve el sistema una seña

Antes de hablar de MediaPipe y las redes neuronales hay que dejar claro un punto básico: una computadora no procesa una fotografía como una persona.

Para una computadora, una imagen en escala de grises es una matriz de números entre 0 (negro) y 255 (blanco). Una imagen a color son tres matrices superpuestas, una por canal. Una fotografía común de 640 por 480 píxeles equivale, solo en un canal, a una matriz de 307,200 números. Tres canales, casi un millón de valores por fotografía.

Entrenar una red directamente sobre esos casi un millón de números es posible, pero caro. Hace falta mucho más cómputo y muchos más datos para que el modelo aprenda a ignorar todo lo que no es la mano: el fondo, la ropa, la iluminación. Y se quede solo con la seña.

Aquí es donde entra MediaPipe.

6.3 MediaPipe: de la matriz de píxeles a puntos por mano

MediaPipe es una librería de Google. Su modelo HandLandmarker no lo entrenamos nosotros: Google lo hizo con millones de imágenes de manos. Su única función es encontrar una mano dentro de la imagen y devolver 21 puntos clave (keypoints) con sus coordenadas x, y, z. 21 puntos por mano.

La decisión más importante del proyecto fue esta: en lugar de darle a la red la fotografía completa (casi un millón de números), le damos solamente 21 puntos por mano. 63 números en total. MediaPipe ya hizo el trabajo pesado de encontrar la mano. Nuestra red solo tiene que aprender a distinguir una seña de otra a partir de esos 63 números, no a encontrar la mano dentro de la imagen.

Esa reducción explica por qué el modelo final pesa apenas 45 KB. El modelo de MediaPipe, ya entrenado por Google, pesa 7.8 MB en su versión float16. Él carga con la parte más pesada del problema de visión.

Las fotografías del abecedario se procesaron en modo IMAGE. Las secuencias de palabras y las letras dinámicas se procesaron en modo VIDEO, que aprovecha la continuidad entre cuadros para un rastreo más estable.

Hubo un error en una primera versión del pipeline: se usó el modo VIDEO también para las fotografías, lo que producía detecciones inconsistentes porque las fotos no tienen relación temporal entre sí. Corregir esa confusión fue uno de los primeros ajustes necesarios.

La configuración final fue: hasta dos manos detectadas, umbral de detección 0.3, umbral de rastreo 0.5. Para el abecedario se usa una sola mano (63 keypoints). Para palabras se usan ambas manos más pose corporal y rasgos faciales, sumando 226 keypoints por cuadro: 126 de manos, 52 de pose y 48 de cara.

6.4 Normalización y features

Cada muestra de entrada al modelo no es un solo cuadro, sino una secuencia de 30 cuadros consecutivos. Aproximadamente un segundo de video a 30 cuadros por segundo. Esto le permite a la red observar el movimiento, no solo una postura estática.

El valor de 30 se eligió porque es suficiente para capturar el gesto completo de las letras dinámicas más lentas, y porque mantiene el tensor de entrada en un tamaño manejable para entrenar en la máquina disponible. Se probaron secuencias más largas (60 u 80 cuadros) informalmente y no mejoraban el accuracy lo suficiente para justificar el costo adicional.

Los keypoints se normalizan por bloque anatómico para que el sistema sea invariante a la posición, escala y rotación de la mano frente a la cámara. Las manos se centran en la muñeca y se escalan por la distancia entre muñeca y nudillo medio. La pose se centra en los hombros y se escala por la distancia entre ellos. La cara se centra en la nariz y se escala por la distancia entre los ojos.

Para el abecedario esto produce 93 features por cuadro a partir de los 63 keypoints originales. 72 features invariantes a rotación, traslación y escala (coordenadas XY normalizadas, extensión de cada dedo, ángulos entre falanges, distancias entre puntas de dedos) más 21 valores de profundidad (Z), suavizados con una media móvil de tres cuadros para capturar la profundidad relativa.

6.5 Augmentación de datos

La augmentación se aplicó en tres niveles, todos respetando los límites anatómicos reales de una mano. A nivel de cuadro individual se aplicaron 10 transformaciones: traslación, rotación 3D, escalado, flip horizontal, dropout de dedos, variación de curvatura, inclinación de muñeca, temblor, distorsión de perspectiva y simulación de persona específica. A nivel de secuencia completa se aplicaron 6 transformaciones temporales: cambio de velocidad, jitter, dropout de frames, recorte temporal, ruido de velocidad y ruido articular. Entre muestras se aplicó mixup.

La estrategia fue asimétrica: las clases con pocas muestras reciben más variaciones. Una clase con 8 muestras o menos recibe 20 variaciones por muestra. Una clase con más de 5,000 muestras recibe solo 4. Esto evita que el entrenamiento quede dominado por las clases más fáciles de fotografiar.

La augmentación se aplicó únicamente al conjunto de entrenamiento. Nunca a validación ni a prueba. Para evitar fuga de datos.


6.6 La red neuronal

Una red neuronal es una función matemática construida a partir de capas de funciones simples encadenadas. Cada capa tiene parámetros ajustables llamados pesos. Al inicio del entrenamiento esos pesos se inicializan de forma aleatoria, por lo que el modelo produce predicciones incorrectas.

En cada iteración, el sistema compara su predicción con la etiqueta real mediante una función de pérdida. Un algoritmo de optimización ajusta los pesos en la dirección que reduce ese error. Tras miles de iteraciones sobre el conjunto de entrenamiento, la red converge hacia parámetros capaces de reconocer patrones consistentes.

La primera operación especializada es una convolución 1D, que detecta patrones locales de movimiento entre cuadros vecinos. Después de las capas convolucionales, una BiLSTM integra el contexto temporal completo de la secuencia, leyendo hacia adelante y hacia atrás. Al final, una capa densa con activación softmax convierte toda esa información en una probabilidad para cada clase posible.

6.7 Arquitectura del modelo

El modelo de abecedario usa Conv1D + BiLSTM. La entrada es un tensor de forma (30, 93): 30 cuadros por secuencia, 93 features por cuadro. La red combina LayerNormalization sobre la entrada, dos capas Conv1D (64 y 96 filtros, kernel 3, activación ReLU) con BatchNormalization y MaxPooling1D después de cada una, SpatialDropout1D(0.2) para regularización, una capa Bidirectional LSTM de 64 unidades, Dropout(0.3), y dos capas densas finales (64 neuronas y la capa de salida con softmax).

El modelo de vocabulario se entrenó con la misma arquitectura base. La entrada en este caso es (30, 226) por la inclusión de pose y cara. La razón de combinar Conv1D con BiLSTM es que cada componente cubre algo distinto. Conv1D detecta patrones locales de movimiento entre cuadros vecinos, por ejemplo la curvatura de un dedo en transición. La BiLSTM sigue la dinámica de la secuencia completa, lo que permite distinguir señas donde el orden temporal importa, como el gesto de abrir y cerrar la mano.

Después del primer modelo de vocabulario, la investigación del proyecto exploró varias arquitecturas candidatas con el objetivo de subir el accuracy de 42%. Los candidatos investigados, todos entrenados desde cero, incluyen:

- TCN (Temporal Convolutional Network) para capturar dependencias de largo alcance con convoluciones dilatadas.
- Bone Vector 126: representación de 126 features por cuadro que codifica huesos y vectores de movimiento entre articulaciones.
- Fusión multivista con consistencia Jensen-Shannon, que valida la robustez de las predicciones entre vistas de la misma seña.
- Cross-signer bagging, que entrena múltiples modelos en subconjuntos de firmantes y promedia sus predicciones.

El mejor candidato registrado a la fecha es la fusión multivista con consistencia Jensen-Shannon, validada en tres semillas y con confirmación leave-one-signer-out. Es un resultado de investigación offline, no constituye una afirmación de reconocimiento robusto en cámara ni reemplaza la validación con personas usuarias de LSM.

6.8 Entrenamiento

El entrenamiento usa el optimizador Adam con tasa de aprendizaje inicial de 1e-3, y FocalLoss (γ=3.0, α=0.25) como función de pérdida. Focal Loss se eligió específicamente para manejar el desbalance entre clases con muchas y pocas muestras: penaliza menos los ejemplos que el modelo ya clasifica bien y concentra el gradiente en los ejemplos difíciles.

Se aplica EarlyStopping con paciencia de 10 épocas sobre la pérdida de validación, restaurando los mejores pesos encontrados, junto con ReduceLROnPlateau (factor 0.5, paciencia 5, tasa mínima 1e-6). El batch size es de 256, con un máximo de 150 épocas. Los datos se dividen en 70% entrenamiento, 15% validación y 15% prueba antes de aplicar cualquier augmentación.

El procesamiento de imágenes para extraer keypoints se paraleliza con 15 procesos de trabajo, de los 20 núcleos disponibles en la máquina de entrenamiento. Los 5 restantes se reservan para el sistema operativo. TensorFlow 2.21.0 corre con aceleración por GPU y memory growth activado para evitar que reserve toda la memoria del acelerador de una sola vez.

6.9 Evaluación y exportación

Se miden accuracy global, precisión, recall y F1-score por clase, además de una matriz de confusión para identificar qué señas se confunden entre sí con mayor frecuencia.

El modelo del abecedario se convirtió de formato Keras (.h5) a TFLite mediante cuantización, un proceso que reduce la precisión numérica de los pesos para disminuir el tamaño del archivo. El modelo pasó de 3.8 MB a 45 KB, un factor de aproximadamente 84 veces, con una pérdida de accuracy mínima.

6.10 Alternativas consideradas y descartadas

Antes de fijar Conv1D + BiLSTM se evaluaron otras dos rutas. La primera fue una red convolucional 2D entrenada directamente sobre las imágenes completas, sin pasar por MediaPipe. Se descartó porque requería órdenes de magnitud más datos y cómputo, y el modelo resultante habría pesado varios megabytes en lugar de kilobytes.

La segunda fue un clasificador SVM sobre los keypoints, similar al reportado en un estudio de 2025. Se descartó porque, aunque más simple de entrenar, no captura bien la dinámica temporal necesaria para las letras dinámicas y las palabras, donde el orden de los movimientos importa tanto como su forma.

Para el vocabulario, las alternativas exploradas durante la investigación posterior al primer modelo incluyen TCN, descriptores óseos como Bone Vector 126, Bone Covariance 168 y Bone Code 190, además de técnicas de optimización como SAM (Sharpness-Aware Minimization) y supervisón contrastiva entre firmantes. El detalle de los candidatos y sus resultados se documenta en la sección de investigación del repositorio.


7. RESULTADOS

7.1 Modelo de abecedario

El modelo de abecedario se entrenó con datos de 26 letras y alcanzó 99.06% de accuracy en validación, estabilizándose alrededor de la época 15 antes de que EarlyStopping detuviera el entrenamiento. El modelo final pesa 45 KB y predice en menos de 50 ms por cuadro, lo que lo hace viable para uso en tiempo real incluso en equipos modestos.

Sobre el conjunto de prueba, de 9,255 muestras el modelo clasificó correctamente 9,058 (97.9% con 95% de confianza). Los 197 errores restantes se concentraron en pares de letras con formas de mano parecidas, sobre todo entre J y K, cuyo movimiento es similar en la fase inicial del gesto.

Estos resultados confirman que, para señas estáticas o con una dinámica corta y bien definida, la combinación de MediaPipe con una red Conv1D relativamente ligera es suficiente para un sistema preciso y rápido.

7.2 Modelo de vocabulario

El segundo modelo, entrenado sobre 210 palabras, tuvo un resultado muy distinto. El accuracy en validación fue de 42.4% (con 671 muestras en el conjunto de prueba), muy por debajo de la meta de 85%. Este resultado se reporta sin atenuantes porque forma parte de las conclusiones técnicas del proyecto.

La causa más probable no es la arquitectura. Es la misma Conv1D+BiLSTM que funcionó bien para el abecedario. El problema está en la cantidad de datos por clase. El conjunto se dividió en 1,470 secuencias de entrenamiento, 210 de validación y 210 de prueba, repartidas entre 210 clases. Un promedio de 17 muestras por clase, frente a las 27,358 fotos por letra del abecedario.

Esto no invalida el enfoque general del proyecto. Confirma que el cuello de botella para el LSM, igual que para EnSeñas AI y buena parte de los proyectos revisados en los antecedentes, sigue siendo la disponibilidad de datos etiquetados, no la arquitectura del modelo. El abecedario funcionó bien porque tuvimos suficientes datos. El vocabulario, todavía no.

7.3 Candidatos investigados para el vocabulario

Después del primer modelo de vocabulario, la investigación exploró múltiples arquitecturas candidatas con el objetivo de subir el accuracy. El mejor candidato registrado hasta la fecha de corte de este documento es la fusión multivista con consistencia Jensen-Shannon, que alcanzó macro-F1 de S08 de 0.202584 en el benchmark de 210 palabras, con validación en tres semillas y confirmación leave-one-signer-out.

Varias candidatas quedaron por debajo del control de recuperación y fueron rechazadas. Otras se encuentran en evaluación activa. El repositorio de investigación documenta el protocolo, los resultados y el estado de cada candidata. S09 permanece cerrado.

7.4 Validación del corpus externo ICKMejia

Como verificación cruzada, se entrenó una TCN simple sobre el corpus ICKMejia (30 señas, 3,000 secuencias, keypoints 3D precalculados) usando la partición de prueba publicada. El resultado fue macro-F1 de 0.919, lo que valida que la ruta de extracción de keypoints y entrenamiento produce resultados comparables con datos públicos de otra fuente. Este resultado no se mezcla con el benchmark de palabras.


8. CONCLUSIONES

Aportación específica de este proyecto

A diferencia de Daktilos, que solo cubre abecedario, y del guante mioeléctrico del IPN, que requiere hardware adicional, este proyecto aporta un sistema que reconoce tanto letras como un vocabulario inicial de palabras, usando exclusivamente una cámara estándar y sin sensores. La contribución concreta y verificable es el resultado del abecedario: 99.06% de accuracy con un modelo de 45 KB, entrenado con un dataset propio de 574,064 imágenes que se documenta y podría reutilizarse en trabajos futuros sobre LSM.

Este proyecto demuestra que es posible construir, con herramientas de código abierto y una computadora estándar, un sistema de reconocimiento del abecedario LSM que funciona en tiempo real y con una precisión alta. La combinación de tres decisiones técnicas explica gran parte de ese resultado. Usar keypoints de MediaPipe en lugar de imágenes crudas hace al sistema invariante a la iluminación y al fondo. La augmentación respeta la anatomía real de la mano. La arquitectura Conv1D relativamente ligera no depende de GPUs costosas para entrenarse ni para ejecutarse.

El modelo de vocabulario, en cambio, no alcanzó la meta planteada. Con 42.4% de accuracy sobre 210 clases y un promedio de 17 muestras por clase, queda claro que el vocabulario del LSM necesita mucho más dato etiquetado del que logramos reunir. Esto coincide con lo que reportan otros equipos, como EnSeñas AI. El obstáculo real no es la arquitectura del modelo, es la escasez de datos.

Durante el desarrollo se identificaron problemas concretos que cualquier equipo que intente replicar este trabajo probablemente encuentre también. La letra Ñ terminó sin ninguna fotografía utilizable y tuvo que excluirse del entrenamiento. El dataset de palabras tenía 35 duplicados exactos que se detectaron por hash MD5. El delegate de GPU para TFLite no estaba disponible en el entorno de despliegue y hubo que caer de vuelta a CPU. Y una confusión inicial entre los modos IMAGE y VIDEO de MediaPipe generó detecciones inconsistentes hasta que se corrigió.

Alcances y limitaciones

El sistema reconoce 26 letras del abecedario LSM en tiempo real con 99.06% de accuracy, usando solo una cámara estándar. Se documenta y publica un dataset propio de imágenes y video del LSM, reutilizable por otros equipos. La app móvil en Expo y React Native ejecuta los modelos con ONNX Runtime, sin enviar datos a la nube, y deja trazas de auditoría de cada predicción.

El modelo de vocabulario (42.4% de accuracy) no es apto para uso real todavía. Se reporta como trabajo en progreso, no como resultado final. El sistema no ha sido probado con usuarios sordos reales fuera del equipo de desarrollo. La letra Ñ no está cubierta. Las pruebas se realizaron en condiciones de laboratorio controladas. El desempeño en condiciones variables de uso real no se ha evaluado todavía.

Próximos pasos

A corto plazo hay que ampliar la recolección de video para las letras dinámicas J, K, Q, X, Z y resolver el caso de la Ñ. También abrir una convocatoria de grabación colaborativa para las categorías de vocabulario con menos ejemplos.

A mediano plazo, evaluar el modelo de vocabulario con usuarios sordos reales y medir su experiencia de uso, no solo el accuracy. Probar el sistema en condiciones variables de iluminación y calidad de cámara, fuera del entorno controlado.

A largo plazo, iniciar la integración con Google Meet, Microsoft Teams y Zoom, antes de escalar hacia WhatsApp, Instagram y Facebook. La meta es que el sistema funcione dentro de las apps que la gente ya usa, no que se descargue una app más.

Una vía concreta para resolver el cuello de botella de datos es invitar a más personas a grabarse haciendo señas específicas, empezando por las letras dinámicas y las categorías de vocabulario más débiles. Abrir la recolección de datos a la colaboración permitiría ampliar el dataset a un ritmo que un equipo pequeño no puede lograr solo.


9. GLOSARIO TÉCNICO

Keypoint: punto clave con coordenadas (x, y, z) que representa una articulación o referencia anatómica detectada por MediaPipe.

Conv1D: capa convolucional que opera sobre una dimensión (el tiempo), detectando patrones locales entre cuadros cercanos de una secuencia.

BiLSTM: red recurrente que procesa una secuencia en ambas direcciones, integrando el contexto temporal completo antes y después de cada punto.

Softmax: función que convierte las salidas de la red en probabilidades que suman 1, una por cada clase posible.

Accuracy: porcentaje de predicciones correctas sobre el total de muestras evaluadas.

Overfitting: cuando un modelo memoriza el conjunto de entrenamiento en vez de aprender patrones generales, y su desempeño cae en datos nuevos.

Augmentación de datos: generación de variantes artificiales de una muestra (rotación, escalado, ruido) para ampliar el dataset de entrenamiento.

TFLite: formato de TensorFlow optimizado para ejecutar modelos en dispositivos con recursos limitados (móviles, embebidos).

Cuantización: técnica que reduce la precisión numérica de los pesos de un modelo para disminuir su tamaño en disco y memoria.

Focal Loss: función de pérdida que reduce el peso de los ejemplos fáciles y concentra el aprendizaje en los ejemplos difíciles o poco frecuentes.

Macro-F1: promedio del F1-score por clase, sin ponderar por frecuencia. Cada clase contribuye igual, sin importar cuántas muestras tenga.

Leave-One-Signer-Out (LOSO): estrategia de validación que entrena dejando a un firmante fuera y prueba sobre ese firmante, rotando por todos los firmantes.

ONNX Runtime: motor de inferencia que ejecuta modelos en formato ONNX en CPU o GPU, sin depender de TensorFlow.

TCN (Temporal Convolutional Network): red convolucional que opera sobre secuencias temporales con convoluciones dilatadas para capturar dependencias de largo alcance.

S09: el conjunto de prueba del benchmark de 210 palabras, mantenido cerrado durante toda la experimentación. Solo se evalúa cuando un candidato pasa las puertas de validación.


10. REFERENCIAS

[1] Instituto Nacional de Estadística y Geografía (INEGI). (2020). Censo de Población y Vivienda 2020. Discapacidad. https://www.inegi.org.mx/temas/discapacidad/

[2] Instituto Nacional de Estadística y Geografía (INEGI). (2021). Prevalencia de la población con discapacidad y/o problema o condición mental. Comunicado de prensa 713/21.

[3] Lugares, C. et al. (2024). Sign language images dataset from Mexican sign language. Data in Brief, 54. https://doi.org/10.1016/j.dib.2024.110533

[4] Google AI. (2024). MediaPipe Hand Landmarker. https://google.github.io/mediapipe/solutions/hands.html

[5] Abecerril, A. et al. (2024). MSL-150 Dataset: Mexican Sign Language with Keypoints. GitHub. https://github.com/armandobecerril/MSL-150-Dataset

[6] Zhang, X. et al. (2020). HandAugment: A Simple Data Augmentation Method for 3D Hand Pose Estimation. arXiv:2003.10544.

[7] Chollet, F. (2021). Deep Learning with Python (2nd ed.). Manning Publications.

[8] UNESCO. (2023). Global Education Monitoring Report: Technology in Education. https://www.unesco.org/gem-report

[9] López Nava, I.H. et al. (2024). Daktilos: Plataforma de aprendizaje de LSM con visión por computadora. CICESE, Ensenada.

[10] Martínez, R. et al. (2024). Reconocimiento de la Lengua de Señas Mexicana usando Deep Learning. Ciencia Latina, 8(4), 14458.

[11] RIDE Revista. (2025). Aplicación Móvil Basada en Deep Learning para la Inclusión Educativa de Personas Sordas. RIDE, 15(29).

[12] EnSeñas AI. (2026). The Neural Translation Engine. https://ensenas.ai/

[13] IMON. (2024). Desafíos de las personas sordas. Excélsior. https://www.excelsior.com.mx/nacional/inclusion-personas-sordas/1676008

[14] INEGI. (2022). Encuesta Nacional sobre Discriminación (ENADIS) 2022. https://www.inegi.org.mx/programas/enadis/2022/

[15] Caballero Hernández, X. (2025). Desarrollo de Software dedicado a la traducción de la Lengua Mexicana de Señas mediante Deep Learning y Machine Learning. PAAKAT: Revista de Tecnología y Sociedad. https://paakat.cugdl.udg.mx/index.php/paakat/article/view/897

[16] Lugares, I. et al. (2025). Mexican Sign Language Recognition Dataset. MDPI Electronics, 14(7), 1423. https://www.mdpi.com/2079-9292/14/7/1423

[17] ADN40. (2024). Guante traductor de Lengua de Señas Mexicana del IPN gana premio James Dyson Award México. https://www.adn40.mx/ciencia/2024-10-31/ipn-guante-traductor-lengua-senas-mexicana-gana-premio-internacional/

[18] Grand View Research. (2024). Digital Accessibility Software Market Size & Share Report, 2024-2030. https://www.grandviewresearch.com/industry-analysis/digital-accessibility-software-market-report

[19] La Jornada. (2024, diciembre 3). Daktilos y Sordbo, herramientas del CICESE para la comunidad sorda. https://www.jornada.com.mx/2024/12/03/ciencias/a06n1cie

[20] Dilo con Señas. (2024). Asociación Civil, Monterrey. https://nacersordo.com/en/apps/

[21] INEGI y CONAPO. (2014). Encuesta Nacional de la Dinámica Demográfica (ENADID) 2014.

[22] Espejel et al. (2024). Mexican sign language dataset. Mendeley Data V1. https://data.mendeley.com/datasets/6rj76z6y3n/1.


11. PENDIENTES ANTES DE LA VERSIÓN FINAL

Esta sección concentra los elementos que todavía requieren información o material del equipo antes de que el documento esté listo para entrega.

Datos por confirmar

- Número de FIPI, nombres reales de los participantes y del asesor (portada). Actualmente son texto de marcador de posición.
- Confirmar si la letra Ñ tiene 200 videos grabados o 0 fotografías utilizables. El documento actual la trata como excluida por falta de datos.
- Verificar en femeci.mx el nombre oficial vigente de la categoría del proyecto. El documento usa "Sistemas informáticos".
- Tiempo real de entrenamiento y modelo específico de GPU utilizado, para la sección de reproducibilidad.
- Lista específica de las 3-4 letras del abecedario que aún necesitan más datos de entrenamiento.

Fotografías y capturas por agregar

- Fotografía real de una seña del abecedario LSM tomada del dataset (sección 6.1).
- Captura de pantalla de una mano con los 21 keypoints de MediaPipe superpuestos, en pleno pipeline (sección 6.3).
- Ejemplos de una misma muestra antes y después de aplicar augmentación (sección 6.5).

Documentación de cumplimiento

- Constancia de consentimiento informado de las personas fotografiadas o grabadas para el dataset, si el proyecto la requiere para la fase de evaluación de cumplimiento de FEMECI.
- Confirmar si la convocatoria de este año pide una sección de presupuesto no incluida actualmente.
