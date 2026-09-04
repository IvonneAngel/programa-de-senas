# Sistema Diccionario Fantasma

- Modelo abecedario entrenado completo (PointNet 21 letras)
- Cuando IA detecta letra `H`, aparece en transparente `ola` (de `HOLA`) como fantasma
- No completa solo por poner H a mano, sino cuando IA detecta H
- Usa diccionario cotidiano 50k + LSM 249, trie 1KB, 0.02ms
- Ejemplo: detecta `H` -> muestra `H` + <span style="opacity:0.4">ola</span>
