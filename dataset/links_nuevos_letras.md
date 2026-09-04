# Links nuevos letras LSM (no usar los 5 viejos: sjt79hnb2f, feria, robbolla11, mendeley_6rj76z6y3n, MSL-ABC)

## 1. Dinamicas J,K,Ñ,Q,X,Z v2 (PRIORIDAD: completa alfabeto a 27) - CC-BY 4.0
- https://zenodo.org/api/records/19154645/files/MSL-dynamic-signs-frontal.7z/content (291,857,017 bytes) DESCARGADO 2026-09-03 VERIFICADO exacto
- https://zenodo.org/api/records/19154645/files/MSL-dynamic-signs-perfil.7z/content (289,355,074 bytes) DESCARGADO 2026-09-03 VERIFICADO exacto
- repo: https://zenodo.org/records/19154645
- ENTRENADO 2026-09-04: 1,242 videos -> landmarks MediaPipe Tasks (99.4% con mano) ->
  manifest (train 1002 / val 120 S17-S18 / test 120) -> pack -> TCN successor_positions126.
  `docs/analisis/fase2-dinamicas/best.pt`: train acc 0.998, val 0.942, best val macro_f1 0.9917,
  TEST acc 0.9667 macro_f1 0.9663 (120 muestras). Early stopping epoch 21/40.

## 2. LSM anotado (abecedario + numeros + palabras en video) - DESCARTADO 2026-09-03
- Se descargo ABECEDARIOIMAGENES.pickle (72 frames sin letras) + ABECEDARIO.pickle (solo nombres).
  Sin etiquetas NO sirve. BORRADO del proyecto y de Descargas. No volver a bajar.
- repo: https://zenodo.org/records/6554337

## 3. Kaggle MSL videos (requiere login kaggle)
- https://www.kaggle.com/datasets/sujaykapadnis/mexicansign-language-dataset
- cmd: kaggle datasets download -d sujaykapadnis/mexicansign-language-dataset

## 4. archive.zip 1.8GB en Descargas (2026-09-03) - DUPLICADO, NO integrar
- 31,417 jpgs, 249 clases MSLwords1/001..249 = mismo mendeley_6rj76z6y3n ya usado.

## 5. PALABRASIMAGES.pickle 221MB en Descargas (renombrado sin N, 2026-09-03) - PENDIENTE inspeccionar
- Mismo tamaño exacto del link #2 (221,190,240). Falta ver si trae etiquetas.

## Descartados (NO son LSM mexicano)
- kirlelea/spanish-sign-language-alphabet-static (lengua española, NO mexicana)
- walidlasseg/moroccan-sign-language-lsm-alphabet-dataset (marroqui, NO mexicano)
