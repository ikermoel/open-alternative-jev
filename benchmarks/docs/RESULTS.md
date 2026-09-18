# Resultados del piloto — 17 de septiembre de 2026

Trabajo Slurm 32677666: completado, código de salida 0. Qwen3.6-27B oficial, bitsandbytes 8-bit, GPU H200 MIG 2g.35gb. 300 preguntas MMLU test seleccionadas con semilla fija. No se entrenó el modelo ni se modificó el motor.

| Modo | Aciertos | Precisión | Preguntas/segundo |
|---|---:|---:|---:|
| A: una por una | 253/300 | 84,33 % | 3,14 |
| B: batch de tres | 252/300 | 84,00 % | 4,01 |
| C: tres concatenadas | 252/300 | 84,00 % | 5,57 |
| C con orden rotado | 253/300 | 84,33 % | 5,57 |

C procesó 1.39 veces más preguntas por segundo que B (aproximadamente 28 % menos tiempo de forward total), y 1,78 veces más que A. Este tiempo excluye carga del modelo, tokenización y preparación de tensores: no es latencia de servicio extremo a extremo. Una sola corrida; falta repetir mediciones de rendimiento.

La diferencia C−A fue −0,33 puntos porcentuales, con intervalo bootstrap por grupos del 95 % entre −2,67 y +2,00 puntos. Para C−B, intervalo entre -2.33 y 2.67 puntos. No demuestra equivalencia ni descarta caídas pequeñas.

Cambiar el orden alteró 28/300 respuestas (9,33 %), aunque la precisión total quedó parecida. C difirió de A en 19/300 respuestas; B difirió de A en 6/300. En las primeras posiciones de C, que tienen el mismo prefijo causal que A, cambiaron 3/100 respuestas. También variaron probabilidades con ese prefijo idéntico: por tanto, no se puede atribuir toda variación a interferencia semántica. La cuantización y las distintas formas de cómputo son posibles causas pendientes de aislar.

Memoria máxima asignada observada: 28,56 GiB (30,66 GB decimales); máxima reservada por PyTorch: 29,20 GiB. Son métricas del allocator, no toda la memoria que podría registrar el driver. La partición asignada tenía 34.896.609.280 bytes.

Las probabilidades son softmax sobre A/B/C/D, sin calibrar. ECE aproximado: A 0,0645; B 0,0697; C 0,0586. Brier: A 0,2274; B 0,2284; C 0,2326. NLL: A 0,4836; B 0,4953; C 0,5064. No hay evidencia de una mejora uniforme en calidad probabilística; el ECE con 300 ejemplos es ruidoso. No se ajustó temperatura con etiquetas de test.

## Interpretación

La idea supera este piloto: mayor velocidad que batching convencional y precisión agregada similar, dentro del límite de VRAM. No demuestra independencia entre preguntas, confianza calibrada ni equivalencia con Jev. Siguiente paso propuesto, aún no ejecutado: muestra mayor y repeticiones temporales, separando sensibilidad al orden de efectos numéricos.

## Archivos

Resultados crudos, métricas, auditoría y metadatos: `results/mmlu_32677666/`. Dataset y revisiones: `data/`. Log completo: `logs/mmlu_32677666.log`. El primer trabajo 32677581 falló antes de recoger medidas por un cambio de interfaz del tokenizador, corregido con `return_dict=False`; no entra en estos resultados.

## Análisis posterior del piloto — 17 de septiembre de 2026

Sin GPU, sobre los registros por llamada de `results/mmlu_32677666/` (`scripts/analyze_run.py`, salida en `analysis.json`).

**La ventaja de velocidad de C sobre B es padding, no ahorro de cómputo.** Regresión tiempo = fijo + pendiente × tokens_con_padding, por modo:

| Modo | Fijo por llamada | ms por token | r | Tokens desperdiciados en padding |
|---|---:|---:|---:|---:|
| A | 0,217 s | 0,77 | 0,986 | 0 % |
| B | 0,163 s | 0,96 | 0,998 | 35,1 % |
| C | 0,148 s | 0,97 | 0,997 | 0 % |

B y C cuestan lo mismo por token procesado. B rellena las tres secuencias hasta la más larga y procesa 35 % más tokens; eso, más el costo fijo, explica el 1,39×. Un motor sin padding (vLLM, SGLang, o HF con atención de longitud variable) borraría esa ventaja para preguntas no relacionadas. El ahorro estructural de concatenar solo existe con un estado compartido: P + k·q tokens frente a k·(P + q). Eso es lo que mide la v2 con RACE.

**Hay interferencia real, no solo ruido numérico.** La posición 0 de C tiene exactamente el mismo prefijo causal que A; sus diferencias son el piso de ruido (kernels int8 con distinta forma de batch). Posiciones 1 y 2 de C están 3–4 veces por encima de ese piso, aunque la precisión agregada no cayó.

| Modo y posición | Respuestas cambiadas frente a A | Media de \|Δp\| | Máximo de \|Δp\| |
|---|---:|---:|---:|
| B, cualquier posición | 2/100 | 0,010–0,014 | 0,29 |
| C, posición 0 (piso de ruido) | 3/100 | 0,013 | 0,15 |
| C, posición 1 | 11/100 | 0,047 | 0,77 |
| C, posición 2 | 5/100 | 0,053 | 0,60 |

**Calibración.** El modelo sale sobreconfiado unos 6 puntos (confianza media 0,90, precisión 0,84). Escalado de temperatura ajustado en dos pliegues cruzados, sin evaluar nunca sobre el pliegue de ajuste: T ≈ 1,45–1,55 en todos los modos; ECE de 0,064 a 0,032 en A y de 0,059 a 0,028 en C. Con 300 ejemplos el ECE es ruidoso; la dirección es clara, la magnitud no.

**Arquitectura del modelo.** `config.json` de Qwen3.6-27B: 64 capas, 48 de atención lineal (Gated DeltaNet) y 16 de atención completa, intervalo 4. En el entorno del clúster no hay `fla`, `triton`, `causal_conv1d` ni `flash_attn`: la atención lineal corre en el respaldo de PyTorch. Consecuencias: los tiempos absolutos no representan un motor optimizado, y B no puede empaquetarse con una máscara bloque-diagonal en esta arquitectura, porque el estado recurrente cruza cualquier máscara.

## Siguiente paso preparado: v2 (pendiente de aprobación, no enviado)

Scripts y datos ya están en el clúster y probados en la Mac con Qwen2.5-0.5B en CPU (A_pad, B y la posición 0 de C reproducen A hasta 1e-6).

- `mmlu1200.jsonl`: las 300 del piloto más 900 nuevas disjuntas, grupos de 12. Modos A, A_pad (control de ruido de forma), B3, C3, C3_rot, C6, C12. Estimación: unos 40 minutos.
- `race1000.jsonl`: 250 pasajes de RACE-high test con exactamente 4 preguntas. Modos A, B4, C4, C4_rot; en C el pasaje va solo en el primer turno. Estimación: unos 30 minutos. En la prueba local C4 procesó 2,6 veces menos tokens que A.

```bash
sbatch --export=ALL,DATASET=mmlu scripts/benchmark_v2.sbatch
sbatch --export=ALL,DATASET=race scripts/benchmark_v2.sbatch
```

## Resultados v2, parte 1: estado compartido (RACE) — 17 de septiembre de 2026

Trabajo Slurm 32679039, código de salida 0. 250 pasajes de RACE-high test con exactamente 4 preguntas cada uno (1000 preguntas), mismo modelo y cuantización que el piloto. En C el pasaje aparece solo en el primer turno; las tres preguntas siguientes van como turnos nuevos sin repetir el pasaje. Archivos en `results/v2_race_32679039/`, log en `logs/v2_32679039.log`.

| Modo | Precisión | Preguntas/segundo | Tokens procesados | ECE (crudo) |
|---|---:|---:|---:|---:|
| A: una por una | 92,60 % | 1,66 | 468.583 | 0,028 |
| B4: batch de cuatro | 92,80 % | 2,00 | 481.924 (2,8 % padding) | 0,031 |
| C4: pasaje una vez, cuatro preguntas | 92,90 % | 4,55 | 186.898 | 0,011 |
| C4 con orden rotado | 93,60 % | 4,57 | 186.898 | 0,015 |

**Aquí el ahorro sí es real.** C4 procesa 2,5 veces menos tokens que A o B4 y responde 2,3 veces más preguntas por segundo que B4. B4 casi no tiene padding (los pasajes dominan la longitud), así que esta ventaja no es artefacto: es la estructura P + 4·q frente a 4·(P + q). El costo por token es el mismo en todos los modos (0,95–0,97 ms), como en el piloto.

**La precisión no cae.** C4 − A = +0,3 puntos, intervalo bootstrap por pasaje del 95 % entre −0,9 y +1,4. C4 − B4 = +0,1 puntos, intervalo entre −1,0 y +1,2. Por posición, C coincide con A sobre las mismas preguntas dentro del ruido:

| Posición en C4 | Precisión C4 | Precisión A, mismas preguntas | Respuestas cambiadas frente a A |
|---|---:|---:|---:|
| 0 (mismo prefijo que A) | 93,2 % | 93,6 % | 1/250 |
| 1 | 94,4 % | 93,2 % | 9/250 |
| 2 | 93,2 % | 92,4 % | 10/250 |
| 3 | 90,8 % | 91,2 % | 19/250 |

**Pero la interferencia crece con la posición.** Los cambios de respuesta frente a A pasan de 1 a 19 por cada 250 conforme la pregunta está más lejos del pasaje, con cambios simétricos (tantos aciertos ganados como perdidos). Rotar el orden cambió 2,4 % de las respuestas. Con cuatro preguntas no cuesta precisión; no sabemos qué pasa con más.

**Calibración.** C4 sale menos sobreconfiado que A (confianza media 0,927 frente a 0,949) y su ECE crudo es menor (0,011 frente a 0,028). La temperatura cruzada apenas lo mueve (T ≈ 1,0–1,1), mientras que A necesita T ≈ 1,3. No hay explicación establecida para esta diferencia; con 1000 ejemplos el ECE sigue siendo ruidoso.

Memoria máxima asignada: 28,3 GiB en C4 y 29,3 GiB en B4, dentro de la partición de 35 GB.

## Resultados v2, parte 2: MMLU 1200 con grupos de 3, 6 y 12 — 17 de septiembre de 2026

Trabajo Slurm 32679038, código de salida 0, 48 minutos (compartió nodo con el trabajo de RACE la mayor parte del tiempo). 1200 preguntas: las 300 del piloto primero, más 900 nuevas disjuntas. Sobre las 300 del piloto, A reproduce exactamente el 84,33 % del piloto. Archivos en `results/v2_mmlu_32679038/`, log en `logs/v2_32679038.log`.

| Modo | Precisión | Diferencia con A (IC 95 % por grupo) | Preguntas/segundo | Tokens procesados | ECE (crudo) |
|---|---:|---:|---:|---:|---:|
| A: una por una | 84,17 % | — | 3,10 | 163.032 | 0,054 |
| A_pad: A rellenado a 1024 | 83,92 % | −0,25 (−1,1 a +0,7) | 0,91 | 1.228.800 | 0,057 |
| B3: batch de tres | 83,75 % | −0,42 (−1,2 a +0,3) | 3,88 | 253.638 (35,7 % padding) | 0,058 |
| C3: tres concatenadas | 84,00 % | −0,17 (−1,3 a +1,0) | 5,44 | 165.432 | 0,052 |
| C3 con orden rotado | 83,25 % | −0,92 (−2,1 a +0,3) | 5,46 | 165.432 | 0,055 |
| C6: seis concatenadas | 84,92 % | +0,75 (−0,6 a +2,2) | 6,21 | 166.032 | 0,047 |
| C12: doce concatenadas | 84,17 % | 0,00 (−1,8 a +1,7) | 6,71 | 166.332 | 0,038 |

**Precisión.** Ningún modo difiere de A más allá del ruido, hasta doce preguntas por secuencia. Con 1200 preguntas el intervalo es de unos ±1,2 puntos; caídas menores que eso siguen sin poder descartarse. Por posición dentro de C12, C y A coinciden sobre las mismas preguntas dentro de ±6 puntos con 100 preguntas por posición, sin tendencia clara a lo largo de la secuencia.

**Piso de ruido medido directamente.** A_pad tiene exactamente el mismo prefijo que A y solo cambia la forma del tensor: cambió 32 de 1200 respuestas (2,7 %), con diferencia media de probabilidad 0,0125. B3 da lo mismo (2,3 %, 0,0126). Ese es el ruido de los kernels int8 con distinta forma de batch, sin ninguna pregunta ajena en el contexto.

**Interferencia, separada del ruido.** Las posiciones 1 y 2 de C3 cambian 8 % de las respuestas frente a A, con diferencia media de probabilidad 0,036: tres veces el piso de ruido. C6 llega a 7,4 % y C12 a 8,7 % (media 0,050). Rotar el orden en C3 cambia 8,2 % de las respuestas. La interferencia es real y crece poco con el tamaño de grupo; los cambios son simétricos y no mueven la precisión agregada.

**Rendimiento.** El costo por token vuelve a ser el mismo en todos los modos (0,95–1,02 ms). C gana frente a A y B3 por dos razones ya identificadas: B3 desperdicia 35,7 % de tokens en padding, y cada llamada tiene un costo fijo de 0,15–0,22 s que C amortiza entre más preguntas. Sin estado compartido, C no ahorra cómputo por token; un motor con batching continuo sin padding reduciría la diferencia frente a B a la amortización del costo fijo.

**Calibración.** Sobreconfianza cruda de unos 5 puntos en A (confianza 0,896, precisión 0,842). El ECE crudo baja al concatenar más preguntas (0,054 en A, 0,038 en C12), igual que en RACE, y la temperatura cruzada lo lleva a 0,01–0,03 en todos los modos con T entre 1,3 y 1,5.

## Lectura conjunta de la v2

1. Leer varias posiciones de una pasada conserva la precisión de este modelo hasta 12 preguntas sin contexto compartido y 4 con pasaje compartido, dentro de ±1 punto.
2. El ahorro real de cómputo aparece solo con estado compartido: 2,5 veces menos tokens y 2,3 veces más preguntas por segundo en RACE frente a batching sin apenas padding. Sin estado compartido el ahorro es amortización y padding, no cómputo.
3. Las respuestas individuales sí se mueven por interferencia (6–9 % de cambios, tres veces el ruido numérico) y por el orden. Para una librería esto significa que una decisión concreta puede cambiar según qué otras preguntas la acompañen, aunque la tasa de acierto no cambie.
4. La confianza cruda está sobreestimada unos 5 puntos; un escalar de temperatura ajustado con validación cruzada la corrige casi por completo. Sigue sin evaluarse en otros modelos ni precisiones.
