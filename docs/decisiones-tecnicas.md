# Decisiones Técnicas: AI Monte Carlo Lab (Fase 1)

## 1. Arquitectura y Paradigma de Diseño

Para la Fase 1, se priorizó una arquitectura desacoplada, funcional y sin efectos colaterales, organizada de acuerdo al siguiente estándar:

- **Inmutabilidad y Tipado Fuerte (`config.py`):**
  Uso de `dataclasses(frozen=True)` y `Enum` nativos de Python. Validación preventiva de invariantes en `__post_init__` (e.g. tasas no negativas, $M \ge 2$, varianzas positivas).
- **Generadores Estocásticos Puros (`simulation/generators.py`):**
  Funciones independientes que reciben explícitamente una instancia de `numpy.random.Generator`. No utilizan estados globales ni módulos pseudoaleatorios no deterministas (`random` o `np.random.seed`).
- **Agregación Vectorizada por Lotes (`simulation/monte_carlo.py`):**
  Para evitar bucles de interpretación en Python al simular millones de tokens aleatorios, se implementó una estrategia vectorizada basada en `np.add.reduceat` combinada con particionamiento en bloques (*chunks* de 2,000 iteraciones). Esto garantiza una huella de memoria inferior a 50 MB y una ejecución de 10,000 iteraciones en menos de 0.2 segundos.
- **Métricas Estadísticas Independientes (`simulation/metrics.py`):**
  Funciones atómicas y reutilizables para el cálculo de medias, desviaciones estándar muestrales ($ddof=1$), percentiles continuos y probabilidades empíricas.
- **Generación de Telemetría Sintética (`data/synthetic.py`):**
  Integración con pandas DataFrames con tipos de datos explícitos (`int64`, `float64`, `datetime64[ns]`) e intervalos temporales modelados mediante procesos puntuales de renovación exponencial.

## 2. Decisiones Estadísticas Clave

1. **Corrección de Bessel ($ddof=1$):**
   La desviación estándar muestral aplica división por $n - 1$ para asegurar un estimador de varianza muestral insesgado. Se exige $n \ge 2$.
2. **Generador Pseudoaleatorio `numpy.random.default_rng`:**
   Se utiliza el algoritmo PCG-64 (estándar moderno de NumPy) en lugar del legado MT19937 (`np.random.seed`), garantizando mejores propiedades espectrales y reproducibilidad exacta entre plataformas.
3. **Validación de Convergencia con Identidad de Wald:**
   La suite de pruebas unitarias incluye pruebas de convergencia analítica que comparan la esperanza empírica del proceso de Poisson compuesto con el valor exacto derivado de la Identidad de Wald.
