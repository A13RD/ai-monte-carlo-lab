# AI Monte Carlo Lab

Laboratorio y motor de simulación estocástica para modelar el consumo de tokens y los costos financieros de inferencia en modelos de lenguaje de gran escala (LLMs).

## Arquitectura por Fases

- **Fase 1: Motor de Simulación Monte Carlo**
  - Modelado de demanda mensual ($N$) vía Poisson, Binomial Negativa o fija.
  - Modelado de tokens de entrada y salida ($T_{\text{in}}, T_{\text{out}}$) con distribuciones Log-Normal, Gamma, Normal truncada y Uniforme.
  - Modelo afín de latencia con fluctuación estocástica (*jitter*).
  - Cálculo vectorizado por lotes mediante `np.add.reduceat` y sumas compuestas.
  - Estimadores estadísticos: media muestral, desviación estándar ($ddof=1$), percentiles ($P_5, P_{50}, P_{95}$) y probabilidad de exceder presupuesto $P(C > B)$.

- **Fase 2: Modelo Predictivo de Demanda**
  - Generación de series temporales con tendencia y estacionalidad semanal.
  - Partición cronológica estricta (*train/test split*) sin filtración temporal (*data leakage*).
  - Regresión Poisson GLM con enlace logarítmico (`PoissonRegressor`).
  - Modelo de referencia (*Baseline*) basado en promedios por día de la semana.
  - Evaluación out-of-sample (MAE, RMSE, Devianza) y proyección sobre horizonte futuro.
  - Coordinación inmutable mediante `dataclasses.replace` hacia la configuración de Monte Carlo.

- **Fase 3: Validación Estadística Avanzada y Análisis de Incertidumbre**
  - **Convergencia:** Checkpoints adaptativos, error estándar de Monte Carlo ($O(1/\sqrt{n})$), intervalos de confianza Student-t / Normal para la media, e intervalo de Wilson Score para la probabilidad de exceder el presupuesto.
  - **Sensibilidad:** Análisis *One-At-a-Time* (OAT), elasticidades normalizadas, tratamiento de parámetros con valor base cero y ranking para diagrama Tornado.
  - **Correlaciones:** Cálculo de Pearson $r$ y Spearman $\rho_s$ con p-valores, distinguiendo asociación empírica, dependencia construida por el generador y causalidad en sistemas reales. Muestreo conjunto opcional vía cópula gaussiana bivariada.
  - **Comparación Analítica:** Formulación teórica de sumas compuestas (Identidad de Wald y Ley de Varianza Total / Fórmula de Eve), evaluación de momentos de demanda (Poisson, Binomial Negativa, Fixed) y prueba de hipótesis con puntuación $Z$ ($Z_{\text{score}}$).

---

## Estructura del Repositorio

```text
ai-monte-carlo-lab/
├── backend/
│   ├── pyproject.toml
│   ├── src/
│   │   └── ai_monte_carlo/
│   │       ├── config.py
│   │       ├── __init__.py
│   │       ├── __main__.py
│   │       ├── analysis/
│   │       │   ├── __init__.py
│   │       │   ├── analytical.py
│   │       │   ├── convergence.py
│   │       │   ├── correlation.py
│   │       │   └── sensitivity.py
│   │       ├── data/
│   │       │   ├── demand.py
│   │       │   └── synthetic.py
│   │       ├── models/
│   │       │   └── demand.py
│   │       ├── services/
│   │       │   └── experiment.py
│   │       └── simulation/
│   │           ├── generators.py
│   │           ├── metrics.py
│   │           └── monte_carlo.py
│   └── tests/
│       ├── test_analytical.py
│       ├── test_convergence.py
│       ├── test_correlation.py
│       ├── test_demand_data.py
│       ├── test_demand_model.py
│       ├── test_experiment_service.py
│       ├── test_metrics.py
│       ├── test_monte_carlo.py
│       ├── test_sensitivity.py
│       └── test_synthetic.py
└── docs/
    ├── decisiones-tecnicas.md
    ├── modelo-matematico.md
    ├── modelo-predictivo.md
    └── validacion-avanzada.md
```

---

## Instalación y Requisitos

Requiere Python $\ge 3.10$.

```powershell
cd backend
python -m pip install -e ".[dev]"
```

Dependencias principales:
- `numpy >= 1.26.0`
- `pandas >= 2.0.0`
- `scipy >= 1.10.0`
- `scikit-learn >= 1.4.0`
- `pytest >= 8.0.0` (desarrollo)

---

## Ejecución de Pruebas

Para correr la suite completa de 97 pruebas automatizadas:

```powershell
cd backend
python -m pytest
```

---

## Ejecución de Demostración CLI

Para ejecutar la demostración completa de las tres fases:

```powershell
cd backend
python -m ai_monte_carlo
```

---

## Documentación Técnica y Académica

- [Modelo Matemático de la Fase 1](docs/modelo-matematico.md)
- [Decisiones Técnicas de la Fase 1](docs/decisiones-tecnicas.md)
- [Modelo Predictivo y Coordinación de la Fase 2](docs/modelo-predictivo.md)
- [Validación Estadística Avanzada y Análisis de Incertidumbre de la Fase 3](docs/validacion-avanzada.md)

