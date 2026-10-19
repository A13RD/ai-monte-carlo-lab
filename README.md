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

- **Fase 4: API RESTful con FastAPI y Pydantic**
  - **Diseño Desacoplado y Modular:** Capa HTTP separada en esquemas (`api/schemas/`), rutas (`api/routes/`), aplicación (`api/app.py`) y motor computacional existente.
  - **Validación Estricta:** Esquemas Pydantic v2 fuertemente tipados con validación en límites físicos y matemáticos (parámetros no negativos, clipping ordenado, tamaños de muestra válidos).
  - **Reproducibilidad y Determinismo:** Semillas configurables en peticiones para auditoría y replicabilidad exacta.
  - **Endpoints Expuestos:**
    - `GET /health` y `GET /api/v1/health`: Estado de salud, versión del paquete y readiness de subsistemas.
    - `GET /docs` y `GET /redoc`: Documentación interactiva OpenAPI (Swagger UI y ReDoc).
    - `GET /api/v1/simulation/defaults`: Configuración nominal por defecto para simulaciones Monte Carlo.
    - `POST /api/v1/simulation/run`: Ejecución directa del motor Monte Carlo y cálculo de métricas financieras.
    - `POST /api/v1/demand/experiment`: Pipeline integrado de predicción Poisson GLM, evaluación out-of-sample y simulación Monte Carlo coordinada.
    - `POST /api/v1/demand/forecast`: Entrenamiento de Poisson GLM y proyección del horizonte futuro de demanda.
  - **Interfaz web:** Next.js + TypeScript; ejecuta la API real, muestra un histograma de las muestras simuladas y compara un escenario de demanda +25%.
  - **Protección de recursos:** máximo de 30 millones de solicitudes simuladas esperadas por ejecución y CORS configurable por origen.

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
│   │       ├── api/
│   │       │   ├── __init__.py
│   │       │   ├── app.py
│   │       │   ├── routes/
│   │       │   │   ├── __init__.py
│   │       │   │   ├── demand.py
│   │       │   │   ├── health.py
│   │       │   │   └── simulation.py
│   │       │   └── schemas/
│   │       │       ├── __init__.py
│   │       │       ├── common.py
│   │       │       ├── demand.py
│   │       │       └── simulation.py
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
│       ├── test_api.py
│       ├── test_convergence.py
│       ├── test_correlation.py
│       ├── test_demand_data.py
│       ├── test_demand_model.py
│       ├── test_experiment_service.py
│       ├── test_metrics.py
│       ├── test_monte_carlo.py
│       ├── test_sensitivity.py
│       └── test_synthetic.py
├── frontend/
│   ├── .env.example
│   ├── package.json
│   └── src/app/
├── render.yaml
└── docs/
    ├── decisiones-tecnicas.md
    ├── ESTADO_PROYECTO.md
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
- `pydantic >= 2.0.0`
- `fastapi >= 0.110.0`
- `uvicorn >= 0.28.0`
- `pytest >= 8.0.0` (desarrollo)
- `httpx >= 0.27.0` (desarrollo)

---

## Ejecución del Servidor API (Fase 4)

Para iniciar el servidor local de desarrollo con recarga automática:

```powershell
cd backend
python -m uvicorn ai_monte_carlo.api.app:app --host 127.0.0.1 --port 8000 --reload
```

O utilizando el punto de entrada integrado del paquete:

```powershell
cd backend
python -m ai_monte_carlo --serve --port 8000
```

Una vez iniciado el servidor:
- **Swagger UI (Documentación interactiva):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc (Documentación técnica):** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Especificación OpenAPI (JSON):** [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)
- **Health Check:** `GET http://127.0.0.1:8000/health`

### Ejemplo de Petición para Simulación Monte Carlo

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/simulation/run" \
  -H "Content-Type: application/json" \
  -d '{
    "n_iterations": 500,
    "budget_limit": 500.0,
    "pricing": {
      "cost_per_million_input": 2.50,
      "cost_per_million_output": 10.00,
      "fixed_monthly_cost": 20.00
    },
    "workload": {
      "mean_daily_requests": 200,
      "days_per_month": 30
    },
    "seed": 42
  }'
```

Con los mismos parámetros y `seed = 42`, el resultado reproducible observado fue:

| Métrica | Resultado |
|---|---:|
| Media del costo mensual | USD 49.988932 |
| Desviación estándar muestral | USD 0.428200 |
| P5 / P50 / P95 | USD 49.267083 / 49.996026 / 50.661549 |
| Probabilidad empírica de exceder USD 500 | 0.0% |

Estos números son una salida sintética del experimento descrito, no una medición de producción. La respuesta de la API incluye precisión completa; la misma configuración y semilla permiten repetirla.

---

## Ejecución de Pruebas

Para correr la suite completa de pruebas automatizadas (Fases 1, 2, 3 y 4):

```powershell
cd backend
python -m pytest
```

Para correr únicamente las pruebas de la API:

```powershell
cd backend
python -m pytest tests/test_api.py
```

## Interfaz web local

En una terminal inicia la API (ver comando anterior). En otra terminal:

```powershell
cd frontend
Copy-Item .env.example .env.local
npm install
npm run dev
```

Abre [http://localhost:3000](http://localhost:3000). El frontend obtiene la configuración inicial de `GET /api/v1/simulation/defaults` y envía cada experimento a `POST /api/v1/simulation/run`; no muestra valores ficticios como resultados.
Valida el frontend con `npm run typecheck` y `npm run build`.

## Variables de entorno

- Backend: `backend/.env.example` documenta `APP_ENV` y `AI_MONTE_CARLO_CORS_ORIGINS`; configura estas variables en el entorno del proceso/plataforma. En producción, define los orígenes HTTPS exactos, separados por comas. En local, FastAPI permite por defecto `localhost:3000` y `127.0.0.1:3000`.
- Frontend: `NEXT_PUBLIC_API_URL` es la URL base de FastAPI (por ejemplo, `http://127.0.0.1:8000` en local). En Vercel se debe configurar con la URL real del backend, nunca con localhost.
- No guardes `.env` ni `.env.local` con valores privados en Git.

## Despliegue

El archivo raíz `render.yaml` prepara un servicio Python en Render que instala `backend/`, inicia Uvicorn y comprueba `/health`. Para completar el despliegue:

1. Crea el proyecto de Vercel usando `frontend/` como directorio raíz y anota el origen HTTPS que Vercel le asigna.
2. En Render, crea un Blueprint desde este repositorio y establece `AI_MONTE_CARLO_CORS_ORIGINS` con ese origen HTTPS exacto (sin comodín).
3. Al terminar el despliegue, copia la URL HTTPS asignada al servicio Render a `NEXT_PUBLIC_API_URL` en Vercel y vuelve a desplegar el frontend.
4. Comprueba `https://<api-render>/health`, luego abre el sitio de Vercel y ejecuta un experimento pequeño.

Las URLs finales dependen de las cuentas y dominios del usuario; no se incluyen URLs inventadas ni se afirma que ya se haya desplegado.

---

## Ejecución de Demostración CLI

Para ejecutar la demostración académica completa de las fases 1 a 3 en consola:

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
- [Estado, pruebas y pasos de despliegue](docs/ESTADO_PROYECTO.md)
