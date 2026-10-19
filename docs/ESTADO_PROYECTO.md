# Estado del proyecto

## Funcionalidad verificada

- Motor Monte Carlo reproducible con demanda configurable, consumo de tokens, tarifa y métricas muestrales.
- Modelo Poisson GLM entrenado con datos sintéticos, evaluación cronológica fuera de muestra y comparación con baseline.
- Herramientas de validación estadística: convergencia, sensibilidad, correlaciones y comparación analítica.
- API FastAPI con rutas de salud, simulación y predicción; la interfaz nueva consume la API de simulación.
- Interfaz Next.js en español con formulario de parámetros, métricas, histograma de muestras reales y comparación de demanda +25%.
- Límite de seguridad de 30 millones de solicitudes simuladas esperadas por ejecución en la API y CORS configurable por origen.
- Configuración inicial de despliegue para API en Render; Next.js puede desplegarse en Vercel.

## Comprobaciones ejecutadas

- Estado inicial: `python -m pytest` — 122 pruebas aprobadas.
- La instalación de desarrollo documentada desde `backend/` con `python -m pip install -e ".[dev]"` se completó correctamente.
- Después de los cambios: `python -m pytest` — 126 pruebas aprobadas; las 2 advertencias corresponden a la integración Starlette/httpx y al alias HTTP 422 de FastAPI.
- `npm run typecheck` — aprobado.
- `npm run build` — compilación de producción aprobada con Next.js 16.4.2.
- `npm audit --audit-level=moderate` — cero vulnerabilidades reportadas.
- `python -m pip check` — sin dependencias Python rotas.
- Prueba local en navegador: formulario carga configuración de API, ejecuta simulación, muestra 8 bins que suman las 20 iteraciones de prueba y compara el escenario +25%.

## Configuración local

1. Backend: desde `backend/`, instalar `python -m pip install -e ".[dev]"` y ejecutar `python -m uvicorn ai_monte_carlo.api.app:app --reload --port 8000`.
2. Frontend: desde `frontend/`, copiar `.env.example` a `.env.local`, ejecutar `npm install` y `npm run dev`.
3. Verificar `http://127.0.0.1:8000/health` y `http://localhost:3000`.

## Despliegue pendiente de intervención manual

- Crear el servicio Render usando `render.yaml` y configurar `AI_MONTE_CARLO_CORS_ORIGINS` con el origen real de Vercel.
- Desplegar `frontend/` en Vercel con `frontend/` como raíz; configurar `NEXT_PUBLIC_API_URL` con la URL HTTPS real del API Render y volver a desplegar.
- Verificar `https://<api-render>/health` y ejecutar un experimento pequeño en la URL real de Vercel.
- Confirmar los orígenes/dominios asignados por las plataformas; no se documentan URLs ficticias ni se afirma que el proyecto ya esté desplegado.

## Limitaciones conocidas

- Las series de entrenamiento, pruebas y simulaciones son sintéticas y no representan telemetría real.
- La API limita el trabajo esperado; la carga aleatoria observada puede fluctuar alrededor de ese valor esperado. El motor estadístico de biblioteca conserva su rango de uso fuera de la API.
- La predicción GLM y el resumen de demanda son aproximaciones basadas en tendencia lineal y estacionalidad semanal.
- La validación de una URL de producción y de CORS requiere completar el despliegue y configurar los dominios reales.

`backend/.env.example` es una referencia de variables: el backend lee el entorno del proceso y no carga archivos `.env` automáticamente. `frontend/.env.example` sí se copia a `.env.local`, que Next.js carga al iniciar/build.
