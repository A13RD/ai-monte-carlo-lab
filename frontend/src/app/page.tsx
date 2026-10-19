"use client";

import { useEffect, useMemo, useState } from "react";

type SimulationRequest = {
  n_iterations: number;
  budget_limit: number;
  pricing: {
    cost_per_million_input: number;
    cost_per_million_output: number;
    fixed_monthly_cost: number;
  };
  input_tokens: { family: string; mean: number; std: number; min_tokens: number; max_tokens: number };
  output_tokens: { family: string; mean: number; std: number; min_tokens: number; max_tokens: number };
  workload: {
    family: string;
    mean_daily_requests: number;
    dispersion_parameter: number;
    days_per_month: number;
  };
  seed: number | null;
};

type SimulationResult = {
  iterations: number;
  budget_limit: number;
  probability_exceeding_budget: number;
  cost_metrics: {
    mean: number;
    sample_std: number;
    p5: number;
    p50_median: number;
    p95: number;
    min: number;
    max: number;
    count: number;
  };
  token_metrics: { mean: number; p95: number; count: number };
  cost_histogram: { lower_bound: number; upper_bound: number; count: number }[];
  seed: number | null;
  execution_time_ms: number;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "");
const formatUsd = (value: number) =>
  new Intl.NumberFormat("es-MX", { style: "currency", currency: "USD" }).format(value);
const formatNumber = (value: number) =>
  new Intl.NumberFormat("es-MX", { maximumFractionDigits: 1 }).format(value);

async function requestSimulation(payload: SimulationRequest): Promise<SimulationResult> {
  if (!API_BASE_URL) {
    throw new Error("Falta configurar NEXT_PUBLIC_API_URL para conectar con la API.");
  }

  const response = await fetch(`${API_BASE_URL}/api/v1/simulation/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map((item: { msg?: string }) => item.msg).join("; ")
      : data.detail;
    throw new Error(detail || `La API respondió con HTTP ${response.status}.`);
  }
  return data as SimulationResult;
}

function Histogram({ result }: { result: SimulationResult }) {
  const maxCount = Math.max(...result.cost_histogram.map((bin) => bin.count), 1);
  const chartWidth = 880;
  const chartHeight = 260;
  const gap = 3;
  const barWidth = chartWidth / result.cost_histogram.length - gap;

  return (
    <div className="chart-wrap">
      <svg
        className="histogram"
        viewBox={`0 0 ${chartWidth} ${chartHeight + 38}`}
        role="img"
        aria-label="Histograma de los costos mensuales simulados en dólares"
      >
        <line x1="0" y1={chartHeight} x2={chartWidth} y2={chartHeight} className="axis" />
        {result.cost_histogram.map((bin, index) => {
          const height = (bin.count / maxCount) * (chartHeight - 16);
          const x = index * (chartWidth / result.cost_histogram.length) + gap / 2;
          const y = chartHeight - height;
          const midpoint = (bin.lower_bound + bin.upper_bound) / 2;
          return (
            <rect
              key={`${bin.lower_bound}-${index}`}
              x={x}
              y={y}
              width={barWidth}
              height={height}
              rx="3"
              className={midpoint > result.budget_limit ? "bar bar-over-budget" : "bar"}
            >
              <title>
                {formatUsd(bin.lower_bound)}–{formatUsd(bin.upper_bound)}: {bin.count} iteraciones
              </title>
            </rect>
          );
        })}
        <text x="0" y={chartHeight + 25} className="axis-label">
          {formatUsd(result.cost_histogram[0]?.lower_bound ?? 0)}
        </text>
        <text x={chartWidth} y={chartHeight + 25} textAnchor="end" className="axis-label">
          {formatUsd(result.cost_histogram.at(-1)?.upper_bound ?? 0)}
        </text>
      </svg>
      <div className="chart-legend">
        <span><i className="legend-swatch" /> Debajo del presupuesto</span>
        <span><i className="legend-swatch over" /> Por encima del presupuesto</span>
      </div>
    </div>
  );
}

function MetricCard({
  label,
  value,
  explanation,
  accent,
}: {
  label: string;
  value: string;
  explanation: string;
  accent?: boolean;
}) {
  return (
    <article className={`metric-card${accent ? " metric-accent" : ""}`}>
      <p>{label}</p>
      <strong>{value}</strong>
      <span>{explanation}</span>
    </article>
  );
}

export default function Home() {
  const [request, setRequest] = useState<SimulationRequest | null>(null);
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [comparison, setComparison] = useState<SimulationResult | null>(null);
  const [loadingDefaults, setLoadingDefaults] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    async function loadDefaults() {
      if (!API_BASE_URL) {
        setError("Configura NEXT_PUBLIC_API_URL en frontend/.env.local para conectar con la API.");
        setLoadingDefaults(false);
        return;
      }
      try {
        const response = await fetch(`${API_BASE_URL}/api/v1/simulation/defaults`);
        if (!response.ok) throw new Error(`No se pudieron cargar los parámetros (HTTP ${response.status}).`);
        const defaults = (await response.json()) as SimulationRequest;
        if (active) setRequest(defaults);
      } catch (cause) {
        if (active) setError(cause instanceof Error ? cause.message : "No se pudo conectar con la API.");
      } finally {
        if (active) setLoadingDefaults(false);
      }
    }
    void loadDefaults();
    return () => {
      active = false;
    };
  }, []);

  const estimatedRequests = useMemo(
    () =>
      request
        ? request.n_iterations * request.workload.mean_daily_requests * request.workload.days_per_month
        : 0,
    [request],
  );

  function updateRequest(update: (current: SimulationRequest) => SimulationRequest) {
    setRequest((current) => (current ? update(current) : current));
  }

  async function runSimulation(compare = false) {
    if (!request) return;
    setBusy(true);
    setError("");
    setComparison(null);
    try {
      if (compare) {
        const higherDemand = {
          ...request,
          workload: {
            ...request.workload,
            mean_daily_requests: request.workload.mean_daily_requests * 1.25,
          },
        };
        const [baseline, stress] = await Promise.all([
          requestSimulation(request),
          requestSimulation(higherDemand),
        ]);
        setResult(baseline);
        setComparison(stress);
      } else {
        setResult(await requestSimulation(request));
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Ocurrió un error al ejecutar la simulación.");
    } finally {
      setBusy(false);
    }
  }

  if (loadingDefaults) {
    return <main className="center-state"><span className="spinner" /> Conectando con la API...</main>;
  }

  if (!request) {
    return (
      <main className="center-state">
        <div className="notice notice-error" role="alert">{error}</div>
        <p>Inicia FastAPI y revisa la variable de entorno del frontend.</p>
      </main>
    );
  }

  const metrics = result?.cost_metrics;

  return (
    <main className="page-shell">
      <header className="topbar">
        <a href="#" className="brand"><span className="brand-mark">MC</span> AI Monte Carlo Lab</a>
        <a className="api-link" href={`${API_BASE_URL}/docs`} target="_blank" rel="noreferrer">
          API interactiva <span aria-hidden="true">↗</span>
        </a>
      </header>

      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow"><span className="pulse-dot" /> LABORATORIO DE INCERTIDUMBRE</p>
          <h1>Entiende el costo<br /><em>antes de escalar.</em></h1>
          <p className="hero-description">
            Simula demanda y consumo de tokens para estimar el costo mensual, cuantificar el riesgo
            presupuestario y explorar escenarios reproducibles.
          </p>
        </div>
        <div className="hero-note">
          <span className="note-icon">∑</span>
          <p><strong>Resultados, no supuestos ocultos.</strong><br />Cada métrica proviene del motor Monte Carlo de la API.</p>
        </div>
      </section>

      {error && <div className="notice notice-error" role="alert">{error}</div>}

      <div className="workspace">
        <form
          className="panel config-panel"
          onSubmit={(event) => {
            event.preventDefault();
            void runSimulation();
          }}
        >
          <div className="panel-heading">
            <div><p className="eyebrow">CONFIGURACIÓN</p><h2>Define tu experimento</h2></div>
            <span className="step-number">01</span>
          </div>

          <fieldset disabled={busy}>
            <legend>Simulación</legend>
            <label className="field">
              <span>Iteraciones Monte Carlo</span>
              <input
                type="number" min="2" max="500000" required
                value={request.n_iterations}
                onChange={(event) => updateRequest((current) => ({ ...current, n_iterations: Number(event.target.value) }))}
              />
              <small>Más iteraciones suelen reducir el error Monte Carlo.</small>
            </label>
            <div className="field-grid">
              <label className="field">
                <span>Semilla reproducible</span>
                <input
                  type="number" value={request.seed ?? ""}
                  onChange={(event) => updateRequest((current) => ({ ...current, seed: event.target.value === "" ? null : Number(event.target.value) }))}
                />
              </label>
              <label className="field">
                <span>Días por ciclo</span>
                <input
                  type="number" min="1" max="365" required
                  value={request.workload.days_per_month}
                  onChange={(event) => updateRequest((current) => ({ ...current, workload: { ...current.workload, days_per_month: Number(event.target.value) } }))}
                />
              </label>
            </div>
          </fieldset>

          <fieldset disabled={busy}>
            <legend>Demanda y presupuesto</legend>
            <div className="field-grid">
              <label className="field">
                <span>Solicitudes promedio por día</span>
                <input
                  type="number" min="0" step="any" required
                  value={request.workload.mean_daily_requests}
                  onChange={(event) => updateRequest((current) => ({ ...current, workload: { ...current.workload, mean_daily_requests: Number(event.target.value) } }))}
                />
              </label>
              <label className="field">
                <span>Distribución de demanda</span>
                <select
                  value={request.workload.family}
                  onChange={(event) => updateRequest((current) => ({ ...current, workload: { ...current.workload, family: event.target.value } }))}
                >
                  <option value="poisson">Poisson</option>
                  <option value="negative_binomial">Binomial negativa</option>
                  <option value="normal">Normal</option>
                  <option value="fixed">Fija</option>
                </select>
              </label>
            </div>
            <div className="field-grid">
              <label className="field">
                <span>Presupuesto mensual (USD)</span>
                <input
                  type="number" min="0" step="any" required
                  value={request.budget_limit}
                  onChange={(event) => updateRequest((current) => ({ ...current, budget_limit: Number(event.target.value) }))}
                />
              </label>
              <label className="field">
                <span>Costo fijo mensual (USD)</span>
                <input
                  type="number" min="0" step="any" required
                  value={request.pricing.fixed_monthly_cost}
                  onChange={(event) => updateRequest((current) => ({ ...current, pricing: { ...current.pricing, fixed_monthly_cost: Number(event.target.value) } }))}
                />
              </label>
            </div>
          </fieldset>

          <fieldset disabled={busy}>
            <legend>Distribución de tokens por solicitud</legend>
            <div className="field-grid">
              <label className="field">
                <span>Distribución de entrada</span>
                <select
                  value={request.input_tokens.family}
                  onChange={(event) => updateRequest((current) => ({ ...current, input_tokens: { ...current.input_tokens, family: event.target.value } }))}
                >
                  <option value="lognormal">Log-normal</option>
                  <option value="gamma">Gamma</option>
                  <option value="normal">Normal truncada</option>
                  <option value="uniform">Uniforme</option>
                </select>
              </label>
              <label className="field">
                <span>Distribución de salida</span>
                <select
                  value={request.output_tokens.family}
                  onChange={(event) => updateRequest((current) => ({ ...current, output_tokens: { ...current.output_tokens, family: event.target.value } }))}
                >
                  <option value="lognormal">Log-normal</option>
                  <option value="gamma">Gamma</option>
                  <option value="normal">Normal truncada</option>
                  <option value="uniform">Uniforme</option>
                </select>
              </label>
            </div>
            <div className="field-grid">
              <label className="field">
                <span>Media de entrada (tokens)</span>
                <input
                  type="number" min="1" step="any" required
                  value={request.input_tokens.mean}
                  onChange={(event) => updateRequest((current) => ({ ...current, input_tokens: { ...current.input_tokens, mean: Number(event.target.value) } }))}
                />
              </label>
              <label className="field">
                <span>Media de salida (tokens)</span>
                <input
                  type="number" min="1" step="any" required
                  value={request.output_tokens.mean}
                  onChange={(event) => updateRequest((current) => ({ ...current, output_tokens: { ...current.output_tokens, mean: Number(event.target.value) } }))}
                />
              </label>
            </div>
            <div className="field-grid">
              <label className="field">
                <span>Desviación estándar de entrada</span>
                <input
                  type="number" min="0" step="any" required
                  value={request.input_tokens.std}
                  onChange={(event) => updateRequest((current) => ({ ...current, input_tokens: { ...current.input_tokens, std: Number(event.target.value) } }))}
                />
              </label>
              <label className="field">
                <span>Desviación estándar de salida</span>
                <input
                  type="number" min="0" step="any" required
                  value={request.output_tokens.std}
                  onChange={(event) => updateRequest((current) => ({ ...current, output_tokens: { ...current.output_tokens, std: Number(event.target.value) } }))}
                />
              </label>
            </div>
          </fieldset>

          <fieldset disabled={busy}>
            <legend>Tarifas de inferencia (USD / millón de tokens)</legend>
            <div className="field-grid">
              <label className="field">
                <span>Tokens de entrada</span>
                <input
                  type="number" min="0" step="any" required
                  value={request.pricing.cost_per_million_input}
                  onChange={(event) => updateRequest((current) => ({ ...current, pricing: { ...current.pricing, cost_per_million_input: Number(event.target.value) } }))}
                />
              </label>
              <label className="field">
                <span>Tokens de salida</span>
                <input
                  type="number" min="0" step="any" required
                  value={request.pricing.cost_per_million_output}
                  onChange={(event) => updateRequest((current) => ({ ...current, pricing: { ...current.pricing, cost_per_million_output: Number(event.target.value) } }))}
                />
              </label>
            </div>
          </fieldset>

          <div className={`workload-note${estimatedRequests > 30_000_000 ? " workload-warning" : ""}`}>
            Trabajo esperado: <strong>{formatNumber(estimatedRequests)}</strong> solicitudes simuladas
            {estimatedRequests > 30_000_000 && <span> · supera el límite de seguridad de la API</span>}
          </div>
          <button className="primary-button" type="submit" disabled={busy || estimatedRequests > 30_000_000}>
            {busy ? <><span className="button-spinner" /> Simulando...</> : <>Ejecutar simulación <span aria-hidden="true">→</span></>}
          </button>
          <button
            className="secondary-button"
            type="button"
            disabled={busy || estimatedRequests * 1.25 > 30_000_000}
            onClick={() => void runSimulation(true)}
          >
            Comparar con demanda +25%
          </button>
          <p className="form-footnote">La comparación utiliza la misma semilla para contrastar escenarios con parámetros iguales.</p>
        </form>

        <section className="results-column" aria-live="polite">
          {!result ? (
            <div className="panel empty-state">
              <div className="empty-illustration" aria-hidden="true"><span>∿</span><span>∑</span><span>↗</span></div>
              <p className="eyebrow">RESULTADOS DEL EXPERIMENTO</p>
              <h2>Tu análisis empieza aquí</h2>
              <p>Ajusta los parámetros y ejecuta una simulación. Las métricas aparecerán aquí cuando la API responda.</p>
            </div>
          ) : (
            <>
              <div className="panel result-panel">
                <div className="panel-heading result-heading">
                  <div><p className="eyebrow">RESUMEN DE COSTOS · USD / MES</p><h2>Distribución simulada</h2></div>
                  <span className="run-badge"><i /> {formatNumber(result.iterations)} iteraciones</span>
                </div>
                <div className="metrics-grid">
                  <MetricCard label="Costo promedio" value={formatUsd(metrics!.mean)} explanation="Media muestral de los escenarios" />
                  <MetricCard label="Riesgo presupuestario" value={`${(result.probability_exceeding_budget * 100).toFixed(1)}%`} explanation={`Probabilidad de superar ${formatUsd(result.budget_limit)}`} accent />
                  <MetricCard label="Desviación estándar" value={formatUsd(metrics!.sample_std)} explanation="Dispersión muestral (ddof = 1)" />
                  <MetricCard label="Percentil P5" value={formatUsd(metrics!.p5)} explanation="Escenario favorable (5%)" />
                  <MetricCard label="Mediana P50" value={formatUsd(metrics!.p50_median)} explanation="La mitad de escenarios queda debajo" />
                  <MetricCard label="Percentil P95" value={formatUsd(metrics!.p95)} explanation="Escenario de estrés (95%)" />
                </div>
                <div className="chart-section">
                  <div className="chart-title"><div><h3>Escenarios de costo mensual</h3><p>Frecuencia de las simulaciones por rango de costo</p></div><span>USD</span></div>
                  <Histogram result={result} />
                </div>
                <div className="result-meta">
                  <span>Semilla: <strong>{result.seed ?? "aleatoria"}</strong></span>
                  <span>Tokens totales promedio: <strong>{formatNumber(result.token_metrics.mean)}</strong></span>
                  <span>Tiempo de cálculo: <strong>{formatNumber(result.execution_time_ms)} ms</strong></span>
                </div>
              </div>
              {comparison && (
                <div className="panel comparison-panel">
                  <p className="eyebrow">ANÁLISIS DE SENSIBILIDAD</p>
                  <h2>Impacto de un 25% más de demanda</h2>
                  <p className="comparison-caption">Estimación basada en dos ejecuciones reales de la API con idéntica semilla.</p>
                  <div className="comparison-grid">
                    <div><span>Media base</span><strong>{formatUsd(result.cost_metrics.mean)}</strong></div>
                    <div><span>Media con +25%</span><strong>{formatUsd(comparison.cost_metrics.mean)}</strong></div>
                    <div><span>Variación de media</span><strong>{formatUsd(comparison.cost_metrics.mean - result.cost_metrics.mean)}</strong></div>
                    <div><span>P95 con +25%</span><strong>{formatUsd(comparison.cost_metrics.p95)}</strong></div>
                  </div>
                </div>
              )}
            </>
          )}
        </section>
      </div>

      <section className="assumptions">
        <div><p className="eyebrow">CÓMO INTERPRETARLO</p><h2>Una simulación informa decisiones;<br />no elimina la incertidumbre.</h2></div>
        <div className="assumption-items">
          <article><span>01</span><p><strong>Datos sintéticos</strong>La demanda y las longitudes de tokens se generan según las distribuciones elegidas; no son telemetría real.</p></article>
          <article><span>02</span><p><strong>Costo estimado</strong>Se calcula a partir de tokens de entrada/salida, tarifas por millón y costo fijo mensual.</p></article>
          <article><span>03</span><p><strong>Supuestos del modelo</strong>Las solicitudes se tratan como independientes y la demanda mensual usa una distribución paramétrica.</p></article>
          <article><span>04</span><p><strong>Riesgo empírico</strong>La probabilidad de excedencia es una proporción Monte Carlo; su precisión depende del número de iteraciones.</p></article>
        </div>
      </section>

      <footer className="footer">
        <span>AI Monte Carlo Lab <i>·</i> Proyecto académico de Estadística para Ingeniería</span>
        <a href={`${API_BASE_URL}/health`} target="_blank" rel="noreferrer">Estado de la API <span aria-hidden="true">↗</span></a>
      </footer>
    </main>
  );
}
