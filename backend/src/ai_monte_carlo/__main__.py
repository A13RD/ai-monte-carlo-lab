"""CLI entrypoint for running AI Monte Carlo Lab simulation and Phase 2 predictive experiment demo."""

from ai_monte_carlo.config import (
    DistributionType,
    MonteCarloConfig,
    PricingConfig,
    TokenDistributionConfig,
    WorkloadConfig,
    WorkloadDistributionType,
)
from ai_monte_carlo.data.demand import DailyDemandConfig
from ai_monte_carlo.services.experiment import (
    ExperimentConfig,
    run_demand_experiment,
)
from ai_monte_carlo.simulation.monte_carlo import run_monte_carlo


def run_phase_1_demo() -> None:
    print("=" * 75)
    print("AI MONTE CARLO LAB - MOTOR DE SIMULACION (FASE 1)")
    print("=" * 75)

    config = MonteCarloConfig(
        n_iterations=5_000,
        budget_limit=500.0,
        pricing=PricingConfig(
            cost_per_million_input=2.50,
            cost_per_million_output=10.00,
            fixed_monthly_cost=20.00,
        ),
        input_tokens=TokenDistributionConfig(
            family=DistributionType.LOGNORMAL,
            mean=650.0,
            std=350.0,
            min_tokens=10,
        ),
        output_tokens=TokenDistributionConfig(
            family=DistributionType.LOGNORMAL,
            mean=400.0,
            std=200.0,
            min_tokens=10,
        ),
        workload=WorkloadConfig(
            family=WorkloadDistributionType.POISSON,
            mean_daily_requests=1_200.0,
            days_per_month=30,
        ),
        seed=42,
    )

    print("Configuracion cargada:")
    print(f"- Iteraciones Monte Carlo: {config.n_iterations:,}")
    print(f"- Solicitudes promedio/mes: {config.workload.expected_monthly_requests:,.0f}")
    print(f"- Presupuesto limite: ${config.budget_limit:.2f} USD")
    print(f"- Tarifas: ${config.pricing.cost_per_million_input:.2f} / 1M in | ${config.pricing.cost_per_million_output:.2f} / 1M out")
    print("-" * 75)
    print("Ejecutando simulacion...")

    result = run_monte_carlo(config)
    summary = result.summary_dict()

    print("\nResultados Estadisticos (Costo Mensual en USD):")
    print(f"  Media (E[Costo]):         ${summary['cost_usd']['mean']:.2f}")
    print(f"  Desv. Estandar Muestral:  ${summary['cost_usd']['sample_std']:.2f}")
    print(f"  Percentil 5 (P5):         ${summary['cost_usd']['p5']:.2f}")
    print(f"  Percentil 50 (Mediana):   ${summary['cost_usd']['p50_median']:.2f}")
    print(f"  Percentil 95 (P95):       ${summary['cost_usd']['p95']:.2f}")
    print(f"  Minimo observado:         ${summary['cost_usd']['min']:.2f}")
    print(f"  Maximo observado:         ${summary['cost_usd']['max']:.2f}")
    print("-" * 75)
    prob_exceed = summary["probability_exceeding_budget"]
    print(f"Probabilidad de Exceder Presupuesto (${config.budget_limit:.2f} USD):")
    print(f"  P(Costo > Presupuesto):   {prob_exceed * 100:.2f}%")
    print("=" * 75)


def run_phase_2_demo() -> None:
    print("\n" + "=" * 75)
    print("AI MONTE CARLO LAB - EXPERIMENTO INTEGRADO (FASE 2)")
    print("Modelo Predictivo de Demanda + Evaluacion Temporal + Monte Carlo")
    print("=" * 75)

    exp_config = ExperimentConfig(
        demand_config=DailyDemandConfig(
            n_days=180,
            start_date="2026-01-01",
            base_demand=1000.0,
            trend_slope=2.5,
            weekly_seasonality=(1.05, 1.10, 1.12, 1.08, 0.95, 0.70, 0.60),
            noise_dispersion=0.05,
            noise_family="poisson",
            seed=42,
        ),
        monte_carlo_config=MonteCarloConfig(
            n_iterations=5_000,
            budget_limit=250.0,
            seed=42,
        ),
        test_size=0.2,
        forecast_days=30,
        refit_full_history=True,
        model_alpha=1e-4,
        seed=42,
    )

    print("Configuracion del Experimento:")
    print(f"- Historial sintetico: {exp_config.demand_config.n_days} dias desde {exp_config.demand_config.start_date}")
    print(f"- Tendencia base: {exp_config.demand_config.base_demand:.0f} req/dia (+{exp_config.demand_config.trend_slope:.1f} req/dia)")
    print(f"- Particion temporal: 80% Entrenamiento / 20% Prueba (cronologica estricta)")
    print(f"- Horizonte de pronostico: {exp_config.forecast_days} dias")
    print(f"- Iteraciones Monte Carlo: {exp_config.monte_carlo_config.n_iterations:,}")
    print("-" * 75)
    print("Ejecutando pipeline predictivo e integracion con Monte Carlo...")

    res = run_demand_experiment(exp_config)
    summary = res.summary_dict()
    eval_s = summary["evaluation"]
    fore_s = summary["horizon_forecast"]
    sim_s = summary["monte_carlo_simulation"]

    print("\n1. Evaluacion Predictiva (Conjunto de Prueba Fuera de Muestra):")
    print(f"   Modelo Poisson GLM:    MAE = {eval_s['model_mae']:.2f} req  |  RMSE = {eval_s['model_rmse']:.2f} req  |  Deviance = {eval_s['model_deviance']:.4f}")
    print(f"   Referencia Base:       MAE = {eval_s['baseline_mae']:.2f} req  |  RMSE = {eval_s['baseline_rmse']:.2f} req")
    print(f"   Reduccion de Error:    Delta MAE = -{eval_s['mae_reduction']:.2f} req  |  Delta RMSE = -{eval_s['rmse_reduction']:.2f} req")
    print(f"   Supera a la Referencia: {'SI (Mejora estadistica comprobada)' if eval_s['model_beats_baseline'] else 'NO'}")

    print("\n2. Pronostico del Horizonte Mensual (30 Dias):")
    print(f"   Demanda Diaria Promedio Pronosticada: {fore_s['mean_daily_demand']:.2f} req/dia")
    print(f"   Rango Diario Estimado:                 [{fore_s['min_daily_demand']:.1f}, {fore_s['max_daily_demand']:.1f}] req/dia")
    print(f"   Demanda Mensual Total Esperada:        {fore_s['total_expected_demand']:,.0f} req/mes")

    print("\n3. Resultados del Motor Monte Carlo con Demanda Pronosticada:")
    print(f"   Costo Mensual Estimado (Media):       ${sim_s['cost_usd']['mean']:.2f} USD")
    print(f"   Desviacion Estandar Muestral:         ${sim_s['cost_usd']['sample_std']:.2f} USD")
    print(f"   Percentil 5 (P5):                     ${sim_s['cost_usd']['p5']:.2f} USD")
    print(f"   Percentil 50 (Mediana):               ${sim_s['cost_usd']['p50_median']:.2f} USD")
    print(f"   Percentil 95 (P95 - Escenario Estres): ${sim_s['cost_usd']['p95']:.2f} USD")
    print(f"   Probabilidad P(Costo > ${exp_config.monte_carlo_config.budget_limit:.2f}):      {sim_s['probability_exceeding_budget'] * 100:.2f}%")

    print("\n4. Coeficientes del Modelo Poisson Aprendidos:")
    for coef_name, val in res.reproduction_info["learned_coefficients"].items():
        print(f"   - {coef_name:<12}: {val:>10.6f}")

    print("-" * 75)
    print("Nota Metodologica:")
    print(f"  {res.reproduction_info['approximation_note']}")
    print("=" * 75)


def main() -> None:
    run_phase_1_demo()
    run_phase_2_demo()


if __name__ == "__main__":
    main()
