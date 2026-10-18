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


def run_phase_3_demo() -> None:
    print("\n" + "=" * 75)
    print("AI MONTE CARLO LAB - VALIDACION ESTADISTICA AVANZADA (FASE 3)")
    print("Convergencia + Sensibilidad + Correlaciones + Referencia Analitica")
    print("=" * 75)

    from ai_monte_carlo.analysis import (
        analyze_convergence,
        analyze_dataframe_correlations,
        compare_simulation_with_analytical,
        generate_synthetic_requests_with_correlation,
        run_sensitivity_analysis,
    )

    base_config = MonteCarloConfig(
        n_iterations=1_000,
        budget_limit=120.0,
        pricing=PricingConfig(
            cost_per_million_input=2.50,
            cost_per_million_output=10.00,
            fixed_monthly_cost=15.00,
        ),
        input_tokens=TokenDistributionConfig(
            family=DistributionType.LOGNORMAL,
            mean=600.0,
            std=250.0,
            min_tokens=10,
        ),
        output_tokens=TokenDistributionConfig(
            family=DistributionType.LOGNORMAL,
            mean=350.0,
            std=150.0,
            min_tokens=10,
        ),
        workload=WorkloadConfig(
            family=WorkloadDistributionType.POISSON,
            mean_daily_requests=200.0,
            days_per_month=30,
        ),
        seed=42,
    )

    # 1. Ejecutar simulacion base
    base_result = run_monte_carlo(base_config)

    # 2. Componente A: Convergencia
    print("\n1. Analisis de Convergencia Monte Carlo (Checkpoints Adaptativos):")
    conv_res = analyze_convergence(base_result, method_ci="t")
    print(f"   {'Iteraciones':>12} | {'Media Costo':>12} | {'Error Est.':>11} | {'IC 95% Media':>24} | {'P(Costo > B)':>13} | {'IC Wilson 95%':>22}")
    print("   " + "-" * 105)
    for cp in conv_res.checkpoints:
        mean_ci_str = f"[{cp.ci_mean_lower:>9.2f}, {cp.ci_mean_upper:>9.2f}]"
        prob_str = f"{cp.prob_exceed_budget * 100:.2f}%" if cp.prob_exceed_budget is not None else "N/A"
        wilson_str = f"[{cp.ci_prob_lower * 100:.2f}%, {cp.ci_prob_upper * 100:.2f}%]" if cp.ci_prob_lower is not None else "N/A"
        print(f"   {cp.checkpoint_size:>12,d} | ${cp.mean:>11.2f} | ${cp.se_mean:>10.4f} | {mean_ci_str:>24} | {prob_str:>13} | {wilson_str:>22}")

    # 3. Componente B: Sensibilidad (OAT)
    print("\n2. Analisis de Sensibilidad One-At-a-Time (Tornado Ranking):")
    sens_res = run_sensitivity_analysis(base_config, perturbation_fraction=0.20, base_result=base_result)
    print(f"   {'Rank':<5} | {'Parametro':<32} | {'Base':>10} | {'Bajo (-20%)':>12} | {'Alto (+20%)':>12} | {'Rango (USD)':>12} | {'Elasticidad':>11}")
    print("   " + "-" * 108)
    for s in sens_res.parameter_summaries:
        elast_str = f"{s.max_abs_elasticity:.2f}" if s.max_abs_elasticity is not None else "N/A (base=0)"
        print(f"   {s.rank_impact:<5} | {s.parameter_name:<32} | {s.base_value:>10.2f} | ${s.cost_at_low:>11.2f} | ${s.cost_at_high:>11.2f} | ${s.swing_mean_cost:>11.2f} | {elast_str:>11}")

    # 4. Componente C: Correlaciones
    print("\n3. Analisis de Correlaciones en Telemetria Sintetica:")
    synth_df = generate_synthetic_requests_with_correlation(n_requests=2_000, seed=42)
    corr_res = analyze_dataframe_correlations(synth_df)
    print(f"   {'Variable X':<18} vs {'Variable Y':<18} | {'Pearson r':>10} (p-val) | {'Spearman rho':>12} (p-val) | Naturaleza Relacion")
    print("   " + "-" * 95)
    for pr in corr_res.pair_results:
        print(f"   {pr.variable_x:<18} vs {pr.variable_y:<18} | {pr.pearson_r:>10.4f} ({pr.pearson_p_value:.3e}) | {pr.spearman_rho:>12.4f} ({pr.spearman_p_value:.3e}) | {pr.relationship_nature}")

    # 5. Componente D: Validacion Analitica
    print("\n4. Comparacion con Esperanza y Varianza Teorica (Suma Compuesta):")
    comp_res = compare_simulation_with_analytical(base_result)
    theor = comp_res.theoretical_moments
    print(f"   - Esperanza Teorica E[C]:        ${theor.expected_monthly_cost:.4f} USD")
    print(f"   - Media Simulada bar{{C}}:         ${comp_res.simulated_mean_cost:.4f} USD")
    print(f"   - Error Estandar Monte Carlo:    ${comp_res.mc_standard_error:.4f} USD")
    print(f"   - Error Absoluto |bar{{C}} - E[C]|:  ${comp_res.cost_mean_abs_error:.4f} USD ({comp_res.cost_mean_relative_error * 100:.3f}%)")
    print(f"   - Puntuacion Z (z-score):        {comp_res.z_score_mean:.4f} (Umbral de consistencia: +/-{comp_res.z_threshold})")
    print(f"   - Consistencia Estadistica:      {'APROBADA (|z| <= 3.0)' if comp_res.is_mean_consistent else 'DISCREPANCIA DETECTADA'}")
    print(f"   - Desviacion Estandar: Teorica = ${theor.std_monthly_cost:.2f} | Simulada = ${comp_res.simulated_std_cost:.2f} (Error: {comp_res.cost_std_rel_error * 100:.2f}%)")
    print(f"   - Conteo Solicitudes E[N]:       Teorico = {theor.expected_requests:.0f} | Simulado = {comp_res.simulated_mean_requests:.1f}")
    print("=" * 75)


import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Monte Carlo Lab CLI and API Server")
    parser.add_argument("--serve", action="store_true", help="Start the FastAPI HTTP server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address for API server")
    parser.add_argument("--port", type=int, default=8000, help="Port for API server")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn hot reloading")

    args = parser.parse_args()
    if args.serve:
        import uvicorn

        uvicorn.run("ai_monte_carlo.api.app:app", host=args.host, port=args.port, reload=args.reload)
    else:
        run_phase_1_demo()
        run_phase_2_demo()
        run_phase_3_demo()


if __name__ == "__main__":
    main()

