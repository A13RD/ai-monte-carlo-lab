"""CLI entrypoint for running AI Monte Carlo Lab simulation demo."""

from ai_monte_carlo.config import (
    DistributionType,
    MonteCarloConfig,
    PricingConfig,
    TokenDistributionConfig,
    WorkloadConfig,
    WorkloadDistributionType,
)
from ai_monte_carlo.simulation.monte_carlo import run_monte_carlo


def main() -> None:
    print("=" * 70)
    print("AI MONTE CARLO LAB - MOTOR DE SIMULACION (FASE 1)")
    print("=" * 70)

    # 1. Definir configuracion de simulacion
    config = MonteCarloConfig(
        n_iterations=10_000,
        budget_limit=500.0,
        pricing=PricingConfig(
            cost_per_million_input=2.50,   # USD por 1M tokens de entrada
            cost_per_million_output=10.00,  # USD por 1M tokens de salida
            fixed_monthly_cost=20.00,       # Costo fijo mensual de hosting/infra
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
    print("-" * 70)
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
    print("-" * 70)
    prob_exceed = summary["probability_exceeding_budget"]
    print(f"Probabilidad de Exceder Presupuesto (${config.budget_limit:.2f} USD):")
    print(f"  P(Costo > Presupuesto):   {prob_exceed * 100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
