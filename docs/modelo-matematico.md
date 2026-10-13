# Modelo Matemático y Probabilístico: AI Monte Carlo Lab (Fase 1)

## 1. Introducción y Alcance

Este documento describe la formulación matemática, estadística y probabilística de la **Fase 1 de AI Monte Carlo Lab**. El propósito del motor es modelar estocásticamente la demanda de inferencia en modelos de lenguaje de gran escala (LLMs), proyectar el consumo agregado de tokens y cuantificar la incertidumbre financiera mensual mediante simulación de Monte Carlo.

> [!WARNING]
> **Declaración de Datos Sintéticos:**
> Los datos generados por el módulo sintético y las trayectorias de simulación provienen exclusivamente de modelos paramétricos teóricos y generadores pseudoaleatorios (`numpy.random.default_rng`). **No representan tráfico productivo real ni observaciones empíricas de usuarios reales** a menos que los hiperparámetros sean formalmente calibrados mediante registros históricos de telemetría.

---

## 2. Variables Aleatorias y Espacio de Probabilidad

Definimos un espacio de probabilidad $(\Omega, \mathcal{F}, \mathbb{P})$ sobre el cual modelamos un ciclo de facturación mensual compuesto por $D$ días operativos (por defecto $D = 30$).

### 2.1. Demanda Mensual de Solicitudes ($N$)

Sea $N$ la variable aleatoria discreta que representa el número total de solicitudes de inferencia atendidas durante el mes.

1. **Proceso de Poisson Homogéneo:**
   Si las solicitudes diarias $N_d \sim \text{Poisson}(\lambda_{\text{día}})$, por la propiedad reproductiva de la distribución de Poisson:
   $$N = \sum_{d=1}^D N_d \sim \text{Poisson}(\lambda_{\text{mes}}), \quad \text{donde } \lambda_{\text{mes}} = D \cdot \lambda_{\text{día}}$$
   Cumpliéndose $\mathbb{E}[N] = \lambda_{\text{mes}}$ y $\text{Var}(N) = \lambda_{\text{mes}}$.

2. **Proceso Sobredisperso (Binomial Negativa):**
   Para modelar ráfagas o estacionalidad no capturada por Poisson donde $\text{Var}(N) > \mathbb{E}[N]$, se emplea la distribución Binomial Negativa parametrizada por la media $\mu_N = \lambda_{\text{mes}}$ y el parámetro de dispersión $\alpha > 0$:
   $$\text{Var}(N) = \mu_N + \alpha \mu_N^2$$
   Los parámetros canónicos $(r, p)$ se derivan como:
   $$r = \frac{1}{\alpha}, \quad p = \frac{1}{1 + \alpha \mu_N}$$

3. **Demanda Fija Determinista:**
   $N = n_{\text{fijo}}$ constante, útil para pruebas de tensión controladas.

---

### 2.2. Longitud de Tokens por Solicitud ($T_{\text{in}}, T_{\text{out}}$)

Cada solicitud individual $i \in \{1, \dots, N\}$ consume dos clases de tokens:
- $T_{\text{in}, i} \in \mathbb{N}^+$: Tokens de entrada (*prompt* o contexto).
- $T_{\text{out}, i} \in \mathbb{N}^+$: Tokens de salida (*completion* o respuesta).

#### Distribución Log-Normal Discretizada (Modelo Principal)
El número de tokens suele exhibir asimetría positiva pronunciada (*right skewness*) con una masa concentrada en prompts breves y una cola larga en prompts estructurados o conversaciones extensas.

Sea $Y \sim \mathcal{N}(\mu_{\ln}, \sigma_{\ln}^2)$. Definimos $X = \lfloor\exp(Y)\rceil$. Dados los momentos en escala natural deseados $\mathbb{E}[X] = \mu$ y $\text{SD}(X) = \sigma$:
$$\sigma_{\ln}^2 = \ln\left(1 + \frac{\sigma^2}{\mu^2}\right)$$
$$\mu_{\ln} = \ln(\mu) - \frac{1}{2}\sigma_{\ln}^2$$

Para garantizar admisibilidad computacional, los valores muestreados se truncan y redondean:
$$T_i = \max\left(T_{\min}, \min(T_{\max}, \lfloor\exp(Y_i)\rceil)\right)$$

#### Distribuciones Alternativas Soportadas
- **Gamma:** Parametrizada por forma $k = \mu^2 / \sigma^2$ y escala $\theta = \sigma^2 / \mu$.
- **Normal Truncada:** $T_i = \max(T_{\min}, \lfloor\mathcal{N}(\mu, \sigma^2)\rceil)$.
- **Uniforme Discreta:** Sobre el intervalo $[a, b]$.

---

### 2.3. Modelo Estocástico de Latencia ($L$)

La latencia por solicitud $L_i$ (en milisegundos) se formula mediante una combinación afín de tokens procesados más un componente estocástico de fluctuación de red y colas de inferencia:
$$L_i = \max\left(L_{\min}, \, \beta_0 + \beta_{\text{in}} T_{\text{in}, i} + \beta_{\text{out}} T_{\text{out}, i} + \epsilon_i\right)$$
donde:
- $\beta_0$: Tiempo base al primer token (*Time-to-First-Token* fijo de inicialización).
- $\beta_{\text{in}}$: Latencia marginal por token de entrada (ms/token).
- $\beta_{\text{out}}$: Latencia marginal por token generado (ms/token, dominante debido a la naturaleza autorregresiva de la decodificación).
- $\epsilon_i \sim \mathcal{N}(0, \sigma_{\text{jitter}}^2)$: Ruido gaussiano estocástico.
- $L_{\min}$: Límite inferior físico de latencia.

---

## 3. Función de Costo y Sumas Compuestas

### 3.1. Estructura Tarifaria
Sean:
- $R_{\text{in}}$: Tarifa en USD por cada $1\,000\,000$ de tokens de entrada.
- $R_{\text{out}}$: Tarifa en USD por cada $1\,000\,000$ de tokens de salida.
- $C_{\text{fijo}}$: Cargo fijo mensual (USD) por infraestructura o cuota base.

### 3.2. Consumo Agregado y Costo Mensual
En una replicación mensual $m$, el número de solicitudes es $N^{(m)}$. El consumo total de tokens es una **suma aleatoria compuesta**:
$$S_{\text{in}}^{(m)} = \sum_{i=1}^{N^{(m)}} T_{\text{in}, i}^{(m)}, \qquad S_{\text{out}}^{(m)} = \sum_{i=1}^{N^{(m)}} T_{\text{out}, i}^{(m)}$$
$$S_{\text{total}}^{(m)} = S_{\text{in}}^{(m)} + S_{\text{out}}^{(m)}$$

El costo monetario total mensual resultante es:
$$C^{(m)} = \left(\frac{S_{\text{in}}^{(m)}}{10^6}\right) R_{\text{in}} + \left(\frac{S_{\text{out}}^{(m)}}{10^6}\right) R_{\text{out}} + C_{\text{fijo}}$$

---

## 4. Propiedades Teóricas y Convergencia (Identidad de Wald)

Bajo el supuesto de que los tamaños de solicitud $\{T_{\text{in}, i}\}$ y $\{T_{\text{out}, i}\}$ son variables aleatorias independientes e idénticamente distribuidas (i.i.d.) e independientes de $N$:

### 4.1. Esperanza Matemática (Identidad de Wald)
$$\mathbb{E}[S_{\text{in}}] = \mathbb{E}[N] \cdot \mathbb{E}[T_{\text{in}}]$$
$$\mathbb{E}[S_{\text{out}}] = \mathbb{E}[N] \cdot \mathbb{E}[T_{\text{out}}]$$
$$\mathbb{E}[C] = \left(\frac{\mathbb{E}[N]\mathbb{E}[T_{\text{in}}]}{10^6}\right) R_{\text{in}} + \left(\frac{\mathbb{E}[N]\mathbb{E}[T_{\text{out}}]}{10^6}\right) R_{\text{out}} + C_{\text{fijo}}$$

### 4.2. Varianza de la Suma Compuesta (Ley de Varianza Total)
Para la suma de tokens de entrada:
$$\text{Var}(S_{\text{in}}) = \mathbb{E}[N] \cdot \text{Var}(T_{\text{in}}) + (\mathbb{E}[T_{\text{in}}])^2 \cdot \text{Var}(N)$$
Esta ecuación cuantifica formalmente cómo la variabilidad del volumen de solicitudes amplifica la varianza del costo final.

---

## 5. Estimadores Estadísticos de Monte Carlo

A partir de $M$ replicaciones mensuales independientes $\{C^{(1)}, C^{(2)}, \dots, C^{(M)}\}$ generadas con `numpy.random.default_rng(seed)`:

### 5.1. Media Muestral ($\bar{C}$)
Estimador insesgado de la esperanza teórica:
$$\bar{C} = \frac{1}{M} \sum_{m=1}^M C^{(m)}$$

### 5.2. Desviación Estándar Muestral ($s$)
Calculada obligatoriamente con la corrección de Bessel ($ddof = 1$) para eliminar el sesgo:
$$s = \sqrt{\frac{1}{M - 1} \sum_{m=1}^M (C^{(m)} - \bar{C})^2}$$

### 5.3. Percentiles Empíricos ($P_5, P_{50}, P_{95}$)
Sea $C_{(1)} \le C_{(2)} \le \dots \le C_{(M)}$ la muestra ordenada de menor a mayor.
- **$P_5$ (Escenario Favorable):** Valor por debajo del cual ocurre el 5% de las trayectorias.
- **$P_{50}$ (Mediana):** Medida robusta de tendencia central frente a asimetría.
- **$P_{95}$ (Escenario de Estrés / VaR 95%):** Nivel de gasto mensual superado solo en el 5% de los peores escenarios.

Calculados formalmente mediante interpolación lineal continua de rangos muestrales:
$$P_p = \text{Percentile}(C, p), \quad p \in \{5, 50, 95\}$$

### 5.4. Probabilidad Empírica de Exceder el Presupuesto ($\hat{P}(C > B)$)
Dado un límite presupuestario $B \ge 0$:
$$\hat{P}(C > B) = \frac{1}{M} \sum_{m=1}^M \mathbb{I}\left(C^{(m)} > B\right)$$
donde $\mathbb{I}(\cdot)$ es la función indicatriz.

El error estándar de Monte Carlo para esta probabilidad sigue la distribución binomial:
$$\text{SE}(\hat{P}) = \sqrt{\frac{\hat{P}(1 - \hat{P})}{M}}$$
Para $M = 10\,000$ iteraciones, el error estándar máximo ($\hat{P}=0.5$) es $\le 0.005$ ($\pm 0.5\%$).

---

## 6. Supuestos y Limitaciones de la Fase 1

1. **Independencia Condicional:** Se asume que las solicitudes dentro de un mes son mutuamente independientes condicionales al parámetro de demanda mensual.
2. **Ausencia de Prompt Caching:** La Fase 1 no descuenta tokens en caché repetitivos; cada solicitud incurre en facturación completa.
3. **Precios Lineales:** Las tarifas por millón de tokens se modelan como constantes sin escalones de descuento por volumen (tiered pricing).
4. **Reproducibilidad Pseudoaleatoria:** Todas las simulaciones son deterministas condicionadas a una semilla entera provista a `default_rng`.
