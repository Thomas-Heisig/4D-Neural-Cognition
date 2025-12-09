# Mathematical Model Description

This document provides detailed mathematical descriptions of the models and algorithms used in the 4D Neural Cognition project.

## Table of Contents
- [Neuron Models](#neuron-models)
- [Synaptic Transmission](#synaptic-transmission)
- [Plasticity Rules](#plasticity-rules)
- [Network Dynamics](#network-dynamics)

---

## Neuron Models

### Leaky Integrate-and-Fire (LIF) Model

The LIF model is a simplified neuron model that captures the essential dynamics of neural computation.

**Membrane Potential Dynamics:**
```
dV/dt = (-(V - V_rest) + I_syn + I_ext) / τ_m
```

Where:
- `V`: Membrane potential (mV)
- `V_rest`: Resting potential (default: -65 mV)
- `I_syn`: Synaptic input current
- `I_ext`: External input current
- `τ_m`: Membrane time constant (default: 10 ms)

**Spike Generation:**
```
If V ≥ V_threshold:
    - Emit spike
    - Reset: V ← V_reset
    - Enter refractory period
```

**Parameters:**
- `V_threshold`: Spike threshold (default: -50 mV)
- `V_reset`: Reset potential (default: -65 mV)
- Refractory period: 2 ms

### Izhikevich Model

The Izhikevich model can reproduce various neuron types with just two equations.

**Dynamics:**
```
dv/dt = 0.04v² + 5v + 140 - u + I
du/dt = a(bv - u)
```

**Spike Reset:**
```
If v ≥ 30 mV:
    v ← c
    u ← u + d
```

**Parameters by Neuron Type:**

| Type | a | b | c | d | Behavior |
|------|---|---|---|---|----------|
| Regular Spiking | 0.02 | 0.2 | -65 | 8 | Adapting excitatory |
| Fast Spiking | 0.1 | 0.2 | -65 | 2 | Non-adapting inhibitory |
| Bursting | 0.02 | 0.2 | -55 | 4 | Burst firing |

### Hodgkin-Huxley Model

The HH model is a biophysically realistic model of action potential generation.

**Membrane Potential:**
```
C_m dV/dt = I - I_Na - I_K - I_L
```

**Ionic Currents:**
```
I_Na = g_Na m³h (V - E_Na)    (Sodium current)
I_K  = g_K n⁴ (V - E_K)       (Potassium current)
I_L  = g_L (V - E_L)          (Leak current)
```

**Gating Variables:**
```
dm/dt = α_m(1 - m) - β_m m
dh/dt = α_h(1 - h) - β_h h
dn/dt = α_n(1 - n) - β_n n
```

**Rate Functions:**
```
α_m = 0.1(V + 40) / (1 - exp(-(V + 40)/10))
β_m = 4 exp(-(V + 65)/18)

α_h = 0.07 exp(-(V + 65)/20)
β_h = 1 / (1 + exp(-(V + 35)/10))

α_n = 0.01(V + 55) / (1 - exp(-(V + 55)/10))
β_n = 0.125 exp(-(V + 65)/80)
```

**Default Parameters:**
- `C_m = 1.0 μF/cm²` (membrane capacitance)
- `g_Na = 120 mS/cm²` (max sodium conductance)
- `g_K = 36 mS/cm²` (max potassium conductance)
- `g_L = 0.3 mS/cm²` (leak conductance)
- `E_Na = 50 mV` (sodium reversal potential)
- `E_K = -77 mV` (potassium reversal potential)
- `E_L = -54.4 mV` (leak reversal potential)

---

## Synaptic Transmission

### Discrete Spike-Time Synaptic Model

Synaptic input is calculated as a weighted sum of presynaptic spikes:

```
I_syn(t) = Σ w_ij δ(t - t_j - τ_d)
```

Where:
- `w_ij`: Weight of synapse from neuron j to neuron i
- `t_j`: Spike time of presynaptic neuron j
- `τ_d`: Synaptic delay (default: 1 ms)
- `δ`: Dirac delta function (spike occurs)

### Excitatory vs Inhibitory Synapses

**Excitatory (Glutamatergic):**
```
w > 0, contributes positive current
```

**Inhibitory (GABAergic):**
```
w < 0, contributes negative current
Effective weight: -|w|
```

---

## Plasticity Rules

### Hebbian Learning

Classic correlation-based learning: "Neurons that fire together, wire together."

**Weight Update:**
```
Δw = η * correlation(pre, post)
```

**Rules:**
1. Both fire together (LTP): `Δw = +η`
2. Pre fires, post doesn't (LTD): `Δw = -0.5η`
3. No pre-synaptic activity: `Δw = 0`

**Constraints:**
```
w ∈ [w_min, w_max]
Default: w_min = 0.0, w_max = 1.0
```

### Spike-Timing-Dependent Plasticity (STDP)

Time-dependent learning rule based on spike order.

**Weight Change:**
```
If Δt = t_post - t_pre > 0 (post after pre):
    Δw = A+ exp(-Δt/τ+)    [LTP]
    
If Δt < 0 (pre after post):
    Δw = -A- exp(Δt/τ-)    [LTD]
```

**Parameters:**
- `A+ = 0.01`: LTP amplitude
- `A- = 0.012`: LTD amplitude (slightly larger for balance)
- `τ+ = 20 ms`: LTP time constant
- `τ- = 20 ms`: LTD time constant

### Weight Decay

Synaptic strength naturally decays over time:

```
dw/dt = -λw
w(t+1) = w(t)(1 - λΔt)
```

Where:
- `λ`: Decay rate (default: 0.001)
- Prevents unbounded weight growth

### Homeostatic Plasticity

Maintains stable firing rates through synaptic scaling:

```
w_scaled = w * (r_target / r_actual)^α
```

Where:
- `r_target`: Target firing rate (e.g., 10 Hz)
- `r_actual`: Current firing rate
- `α`: Scaling strength (default: 0.1)

### Metaplasticity (BCM Rule)

Sliding threshold for LTP/LTD transition:

```
τ_θ dθ/dt = r² - θ

Δw = η * r * (r - θ)
```

Where:
- `θ`: Modification threshold
- `r`: Postsynaptic firing rate
- `τ_θ`: Threshold time constant

---

## Network Dynamics

### Population Activity

Mean field approximation for population firing rate:

```
r̄(t) = (1/N) Σ_i s_i(t)
```

Where:
- `N`: Number of neurons
- `s_i(t)`: Spike indicator (1 if spike, 0 otherwise)

### Excitation-Inhibition Balance

Network stability requires balanced E-I ratio:

```
E/I ratio = I_exc / |I_inh|
Optimal range: 1.0 - 4.0
```

**Balance Metrics:**
```
E-I balance = (E - I) / (E + I)
Range: [-1, 1]
    -1: Fully inhibitory
     0: Balanced
    +1: Fully excitatory
```

### Connectivity

**4D Euclidean Distance:**
```
d(i,j) = √[(x_i - x_j)² + (y_i - y_j)² + (z_i - z_j)² + (w_i - w_j)²]
```

**Distance-Dependent Connection Probability:**
```
P(connection) = P_0 exp(-d/λ)
```

Where:
- `P_0`: Base connection probability
- `λ`: Connection length constant

### Network Oscillations

Population oscillations emerge from E-I interactions:

```
Frequency ≈ 1/(τ_E + τ_I)
```

Where:
- `τ_E`: Excitatory time constant
- `τ_I`: Inhibitory time constant

Typical ranges:
- **Gamma (30-80 Hz)**: Fast inhibition
- **Beta (15-30 Hz)**: Motor control
- **Alpha (8-12 Hz)**: Resting state
- **Theta (4-8 Hz)**: Memory encoding

---

## Numerical Integration

### Euler Method (Default)

Simple forward integration:

```
x(t + Δt) = x(t) + Δt * dx/dt
```

**Time step guidelines:**
- LIF: Δt = 1.0 ms
- Izhikevich: Δt = 1.0 ms
- Hodgkin-Huxley: Δt = 0.01 - 0.1 ms (smaller for accuracy)

### Stability Conditions

For stable integration:

```
Δt < 2τ_min / λ_max
```

Where:
- `τ_min`: Smallest time constant in system
- `λ_max`: Largest eigenvalue of system matrix

---

## Statistical Analysis

### Spike Train Correlation

Cross-correlation between spike trains i and j:

```
C_ij(τ) = ⟨s_i(t) s_j(t + τ)⟩_t
```

Normalized:
```
ρ_ij(τ) = C_ij(τ) / √[C_ii(0) C_jj(0)]
```

### Inter-Spike Interval (ISI)

Distribution of time intervals between consecutive spikes:

```
ISI_k = t_k - t_{k-1}
```

Metrics:
- Mean ISI: `⟨ISI⟩`
- CV (coefficient of variation): `σ_ISI / μ_ISI`
- CV = 0: Regular firing
- CV = 1: Poisson process
- CV > 1: Bursty firing

### Fano Factor

Measure of spike count variability:

```
F = Var(n) / Mean(n)
```

Where `n` is spike count in time window.

---

## References

1. Hodgkin & Huxley (1952): "A quantitative description of membrane current"
2. Izhikevich (2003): "Simple model of spiking neurons"
3. Dayan & Abbott (2001): "Theoretical Neuroscience"
4. Gerstner et al. (2014): "Neuronal Dynamics"

---

*Last Updated: December 2025*
*For implementation details, see source code in `src/neuron_models.py` and `src/plasticity.py`*
