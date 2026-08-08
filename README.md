# MATLAB to Python Shift: Semiconductor Devices

A Python-based implementation of **Semiconductor Devices / Solid State Electronics laboratory experiments**, developed as an exploration of how the same engineering computations can be implemented using the scientific Python ecosystem.

The notebook covers semiconductor statistics, carrier concentrations, BJT and diode characteristics, MOSFET I-V behaviour, and MOS-device electrostatics using **NumPy, SciPy, Matplotlib, and Seaborn**.

> **Note:** This is an independent Python implementation for learning and experimentation. The corresponding academic laboratory work may still require MATLAB as specified by the course.

## Overview

The implementation focuses not only on reproducing the expected semiconductor plots, but also on the **numerical methods behind them**.

Particular emphasis is placed on:

- NumPy array broadcasting and vectorization
- `float64` numerical precision
- Element-wise mathematical evaluation
- Conditional vectorized computation
- Nonlinear root finding with SciPy
- Numerically stable special functions
- Numerical integration
- Scientific visualization

The goal was to treat the experiments as **computational engineering problems**, rather than simply translating MATLAB syntax line-by-line.

---

## Experiments

| # | Experiment | Main Python / NumPy techniques | SciPy methods |
|---|---|---|---|
| 1 | Fermi-Dirac Distribution | Broadcasting, `np.newaxis`, vectorized exponential evaluation | `scipy.special.expit` |
| 2 | Intrinsic Carrier Concentration | Vectorized power/exponential operations, `np.sqrt` | SciPy constants ecosystem |
| 3 | Extrinsic Carrier Concentration | Vectorized semiconductor equations, array-based initial estimates | `scipy.optimize.newton` |
| 4 | n-p-n BJT Characteristics | 2D broadcasting, vectorized device equations | — |
| 5 | Diode I-V & Load Line | Vectorized Shockley equation | `scipy.optimize.fsolve` |
| 6 | Temperature Effects on Diode | Matrix broadcasting across voltage and temperature | `scipy.constants` |
| 7 | n-channel MOSFET Characteristics | `np.where`, `np.maximum`, broadcasting | — |
| 8 | MOS Potential & Charge Density | Vectorized electrostatic equations, array operations | `scipy.integrate.cumulative_trapezoid` |

---

# 1. Fermi-Dirac Distribution

The Fermi-Dirac occupation probability is evaluated as

$$
f(E)=\frac{1}{1+\exp\left(\frac{E-E_F}{kT}\right)}
$$

### Implementation

Instead of evaluating every temperature independently, the energy and temperature arrays are shaped as:

```python
E = np.linspace(0, 1, 500)[:, np.newaxis]
temperatures = np.array([0.1, 100, 300, 500, 800])[np.newaxis, :]
```

This produces compatible `(500, 1)` and `(1, 5)` arrays, allowing NumPy broadcasting to evaluate the complete `500 × 5` parameter space in one operation.

### SciPy

```python
from scipy.special import expit
```

The numerically stable sigmoid implementation

```python
expit((E_F - E) / (k_eV * temperatures))
```

is used instead of directly evaluating the reciprocal exponential expression.

This avoids exponential overflow problems at extremely low temperatures.

---

# 2. Intrinsic Carrier Concentration

For silicon,

$$
n_i = \sqrt{N_C N_V}
\exp\left(-\frac{E_g}{2kT}\right)
$$

The effective density of states is temperature dependent:

$$
N_C,N_V \propto T^{3/2}
$$

### NumPy techniques

The entire temperature sweep is evaluated through vectorized operations:

```python
Nc_T = N_c0 * (T_range / 300.0)**1.5
Nv_T = N_v0 * (T_range / 300.0)**1.5

ni = np.sqrt(Nc_T * Nv_T) * np.exp(...)
```

No explicit numerical loop is required for the calculation.

The resulting carrier concentration is plotted on a logarithmic y-axis using `plt.semilogy()`.

---

# 3. Extrinsic Carrier Concentration

This section goes beyond a simple freeze-out/extrinsic/intrinsic piecewise approximation.

The initial estimate is generated using NumPy:

```python
n_freeze = ...
n_ext_region = np.minimum(n_freeze, N_D)
n_guess = np.maximum(n_ext_region, ni_ext)
```

The charge-neutrality equation is then solved numerically.

### SciPy: Newton-Raphson

```python
from scipy.optimize import newton
```

The notebook supplies both the nonlinear charge-neutrality function and its analytical derivative:

```python
newton(
    charge_neutrality,
    n_guess,
    fprime=charge_neutrality_prime,
    args=(Nc_ext, ni_ext, T_ext)
)
```

This produces a numerical solution across the temperature range instead of relying solely on piecewise approximations.

### Numerical convergence note

The original vectorized Newton implementation reports an occasional:

```text
RuntimeWarning: some failed to converge after 50 iterations
```

for a small subset of temperature points.

This warning is **intentionally not hidden**. It reflects the behaviour of the iterative numerical solver for a highly nonlinear charge-neutrality equation, particularly in the low-temperature region. The warning is therefore documented as part of the numerical behaviour of the implementation rather than suppressed simply to produce a warning-free notebook.

---

# 4. n-p-n BJT Characteristics

The BJT section models both input and output characteristics.

### Input characteristic

The base current is evaluated from an exponential relationship involving $V_{BE}$:

$$
I_B =
\frac{I_S}{\beta}
\left[
\exp\left(\frac{V_{BE}}{V_T}\right)-1
\right]
$$

The complete voltage sweep is calculated using NumPy vectorized operations.

### Output characteristic

Multiple base-current values are evaluated simultaneously.

The voltage and base-current arrays are deliberately shaped as:

```text
VCE : (100, 1)
IB  : (1, 5)
```

allowing NumPy broadcasting to generate a:

```text
(100, 5)
```

current matrix.

The collector current includes the Early effect:

$$
I_C \approx \beta I_B
\left(1+\frac{V_{CE}}{V_A}\right)
$$

A smooth saturation factor is additionally applied to model the low-$V_{CE}$ knee.

---

# 5. Diode I-V & Load Line Analysis

The diode follows the Shockley equation:

$$
I_D =
I_S
\left[
\exp\left(\frac{V_D}{\eta V_T}\right)-1
\right]
$$

The load line is evaluated independently:

$$
I_D=\frac{V_{DD}-V_D}{R}
$$

Both curves are generated using NumPy vectorization.

### Finding the Q-point

The operating point is obtained from the intersection of the diode equation and load line.

The nonlinear equation is passed to:

```python
from scipy.optimize import fsolve
```

and solved numerically:

```python
V_DQ = fsolve(q_point_func, initial_guess)[0]
```

The corresponding diode current is then calculated and displayed directly on the I-V/load-line plot.

---

# 6. Temperature Effects on Diode I-V

This experiment evaluates the diode characteristic at multiple temperatures simultaneously.

Temperature affects both:

- Saturation current $I_S$
- Thermal voltage $V_T$

The voltage array is shaped as:

```text
V_D         → (200, 1)
temperature → (1, 3)
```

NumPy broadcasting therefore produces a complete:

```text
(200, 3)
```

current matrix without manually looping through temperature cases.

SciPy's physical constants are used for:

```python
constants.e
constants.k
```

to calculate the thermal voltage.

---

# 7. n-channel MOSFET I-V Characteristics

The MOSFET model distinguishes between:

### Cutoff

$$
V_{GS}<V_{TH}
$$

### Linear / Triode

$$
V_{DS}<V_{GS}-V_{TH}
$$

### Saturation

$$
V_{DS}\geq V_{GS}-V_{TH}
$$

Instead of writing element-by-element conditional loops, the implementation uses NumPy's vectorized conditional selection:

```python
np.where(...)
```

with the overdrive voltage calculated using:

```python
np.maximum(0, V_GS - V_TH)
```

This allows the complete set of $V_{GS}$-$V_{DS}$ combinations to be evaluated as an array.

The saturation boundary is also calculated directly from the device equations and plotted alongside the I-V curves.

---

# 8. MOS Potential & Charge Density Profiles

The final section moves from lumped device equations toward a spatial electrostatic model of an n-MOS structure.

The implementation evaluates:

- Surface potential
- Electric field
- Physical depth
- Ionized depletion charge density
- Mobile inversion charge density

The relationship

$$
dx=-\frac{d\phi}{E(\phi)}
$$

is numerically integrated to map potential to physical depth.

### SciPy: Numerical Integration

```python
from scipy.integrate import cumulative_trapezoid
```

is used to perform the cumulative numerical integration:

```python
x = cumulative_trapezoid(
    dx_dphi,
    phi,
    initial=0.0
)
```

This converts the potential-space calculation into a physical depth profile.

The resulting depletion and inversion charge densities are plotted on a logarithmic scale to make their spatial variation visible.

---

# Numerical Computing Techniques Used

The notebook deliberately uses several techniques that are useful beyond semiconductor modelling.

### NumPy broadcasting

Instead of repeatedly calculating scalar values:

```python
for T in temperatures:
    for V in voltages:
        ...
```

parameter spaces are represented as compatible multidimensional arrays.

For example:

```text
Voltage      (200, 1)
Temperature  (1, 3)
```

automatically produces:

```text
200 × 3
```

evaluations through broadcasting.

### Vectorized conditionals

Device operating regions are handled using:

```python
np.where()
np.maximum()
np.minimum()
```

rather than element-wise Python control flow.

### Numerical root finding

Two different nonlinear problems use two different SciPy solvers:

- **Newton-Raphson:** extrinsic carrier concentration
- **fsolve:** diode operating-point calculation

### Numerically stable special functions

`scipy.special.expit` is used where direct exponential evaluation could overflow.

### Numerical integration

`scipy.integrate.cumulative_trapezoid` converts the electric-field/potential relationship into a spatial depth profile.

---

# Libraries

```text
Python
├── NumPy
├── SciPy
│   ├── scipy.constants
│   ├── scipy.optimize
│   ├── scipy.special
│   └── scipy.integrate
├── Matplotlib
└── Seaborn
```

## Installation

```bash
pip install numpy scipy matplotlib seaborn jupyter
```

## Running the notebook

Clone the repository:

```bash
git clone https://github.com/AmanBanik/MatPy-nb.git
cd MatPy-nb
```

Launch Jupyter:

```bash
jupyter notebook
```

Then open the `.ipynb` notebook and execute the cells sequentially.

---

# Repository Structure

```text
.
├── semiconductor_devices_python.ipynb
├── README.md
└── requirements.txt
```

Optional `requirements.txt`:

```text
numpy
scipy
matplotlib
seaborn
jupyter
```

---

# Why Python?

The objective was not simply to replace MATLAB syntax with Python syntax.

The implementation explores how semiconductor-device calculations can be expressed through the **scientific Python stack**, particularly:

> **NumPy for vectorized numerical computation → SciPy for numerical methods → Matplotlib/Seaborn for visualization.**

The result is a notebook where the semiconductor equations and the computational methods are both part of the experiment.

---

## Topics Covered

`Semiconductor Physics` · `Fermi-Dirac Statistics` · `Carrier Concentration` · `BJT` · `Diodes` · `Load Line Analysis` · `MOSFETs` · `MOS Electrostatics` · `Numerical Methods` · `NumPy Broadcasting` · `Scientific Python` · `SciPy`

---

## Disclaimer

This repository is an educational implementation created to explore scientific computing techniques through Semiconductor Devices experiments. Device models are simplified analytical models intended for computational demonstration and should not be interpreted as full physical semiconductor-device simulations.
