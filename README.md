# Phase-Space Reconstruction Toolkit

Note: Only this README was written with AI (Claude Opus 5.5) from a sample text I wrote.

Python tool for reconstructing the phase space of a dynamical system from a single scalar time series using **time-delay embedding** (Taken's Theorem). The modules estimate the two parameters the reconstruction needs, the **embedding delay τ** and the **embedding dimension d**, and include the chaotic Lorenz system as a test signal. The code function takes a plain time series as input.


| Parameter | What it controls | Method | Module |
|-----------|------------------|--------|--------|
| **τ** (delay) | Spacing between the delayed coordinates | Average Mutual Information (Fraser & Swinney, 1986) | `ami.py` |
| **d** (dimension) | Number of delayed coordinates | False Nearest Neighbours (Kennel et al., 1992) | `fnn.py` |
| — | Test signal | Lorenz system (Lorenz, 1963) | `lorenz.py` |

---

## Background

### Why nonlinear dynamics?

Techniques such as electroencephalography (EEG) capture the macroscopic dynamics of the brain's electrical activity as voltage differences between electrodes. A relatively recent approach to analysing these signals comes from nonlinear dynamical systems theory (chaos theory): the signal is treated as the output of a system whose time evolution is defined in some phase space. Because nonlinear systems can exhibit deterministic chaos, this is a natural starting point when a signal looks irregular (Stam, 2005).

Once the phase space has been reconstructed, it can be characterised with invariants such as its fractal (correlation) dimension or its maximal Lyapunov exponent, which quantifies how unpredictable the dynamics are due to sensitivity to initial conditions. In the original project the goal was to use these tools to characterise the dynamics underlying epileptic seizures.

### The measurement problem

A recorded signal is a scalar measurement: a projection of the unobserved internal variables of the original system onto the real line. Besides reducing dimensionality, this projection usually mixes several internal variables together. Recovering the *original* phase space exactly is generally impossible, but it is also unnecessary. It is enough to build a new space in which the attractor is **topologically equivalent** to the original one, preserving its essential properties. Invariants such as the Lyapunov exponent and the fractal dimension computed on that reconstruction are the same as for the original system (Kantz & Schreiber, 2004).

This works because, in a deterministic system, all variables are generally coupled: each one implicitly carries information about the others.

### Delay embedding

Let {x₀, x₁, …, xₙ} be the measured series. At time *t* we only know xₜ, but later measurements xₜ₊τ, xₜ₊₂τ, … keep collecting information not only about the observed variable but also about the hidden ones. If τ is chosen well, these delayed values can stand in for the missing variables.

Takens' **delay embedding theorem** (Takens, 1981) formalises this: for a sufficiently large embedding dimension *d*, the delay vectors

$$
\mathbf{p}(t) = \left(x_t,\; x_{t+\tau},\; x_{t+2\tau},\; \dots,\; x_{t+(d-1)\tau}\right)
$$

define a space with exactly the same invariant properties as the original system. The practical question is then how to choose τ and *d*.

---

## Choosing the delay τ: Average Mutual Information (`ami.py`)

A good delay has to satisfy two criteria:

1. **Large enough** that xₜ₊τ carries information that is relevant and noticeably different from what xₜ already tells us.
2. **Not larger** than the typical time over which the system forgets its initial state. Otherwise the reconstructed space consists of uncorrelated points and looks essentially random.

A common rule of thumb takes the first zero crossing of the autocorrelation function. That works for regular signals but can mislead for chaotic ones, because autocorrelation only captures *linear* dependence. Fraser and Swinney (1986) proposed using the **first minimum of the mutual information** between xₜ and xₜ₊τ instead, which also captures nonlinear dependence.

The mutual information between two variables measures how much knowing one tells you about the other:

$$
I(X,Y) = \iint p(x,y)\, \log \frac{p(x,y)}{p(x)\,p(y)} \, dx\, dy
$$

It is zero when X and Y are independent.

**Estimator.** The series is compared with a delayed copy of itself, both of length *n − m* (where *n* is the series length and *m* the largest delay considered):

$$
X = (x_1, \dots, x_{n-m}), \qquad X_\tau = (x_{1+\tau}, \dots, x_{n-m+\tau})
$$

The ranges of X and X_τ are each split into N equal bins. Counting how many values fall into each bin (and each 2-D bin for the joint distribution) and dividing by the number of observations gives the probabilities P_X, P_Xτ and P_XXτ. The estimated mutual information for delay τ is

$$
\hat{I}(\tau) = \sum_{i=1}^{N}\sum_{j=1}^{N} P_{XX_\tau}(i,j)\, \ln \frac{P_{XX_\tau}(i,j)}{P_X(i)\,P_{X_\tau}(j)}
$$

This is computed for every delay up to *m*, and the optimal delay is the **first local minimum** of Î(τ). The implementation uses `numpy.histogram` and `numpy.histogram2d`, and uses the natural logarithm, so values are in nats.

### API

```python
Ixy, tau = AMI(X, m, nb=16, plot=False, verbose=False, initial_verbose=True)
```

| Argument | Description |
|----------|-------------|
| `X` | Scalar time series |
| `m` | Largest delay to consider (at least 3) |
| `nb` | Number of histogram bins (default 16) |
| `plot` | Plot mutual information vs. delay |
| `verbose` | Print the optimal delay when finished |
| `initial_verbose` | Print a message when the computation starts |

**Returns:** `Ixy`, an array with the mutual information (nats) for each delay `0 … m-1`, and `tau`, the optimal delay.

**Warning message:** `No first minimum in the delay specified` means the curve kept decreasing up to `m`. The code continues with a delay of `m - 2`; increase `m` and run it again.

---

## Choosing the dimension d: False Nearest Neighbours (`fnn.py`)

Takens' theorem guarantees a faithful embedding for every sufficiently large *d*, so the goal is to find the **smallest** dimension that fully unfolds the attractor. The method rests on the assumption that the attractor of a deterministic system folds and unfolds smoothly, without sudden irregularities.

Attractors are usually compact, so points on an orbit have close neighbours. If the embedding dimension is too small, some of those neighbours are only close because the attractor has been projected into too few dimensions. These are **false neighbours**: they move far apart as soon as another dimension is added. Once the dimension is large enough, every neighbour is a true neighbour.

The Hénon map is a classic illustration. Its attractor is two-dimensional. Projected onto the x-axis alone, points from different branches of the attractor can sit right next to each other; plotted in the 2-D space (xₙ, xₙ₊₁), those same points turn out to be far apart, while points that are neighbours on the same branch stay close.

**Kennel's criterion.** In dimension *d*, the squared Euclidean distance between a point **p**(t) and its nearest neighbour **p**⁽ʳ⁾(t) is

$$
R_d^2(t) = \sum_{k=0}^{d-1} \left[x_{t+k\tau} - x^{(r)}_{t+k\tau}\right]^2
$$

Going to dimension *d + 1* adds the coordinate xₜ₊d τ, so

$$
R_{d+1}^2(t) = R_d^2(t) + \left[x_{t+d\tau} - x^{(r)}_{t+d\tau}\right]^2
$$

A neighbour is declared **false** when the relative increase in distance is large:

$$
\frac{\left|x_{t+d\tau} - x^{(r)}_{t+d\tau}\right|}{R_d(t)} > R_{tol}
$$

Kennel et al. found R_tol = 10 to work well for most data sets (there is no formal proof), and that value is used here. If results look poor, it is worth repeating the calculation with other thresholds. Nearest neighbours are found with `scipy.spatial.cKDTree`. The chosen dimension is the first one whose fraction of false neighbours falls to `thr` or below.

### API

```python
R, dim = false_nearest_neighbors(x, tau, D, thr, plot=False, verbose=False, initial_verbose=True)
```

| Argument | Description |
|----------|-------------|
| `x` | Scalar time series |
| `tau` | Embedding delay (e.g. from `AMI`) |
| `D` | Largest embedding dimension to test |
| `thr` | Fraction of false neighbours considered negligible (0–1; 0.01 recommended) |
| `plot` | Plot the fraction of false neighbours vs. dimension |
| `verbose` | Print progress and the chosen dimension |
| `initial_verbose` | Print a message when the computation starts |

**Returns:** `R`, the fraction of false neighbours for each dimension `1 … D`, and `dim`, the embedding dimension for the threshold `thr`.

**Warning message:** `Dimension does not converge with threshold …` means the fraction never dropped below `thr`. The code returns `D`; try a larger `D` or a larger `thr`.

---

## Test signal: Lorenz system (`lorenz.py`)

```python
xs = Lorenz(n, plot=False)
```

Integrates the Lorenz equations (σ = 10, ρ = 28, β = 2.667 ≈ 8/3) from (9, 9, 25) with a forward-Euler step of `dt = 0.01` for `n` steps, and returns the x-component (length `n + 1`). Set `plot=True` to draw the 3-D attractor.

---

## Installation

Requires Python 3.6+.

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

## Quick start

```python
import matplotlib.pyplot as plt
from lorenz import Lorenz
from ami import AMI
from fnn import false_nearest_neighbors

X = Lorenz(1000)                                   # scalar time series

Ixy, tau = AMI(X, m=100, nb=16, plot=True)          # 1. choose the delay
R, dim = false_nearest_neighbors(X, tau, D=10,      # 2. choose the dimension
                                 thr=0.01, plot=True)

print(f"tau = {tau}, embedding dimension = {dim}")
plt.show()                                          # display the FNN plot
```

Note that `AMI(..., plot=True)` opens its figure immediately, while `false_nearest_neighbors(..., plot=True)` only creates the figure, so call `plt.show()` to display it.

Each module also runs a self-contained demo:

```bash
python ami.py   # AMI curve for the Lorenz signal
python fnn.py   # FNN curve plus 2-D and 3-D reconstructions of the attractor
```

## Example results

**Lorenz system.** With 1,000 samples, AMI gives τ = 16 samples and FNN gives **d = 3**, matching the true dimension of the Lorenz system.

**Rat EEG.** In the original project the tools were applied to EEG recorded from rats with pilocarpine-induced chronic temporal-lobe epilepsy (two temporal-lobe EEG electrodes, two occipital reference electrodes, and neck EMG), sampled at 15,000 samples per minute (250 Hz). Processing the signal in one-minute windows, the delay was typically τ ≈ 25–33 samples and the embedding dimension usually d = 6, occasionally 5 or 7 (with `thr = 0.01`, `D = 10`, `m = 100`). The recordings themselves are not included in this repository.

## Parameter tips

- **`m`:** choose it comfortably larger than the expected minimum. For the EEG data, `m = 100` was sufficient.
- **`nb`:** 16 bins is a sensible default. Too many bins on a short series gives noisy probability estimates.
- **`D`:** 10 is usually plenty; computation time grows with `D`.
- **`thr`:** 0.01 (1 % false neighbours) is a common choice. Noisy real-world data may never reach it.
- **Window length:** results are only meaningful when the window contains many orbits of the system. Very short windows make both estimates unstable.

## Limitations

- The Lorenz integrator uses forward Euler, which is fine for generating a demo signal but not for accurate simulation. Use `scipy.integrate.solve_ivp` if accuracy matters.
- `fnn.py` implements Kennel's first criterion only, with R_tol fixed at 10. The second criterion (distance relative to the size of the attractor), which helps with noisy data, is not included.
- Measurement noise inflates the apparent embedding dimension. Interpret results on real recordings with care.

## Related analyses (not included)

The original project also estimated the **maximal Lyapunov exponent** (by tracking how the average distance between initially close points grows over time in the reconstructed space) and the **correlation dimension D₂** (from the slope of the linear part of log C(r) vs. log r, following Grassberger & Procaccia, 1983). On the EEG data these gave an MLE of about 0.14–0.20 and D₂ of about 4.2–5.1 per minute. Those modules are not part of this repository.

## Acknowledgements

Developed during an internship at the Centro Interdisciplinario de Neurociencia de Valparaíso (CINV), Chile. The EEG recordings used in the original study were made by Carola Mantellero Gutiérrez (Universidad de Santiago de Chile) as part of her doctoral thesis.

## References

- Fraser, A. M., & Swinney, H. L. (1986). Independent coordinates for strange attractors from mutual information. *Physical Review A*, 33(2), 1134.
- Grassberger, P., & Procaccia, I. (1983). Characterization of strange attractors. *Physical Review Letters*, 50(5), 346.
- Hénon, M. (1976). A two-dimensional mapping with a strange attractor. *Communications in Mathematical Physics*, 50, 69–77.
- Kantz, H., & Schreiber, T. (2004). *Nonlinear Time Series Analysis*. Cambridge University Press.
- Kennel, M. B., Brown, R., & Abarbanel, H. D. I. (1992). Determining embedding dimension for phase-space reconstruction using a geometrical construction. *Physical Review A*, 45(6), 3403.
- Lorenz, E. N. (1963). Deterministic nonperiodic flow. *Journal of the Atmospheric Sciences*, 20(2), 130–141.
- Perc, M. (2006). Introducing nonlinear time series analysis in undergraduate courses. *Fizika A*, 15(1), 91.
- Stam, C. J. (2005). Nonlinear dynamical analysis of EEG and MEG: review of an emerging field. *Clinical Neurophysiology*, 116(10), 2266–2301.
- Takens, F. (1981). Detecting strange attractors in turbulence. In *Dynamical Systems and Turbulence, Warwick 1980* (pp. 366–381). Springer.
- Thomas, R. D., et al. (2014). An efficient algorithm for the computation of average mutual information: Validation and implementation in Matlab. *Journal of Mathematical Psychology*, 61, 45–59.

## License

This project is released under the terms of the license in the [LICENSE](LICENSE) file.

