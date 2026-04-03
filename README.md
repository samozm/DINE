# Dynamic Co-Expression Network Estimation (DCENt) 

## Installation

To install the DCENt R package, use the `install_github` command from the `devtools` package.

```devtools::install_github("samozm/DCENt")```

## Use
The algorithm has 6 required inputs. They are as follows

- X: the fixed effect matrix. This should include an intercept, timepoints, and factors for each OTU. The order of rows of X should be all times for node 1 for subject 1, all times for node 2 for subect 1, and so on.
- y: response vector (node expression). This must be in the same order as the X matrix
- Z: a list of matrices. Each element in the list is the random effect matrix for one subject. Each matrix should have the same general form. Denoting $T_i$ the number of timepoints for subject $i$ and $[\mathbf{T_i}] \coloneqq [1,...,T_i]^\top$, then 
$Z = \begin{bmatrix}\textbf{1}_{T_1 \times 1} & \mathbf{[T_1]} & & \textbf{0}\\ & \ddots & \ddots \\ \textbf{0} & & \textbf{1}_{T_n \times 1} & \mathbf{[T_n]}
\end{bmatrix}$
- n0: total number of subjects
- k0: total number of nodes
- t0: total number of timepoints
- ALGORITHM: algorithm number (1 or 2)

Then the estimator can be run as
```DCENt::estimate(X,y,Z,n0,k0,t0,ALGORITHM)```
