# Core Functions for Optimizers

## Prototype

$$
{w_{t+1} = w_t + \Delta w_t}\\
{g_t = \nabla \text{Loss}(w_t)}
$$

## Implementation

### Gradient Descent Optimizers

$${\Delta w_t = - \lambda g_t}$$

### Momentum Optimizers

$${\Delta w_t = \beta \Delta w_{t-1} - \lambda g_t}$$

### Adaptive Learning Rate Optimizers

$$
{\Delta w_t = - \frac{\lambda}{\text{RMS}(g_t)} g_t}\\
{\text{RMS}(g_t) = \sqrt{v_t + \epsilon}}\\
{v_t = \beta v_{t-1} + (1 - \beta) g_t^2}
$$

## Adaptive Moment Estimation (Adam)

$$
{\Delta w_t = - \lambda \frac{\hat{m_t}}{\sqrt{\hat{v_t} + \epsilon}}}\\
{\hat{m_t} = \frac{m_t}{1 - \beta_1^t}}\\
{m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t}\\
{\hat{v_t} = \frac{v_t}{1 - \beta_2^t}}\\
{v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2}
$$
