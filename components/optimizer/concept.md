# Optimizers

## Gradient Descent Optimizers

1. Categories:
   1. Gradient Descent (GD)
      1. Equations:
         - ${w_{t+1} = w_t - \lambda g_t}$
      2. Cons:
         1. It's inefficient/expensive to compute all data in each iteration.
   2. Stochastic Gradient Descent (SGD)
      1. Definition: Derive the gradients from a single data example.
      2. Pros:
         1. It's efficient to compute one data example in each iteration.
      3. Cons:
         1. It's noisy and may not converge.
   3. Mini-Batch Gradient Descent (MB-GD)
      1. Definition: Derive the gradients from a mini-batch of data examples.
      2. Pros:
         1. It's efficient to compute a mini-batch of data examples in each iteration. Also, it prevents frequent fluctuations in optimization.
2. Cons:
   1. Slow convergence, especially in gradual slopes.
   2. Stochasticity in SGD.

## Momentum Optimizers

1. Momentum Optimizer
   1. Equations:
      - ${w_{t+1} = w_t + m_t}$
      - ${m_t = \beta m_{t-1} - \lambda g_t}$
2. Nesterov Accelerated Gradient (NAG) / Nesterove Momentum Optimizer
   1. Equations:
      - ${w_{t+1} = w_t + m_t}$
      - ${m_t = \beta m_{t-1} - \lambda \nabla_{w_t}f(w_t + \beta m_{t-1})}$

## Adaptive Learning Rate Optimizers

1. AdaGrad Optimizer
   1. Equations:
      - ${w_{t+1} = w_t - \frac{\lambda}{\sqrt{v_t + \epsilon}} g_t}$
      - ${v_t = v_{t-1} + g_t^2}$
2. RMSProp Optimizer (Root Mean Squared Propagation)
   1. Equations:
      - ${w_{t+1} = w_t + \Delta w_t}$
      - ${\Delta w_t = - \frac{\lambda}{\text{RMS}(g_t)}g_t}$
      - ${\text{RMS}(g_t) = \sqrt{v_t + \epsilon}}$
      - ${vt = \beta v_{t-1} + (1 - \beta) g_t^2}$
3. AdaDelta Optimizer
   1. Equations:
      - ${w_{t+1} = w_t + \Delta w_t}$
      - ${\Delta w_t = - \frac{\text{RMS}(\Delta w_{t-1})}{\text{RMS}(g_t)}g_t}$
      - ${\text{RMS}(\Delta w_{t-1}) = \sqrt{u_t + \epsilon}}$
      - ${u_t = \beta u_{t-1} + (1 - \beta) \Delta w_{t-1}^2}$
      - ${\text{RMS}(g_t) = \sqrt{v_t + \epsilon}}$
      - ${v_t = \beta v_{t-1} + (1 - \beta) g_t^2}$

## Adaptive Moment Estimation Optimizers

1. Adam Optimizer (Adaptive Moment Estimation)
   1. Equations:
      $$
      {w_{t+1} = w_t + \Delta w_t}\\
      {w_t = \lambda \frac{\hat{m_t}}{\sqrt{\hat{v_t} + \epsilon}}}\\
      {\hat{m_t} = \frac{m_t}{1 - \beta_1^t}}\\
      {m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t}\\
      {\hat{v_t} = \frac{v_t}{1 - \beta_2^t}}\\
      {v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2}
      $$

## Lookahead Optimizers
