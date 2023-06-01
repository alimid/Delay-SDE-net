# Delay-SDE-net

This repository contains the Delay-SDE-net together with some run examples. The repository is under construction.

The Delay-SDE-net was introduced in [Eggen and Midtfjord (2023)](https://arxiv.org/pdf/2303.08587.pdf). It is a neural network model based on stochastic delay differential equations (SDDEs), a generalization of the the SDE-net presented in Kong et al. (2020). The Delay-SDE-net it suitable model for time series with memory effects, as it includes memory through previous states of the system. The stochastic part of the Delay-SDE-net provides a basis for estimating uncertainty in modelling, and is split into two neural networks to account for aleatoric and epistemic uncertainty. The uncertainty is provided instantly, making the model suitable
for applications where time is sparse

## References

Eggen, Mari Dahl, and Alise Danielle Midtfjord. "Delay-SDE-net: A deep learning approach for time series modelling with memory and uncertainty estimates." arXiv preprint arXiv:2303.08587 (2023).

Kong, L., Sun, J. & Zhang, C. "SDE-Net: Equipping Deep Neural Networks with Uncertainty Estimates." Proceedings of the 37th International Conference on Machine Learning, in Proceedings of Machine Learning Research (2020). 
