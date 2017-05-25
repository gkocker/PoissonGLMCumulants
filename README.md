# PoissonGLM_Cumulants

This repository contains code to compute joint cumulants of the spike trains of nonlinear point process GLMs (nonlinear multivariate Hawkes processes) to one loop order. Details are contained in "Linking structure and activity in nonlinear spiking networks" (https://arxiv.org/abs/1610.03828; in press, PloS CB).

That paper presents the rules in the temporal domain. The code here operates in the Fourier domain in order to avoid a number of time integrals. The Feynman rules in the Fourier domain are the same as those in the temporal domain except for the following:
1) Vertices do not carry time variables.
2) Every edge has an associated frequency variable (the argument of the propagator or interaction function). External vertices carry the frequency variable of their input edge, while internal vertices do not carry a frequency variable.
3) The factor for an internal vertex with a outgoing and b incoming edges is multiplied by: $(1/2*pi)^b * 2*pi*\delta(\sum_b \omega_b - \sum_a \omega_a)$. The delta function conserves momentum; the sums over a and b are over the frequencies of all incoming and outgoing edges for the vertex in question. 

As for the time-domain rules, sum over all neuron indices and integrate over all internal frequencies in order to compute the cumulant.
