# PoissonGLM_Cumulants

This repository contains code to compute joint cumulants of the spike trains of nonlinear point process GLMs (nonlinear multivariate Hawkes processes, with inhibition) to one-loop order in a fluctuation expansion. Introductions, details and derivations are contained in ["Linking structure and activity in nonlinear spiking networks"](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005583).

That paper presents the rules for constructing and interpreting Feynman diagrams corresponding to any joint spike train cumulant function.

The code here operates in the temporal Fourier domain in order to avoid a number of time integrals. The Feynman rules in the Fourier domain are the same as those in the temporal domain (see the paper) except for the following:
1) Vertices do not carry time variables.
2) Every edge has an associated frequency variable (the argument of the propagator or interaction function). External vertices carry the frequency variable of their input edge, while internal vertices do not carry a frequency variable.
3) The factor for an internal vertex with a outgoing and b incoming edges is multiplied by: $(1/(2*\pi))^b * 2*\pi*\delta(\sum_b \omega_b - \sum_a \omega_a)$. The delta function conserves momentum; the sums over a and b are over the frequencies of all incoming and outgoing edges for the vertex in question. 

As for the time-domain rules, sum over all neuron indices and integrate over all internal frequencies in order to compute the cumulant.

The file PlotFig_instab.py is a script to reproduce panels from Fig. 14 of the paper; PlotFig_instab_quadratic.py reproduces panels from Fig. 15. PlotFig_instab_exponential.py (coming soon) reproduces panels from Fig. 18. Before running them, make sure that the correct transfer function is uncommented in phi.py, and make sure that the baseline drive ("b" in params.py) is set correctly. Also set the save directory in the PlotFig script. 
