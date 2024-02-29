# Signature GAN
Implementation and numerical experiments of SigWGAN.

- Paper: Sig-Wasserstein GANs for Time Series Generation. 
- Souce code: https://github.com/SigCGANs/Sig-Wasserstein-GANs/tree/main
  
## User Guide
Users are welcome to check the notebooks to see the implementation details and numerical results. 

Datasets:
- Geometric Browian motion
- Rough Bergomi
- S\&P 500

## update
- 29/Feb/2024: Rewrite get_gbm in lib/dataset.py. The numerical results in GBM.ipynb should be slightly different if you rerun the notebook because of randomness.