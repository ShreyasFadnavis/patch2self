# Patch2Self: Denoising Diffusion MRI with Self-Supervised Learning

This repo demonstrates a framework (Patch2Self) for denoising Diffusion MRI, as described in the paper. 

The main model to perform the denoising is contained in: 
`model/patch2self.py`

We provide support for 4 different types of regressors: 
- Ordinary Least Squares ('ols')
- Ridge ('ridge')
- Lasso ('lasso')
- Multilayer Perceptron ('mlp')
***

## API: To denoise any diffusion data via Patch2Self:
```
# load the 4D DWI data 
denoised_data = patch2self(data, model='ridge')

# one can save the denoised data in the Nifti file format using DIPY:
from dipy.io.image import save_nifti
save_nifti('filename.nii.gz', denoised data, np.eye(4))
```
Tutorial Notebooks:

[Denoise various datasets](notebooks/Denoise_Various_Data.ipynb) shows how `patch2self` can be used to denoise different datasets. 

[Denoising simulated phantom](notebooks/Phantom_Denoising.ipynb) shows how `patch2self` works on denoising on simulated data. 

[Regression Comparison](notebooks/Regression_Comparison.ipynb) shows effect of `patch2self` using different regressors mentioned above. To run the MLP Regressor, make sure to run the [MLP Regressor](notebooks/Regression_MLP.ipynb) first.

[Tractography Comparison](notebooks/Tracking_FiberBundleCoherency.ipynb) shows effect of `patch2self` on tractography using probabilistic tractography via `CSA` and `CSD`. We show comparison between different denoising algorithms using Fiber Bundle Coherency metric explaines in the paper.

[Voxel-wise k-fold Crossvalidation](notebooks/voxel_k-fold_crossvalidation.ipynb) and [Box-plots: k-fold Crossvalidation](notebooks/R2_K-Fold_Box_Plots.ipynb) shows the k-fold crossvalidation analyses. 

[Diffusion Kurtosis Imaging Analysis](notebooks/DKI_Effects_Denoising.ipynb) depicts the effects denoising on DKI maps. 

- Links to the data used in the above tutorials are provided in the supplement.

Dependencies are in the `environment.yml` file.
