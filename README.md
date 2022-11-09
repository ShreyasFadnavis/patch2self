# This repo is `Outdated` and work in progress. Please use the implementation in here: [DIPY](https://dipy.org/documentation/1.5.0/examples_built/denoise_patch2self/#example-denoise-patch2self) (>=1.4.0) for the best possible implementation of Patch2Self.


## Video on how to use: [YouTube](https://www.youtube.com/watch?v=S4MoCox98M8&t=1s)

## Medium Article for detailed instructions: [Blogpost Link](https://shreyasfadnavis.medium.com/patch2self-self-supervised-denoising-via-statistical-independence-4601fda38c20)


<img src="https://upload.wikimedia.org/wikipedia/en/thumb/0/08/Logo_for_Conference_on_Neural_Information_Processing_Systems.svg/1200px-Logo_for_Conference_on_Neural_Information_Processing_Systems.svg.png" width=200>

# Patch2Self: Denoising Diffusion MRI with Self-Supervised Learning

This repo demonstrates a framework (Patch2Self) for denoising Diffusion MRI, as described in the paper:
[NeurIPS Spotlight](https://papers.nips.cc/paper/2020/file/bc047286b224b7bfa73d4cb02de1238d-Paper.pdf)

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
