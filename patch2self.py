from sklearn import linear_model
import numpy as np


def patch2self(data, patch_radius=[1, 1, 1], mask=None):

    if mask is None:
        # If mask is not specified, use the whole volume
        mask = np.ones_like(data, dtype=bool)[..., 0]

    def _extract_3d_patches(arr, patch_radius=[1, 1, 1]):

        if isinstance(patch_radius, int):
            patch_radius = np.ones(3, dtype=int) * patch_radius
        if len(patch_radius) != 3:
            raise ValueError("patch_radius should have length 3")
        else:
            patch_radius = np.asarray(patch_radius).astype(int)
        patch_size = 2 * patch_radius + 1

        dim = arr.shape[-1]

        all_patches = []
        # loop around and find the 3D patch for each direction at each pixel
        for k in range(patch_radius[2], arr.shape[2] - patch_radius[2], 5):
            for j in range(patch_radius[1], arr.shape[1] - patch_radius[1], 5):
                for i in range(patch_radius[0], arr.shape[0] - patch_radius[0], 5):
                    # Shorthand for indexing variables:
                    ix1 = i - patch_radius[0]
                    ix2 = i + patch_radius[0] + 1
                    jx1 = j - patch_radius[1]
                    jx2 = j + patch_radius[1] + 1
                    kx1 = k - patch_radius[2]
                    kx2 = k + patch_radius[2] + 1

                    X = arr[ix1:ix2, jx1:jx2, kx1:kx2].reshape(np.prod(patch_size), dim)
                    all_patches.append(X)

        return np.array(all_patches).T

    train = _extract_3d_patches(np.pad(data, ((1, 1), (1, 1), (1, 1)))
        , patch_radius=patch_radius)
    print(train.shape)

    print('Patch Extraction Done...')

    patch_radius = np.asarray(patch_radius).astype(int)

    denoised_array = np.zeros((data.shape))

    for f in range(0, data.shape[3]):
        model = linear_model.LinearRegression(copy_X=False,
                                              fit_intercept=True,
                                              n_jobs=-1, normalize=False)

        # Delete the f-th volume
        X1 = train[:f, :, :]
        X2 = train[f+1:, :, :]

        print('Training for resolution: ', patch_radius)
        cur_X = np.reshape(np.concatenate((X1, X2), axis=0),
                           (-1, train.shape[2]))

        Y = train[f, train.shape[1]//2, :]

        model.fit(cur_X.T[::100], Y.T[::100])

        del cur_X, Y, X1, X2
        print(' -> Trained to Denoise Volume: ', f)

        denoised_array[..., f] = model.predict(cur_X.T).reshape(
            data.shape[0], data.shape[1], data.shape[2])

        print('Denoising Volume ', f, ' Complete...')

    denoised_array[mask == 0] = 0
    return denoised_array
