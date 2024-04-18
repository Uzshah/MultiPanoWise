import numpy as np
from skimage import filters
import torch
def reconstruct(d_batch):
    """
    Compute equirectangular normal map from the equirectangular depth map for a batch of depth maps.

    Parameters:
    d_batch (numpy.ndarray): Batch of depth maps (equirectangular projection) in millimeters.
                            Shape: (batch_size, height, width)

    Returns:
    n_hat_batch (numpy.ndarray): Batch of equirectangular normal maps.
                                Shape: (batch_size, height, width, 3)
    """
    d_batch = d_batch.cpu()
    d_batch = d_batch.squeeze(1)
    output = []
    for d in d_batch:
        d = d.detach().numpy()
        # Compute gradients of the depth map
        d_theta = filters.scharr_h(d)
        d_phi = filters.scharr_v(d)
    
        # Constants for converting pixel indices to spherical angles
        w, h = d.shape
        k_u = np.pi / w
        k_v = 2.0 * np.pi / h
    
        # Create meshgrids for theta and phi
        j = np.arange(w) + 0.5
        i = np.arange(h) + 0.5
        theta, phi = np.meshgrid(k_u * j, k_v * i - np.pi, indexing='ij')
        
        # Compute vectors in spherical coordinates
        r = np.stack((np.sin(theta) * np.cos(phi), 
                      np.cos(theta), 
                      -np.sin(theta) * np.sin(phi)), axis=-1)
        r_theta = np.stack((np.cos(theta) * np.cos(phi), 
                            -np.sin(theta), 
                            -np.cos(theta) * np.sin(phi)), axis=-1)
        r_phi = np.stack((-np.sin(phi), 
                          np.zeros_like(phi), 
                          -np.cos(phi)), axis=-1)
    
        # Compute perturbation vectors
        p_t = (d_theta[..., np.newaxis] * r) + (k_u * d[..., np.newaxis] * r_theta)
        p_p = (d_phi[..., np.newaxis] * r) + (k_v * d[..., np.newaxis] * r_phi)
    
        # Compute cross product of perturbation vectors to get normals
        n_hat = np.cross(p_p, p_t)
        
        # Normalize the normals
        n_hat_norm = np.linalg.norm(n_hat, axis=2, keepdims=True)
        n_hat_norm = np.where(n_hat_norm > 0, n_hat_norm, 1)
        n_hat = n_hat / n_hat_norm
        output.append(torch.from_numpy(n_hat.transpose((2, 0, 1))))
    n_hat = torch.from_numpy(np.stack(output))
    return n_hat