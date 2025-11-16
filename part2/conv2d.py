import numpy as np
import math
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal

@nki.compiler.skip_middle_end_transformations
@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):
    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]
    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"
    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1
    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == out_channels % 128 == 0
    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width
    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )
    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax
    # Tile sizes
    c_out_tile = 128  # Maximum partition dimension
    c_in_tile = 128   # Maximum for input channels in matmul
    
    # Process the images in batches
    for b in nl.affine_range(batch_size):
        # Process output channels in tiles
        for c_out_start in nl.affine_range(0, out_channels, c_out_tile):
            c_out_end = c_out_start + c_out_tile
            
            # Initialize output tile in SBUF for accumulation
            # Output shape: (c_out_tile, out_height, out_width)
            output_sbuf = nl.ndarray(
                shape=(c_out_tile, out_height, out_width),
                dtype=X.dtype,
                buffer=nl.sbuf
            )
            output_sbuf[...] = 0.0
            
            # Iterate over filter height
            for i in nl.affine_range(filter_height):
                # Iterate over filter width
                for j in nl.affine_range(filter_width):
                    # Process input channels in tiles
                    for c_in_start in nl.affine_range(0, in_channels, c_in_tile):
                        c_in_end = c_in_start + c_in_tile
                        
                        # Load input tile from HBM and extract shifted region
                        # Shape: (c_in_tile, input_height, input_width)
                        input_tile = nl.ndarray(
                            shape=(c_in_tile, input_height, input_width),
                            dtype=X.dtype,
                            buffer=nl.sbuf
                        )
                        nisa.dma_copy(src=X[b, c_in_start:c_in_end, :, :], dst=input_tile)
                        
                        # Create shifted input: select the appropriate spatial region
                        # Shape: (c_in_tile, out_height, out_width)
                        input_shifted = nl.ndarray(
                            shape=(c_in_tile, out_height, out_width),
                            dtype=X.dtype,
                            buffer=nl.sbuf
                        )
                        
                        # Extract the shifted region from the input
                        input_shifted[...] = input_tile[:, i:i+out_height, j:j+out_width]
                        
                        # Load filter slice at position (i, j) into SBUF
                        # W is (out_channels, in_channels, filter_height, filter_width)
                        # We need shape: (c_out_tile, c_in_tile)
                        filter_sbuf = nl.ndarray(
                            shape=(c_out_tile, c_in_tile),
                            dtype=W.dtype,
                            buffer=nl.sbuf
                        )
                        # Use nisa.dma_copy with src and dst parameters
                        nisa.dma_copy(src=W[c_out_start:c_out_end, c_in_start:c_in_end, i, j], dst=filter_sbuf)
                        
                        # perform matmul on each spatial location
                        # for each output pixel position, we compute the dot product
                        for h in nl.affine_range(out_height):
                            for w in nl.affine_range(out_width):
                                # Get input vector at this position: (c_in_tile, 1)
                                input_vec = nl.ndarray(
                                    shape=(c_in_tile, 1),
                                    dtype=X.dtype,
                                    buffer=nl.sbuf
                                )
                                input_vec[:, 0] = input_shifted[:, h, w]
                                
                                # result shape: (c_out_tile, 1)
                                result_vec = nl.matmul(filter_sbuf, input_vec)
                                
                                # accumulate to output in SBUF
                                output_sbuf[:, h, w] += result_vec[:, 0]
            
            # store result tile from SBUF to HBM (no pooling for naive case)
            nisa.dma_copy(src=output_sbuf, dst=X_out[b, c_out_start:c_out_end, :, :])
    return X_out
