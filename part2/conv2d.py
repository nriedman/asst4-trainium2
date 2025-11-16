import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal

"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""
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

    c_out_pmax = nl.tile_size.pmax
    n_tiles_c_out = out_channels // c_out_pmax

    hw_pmax = nl.tile_size.pmax
    n_tiles_hw = (input_height * input_width) // hw_pmax

    w_pmax = nl.tile_size.pmax
    n_tiles_w = (input_width + w_pmax - 1) // w_pmax

    pool_tile_height = 2 // pool_size
    pool_tile_width = hw_pmax // pool_size

    # Load W into sbuf and transpose
    W_res = W.reshape((c_out_pmax, c_in_pmax, n_tiles_c_out, n_tiles_c_in, filter_height, filter_width))
    W_T = nl.ndarray((c_in_pmax, c_out_pmax, n_tiles_c_out, n_tiles_c_in, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)

    B_sbuf = nl.ndarray((c_out_pmax, n_tiles_c_out), dtype=bias.dtype, buffer=nl.sbuf)
    
    for c_out_tile in nl.affine_range(n_tiles_c_out):
        # load in bias while we're at it
        nisa.dma_copy(src=bias[c_out_tile*c_out_pmax:(c_out_tile+1)*c_out_pmax], dst=B_sbuf[:, c_out_tile])

        for c_in_tile in nl.affine_range(n_tiles_c_in):
            for i in nl.affine_range(filter_height):
                for j in nl.affine_range(filter_width):
                    w = nl.ndarray((c_out_pmax, c_in_pmax), dtype=W_res.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(src=W_res[:, :, c_out_tile, c_in_tile, i, j], dst=w)

                    w_T_psum = nisa.nc_transpose(w, dtype=w.dtype, engine=nki.isa.tensor_engine)

                    W_T[:, :, c_out_tile, c_in_tile, i, j] = nisa.tensor_copy(w_T_psum, dtype=w_T_psum.dtype)
    
    # W_T now has all the weights in SBUF, transposed and chunked by c_in and c_out tile

    # Main loop

    for b in nl.affine_range(batch_size):

        for c_out_tile in nl.affine_range(n_tiles_c_out):

            for out_row in nl.affine_range(out_height // 2, 2):
                pass

    return X_out


"""
for c_out_tile in nl.affine_range(n_tiles_c_out):

            for hw_tile_pair in nl.affine_range(n_tiles_hw // 2):
                
                # Accumulate the results for this output tile in psum 
                conv_tile_out_psum = nl.zeros((c_out_pmax, 2, hw_pmax), X_out.dtype, buffer=nl.psum)

                for c_in_tile in nl.affine_range(n_tiles_c_in):
                    for i in nl.affine_range(filter_height):
                        for j in nl.affine_range(filter_width):
                            
                            # Transposed weights are pre-loaded
                            # Shape: (c_in_pmax, c_out_pmax)
                            w_T = W_T[:, :, c_out_tile, c_in_tile, i, j]

                            # Load in shifted image input
                            x_in = nl.zeros((c_in_pmax, 2, hw_pmax), dtype=X.dtype, buffer=nl.sbuf)

                            h_start = (hw_tile_pair // n_tiles_w) * 2

                            conv_tile_out_psum += nisa.nc_matmul(w_T[...], x_in[...])

                # Collect the accumulated convolution tile
                # Shape: (c_out_pmax, 2, hw_pmax)
                pool_tile_out = nisa.tensor_scalar(conv_tile_out_psum, nl.add, B_sbuf[:, c_out_tile])

                if pool_size > 1:
                    pool_tile_out_res = pooL_tile_out.reshape((c_out_pmax, 1, 2, hw_pmax//2, 2))
                    pool_tile_out = nisa.tensor_reduce(nl.max, pool_tile_out_res, axis=[2,4])
                
                # Pool out has the convolution with the bias + max pooling
                # Shape: (c_out_pmax, 2 // pool_size, hw_pmax // pool_size)

                # At this point, we can go ahead and write the results to HBM
                out_width_start = (hw_tile_pair % n_tiles_w) * pool_tile_width
                out_width_end = out_width_start + pool_tile_width

                out_height_start = (hw_tile_pair // n_tiles_w) * pool_tile_height
                out_height_end = out_height_start + pool_tile_height
                
                nisa.dma_copy(
                    src=pool_tile_out,
                    dst=X_out[b, c_out_tile*c_out_pmax:(c_out_tile+1)*c_out_pmax, out_height_start:out_height_end, out_width_start:out_width_end]
                )
"""