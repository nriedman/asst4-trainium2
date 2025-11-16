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

    # Load W into sbuf and transpose
    W_res = W.reshape((c_out_pmax, n_tiles_c_out, c_in_pmax, n_tiles_c_in, filter_height, filter_width))
    W_T = nl.ndarray((c_in_pmax, n_tiles_c_in, c_out_pmax, n_tiles_c_out, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)

    # Prepare bias arranged in the same tile ordering as the kernel reshape
    B_hbm = nl.ndarray((c_out_pmax, n_tiles_c_out), dtype=bias.dtype, buffer=nl.hbm)
    # Copy the reshaped bias (shape (c_out_pmax, n_tiles_c_out)) into HBM so per-column DMA is possible
    nisa.dma_copy(src=bias.reshape((c_out_pmax, n_tiles_c_out)), dst=B_hbm)

    for c_out_tile in nl.affine_range(n_tiles_c_out):
        for c_in_tile in nl.affine_range(n_tiles_c_in):
            for i in nl.affine_range(filter_height):
                for j in nl.affine_range(filter_width):
                    w = nl.ndarray((c_out_pmax, c_in_pmax), dtype=W_res.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(src=W_res[:, c_out_tile, :, c_in_tile, i, j], dst=w)

                    w_T_psum = nisa.nc_transpose(w, dtype=w.dtype, engine=nki.isa.tensor_engine)

                    W_T[:, c_in_tile, :, c_out_tile, i, j] = nisa.tensor_copy(w_T_psum, dtype=w_T_psum.dtype)
    
    # W_T now has all the weights in SBUF, transposed and chunked by c_in and c_out tile

    X_res = X.reshape((batch_size, c_in_pmax, n_tiles_c_in, input_height, input_width))
    X_out_res = X_out.reshape((batch_size, c_out_pmax, n_tiles_c_out, out_pool_height, out_pool_width))

    # Main loop

    for b in nl.affine_range(batch_size):

        for out_row_pair in nl.affine_range(out_height // 2):

            # Make space to store output rows in sbuf
            conv_out_rows = nl.ndarray((c_out_pmax, n_tiles_c_out, 2, out_width), dtype=X_out.dtype, buffer=nl.sbuf)

            # For each output row pair, we need filter_height + 1 input rows
            # out_row_pair ranges from 0 to out_height//2 - 1
            # We need input rows [out_row_pair*2 : out_row_pair*2 + filter_height + 1]
            # Shape: (c_in_pmax, n_tiles_c_in, filter_height + 1, input_width)
            x_in_rows = nl.ndarray((c_in_pmax, n_tiles_c_in, filter_height + 1, input_width), dtype=X.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=x_in_rows,
                src=X_res[b, :, :, out_row_pair*2:out_row_pair*2 + filter_height + 1, :]
            )

            for c_out_tile in nl.affine_range(n_tiles_c_out):
                # Load bias column for this tile into SBUF as (c_out_pmax,1)
                b_sbuf = nl.ndarray((c_out_pmax, 1), dtype=bias.dtype, buffer=nl.sbuf)
                # Copy the column from the HBM-resident reshaped bias into SBUF (2D->2D DMA)
                nisa.dma_copy(src=B_hbm[:, c_out_tile:c_out_tile+1], dst=b_sbuf)

                # Now, process one input row at a time
                for out_row in nl.affine_range(2):
                    # Shape: (c_out_pmax, out_width)
                    row_conv_psum = nl.zeros((c_out_pmax, out_width), nl.float32, buffer=nl.psum)

                    # Compute the convolution via matrix mult
                    for c_in_tile in nl.affine_range(n_tiles_c_in):
                        for i in nl.affine_range(filter_height):
                            for j in nl.affine_range(filter_width):
                                # W_T Shape: (c_in_pmax, n_tiles_c_in, c_out_pmax, n_tiles_c_out, filter_height, filter_width)
                                # Shape: (c_in_pmax, c_out_pmax)
                                w_T = W_T[:, c_in_tile, :, c_out_tile, i, j]
                                
                                # Shape: (c_in_pmax, out_width)
                                x_in = nisa.tensor_copy(x_in_rows[:, c_in_tile, out_row + i, j:j+out_width])

                                # Do the matrix multiplication and accumulate!
                                row_conv_psum += nisa.nc_matmul(w_T[...], x_in[...])
                    
                    # Add the bias and move to sbuf
                    # Shape: (c_out_pmax, out_width)
                    conv_out_rows[:, c_out_tile, out_row, :] = nisa.tensor_scalar(row_conv_psum, nl.add, b_sbuf[:, 0])
            
            # Now, conv_out_rows stores two complete rows of convolutions plus bias
            # Shape: (c_out_pmax, n_tiles_c_out, 2, out_width)

            # We can go ahead and apply max pooling if necessary
            conv_out_rows_res = conv_out_rows.reshape((c_out_pmax, n_tiles_c_out, 2 // pool_size, pool_size, out_width // pool_size, pool_size))
            pool_out_rows = nisa.tensor_reduce(nl.max, conv_out_rows_res, axis=[3,5])

            # Pool out_rows now contains fused convolution/max-pool row(s).
            # Shape: (c_out_pmax, n_tiles_c_out, 1 or 2, out_pool_width)

            # Go ahead and write to X_out_res
            # Copy per-tile / per-output-subrow to avoid multi-dim DMA ordering issues
            nisa.dma_copy(
                src=pool_out_rows,
                dst=X_out_res[b, :, :, out_row_pair*(2//pool_size):(out_row_pair+1)*(2//pool_size), :]
            )

    return X_out_res.reshape(X_out.shape)
