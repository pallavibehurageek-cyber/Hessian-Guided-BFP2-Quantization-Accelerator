`timescale 1ns / 1ps

module fp2_block_decode_wrapper (
    input  wire        clk,
    input  wire        rst_n,
    
    // Instruction/Format controls
    input  wire        enc_type,  // 0: E1M0, 1: E0M1
    
    // Sequential address from MAC controller
    input  wire [7:0]  seq_raddr, 
    
    // Reordered data inputs (already fetched via sequential seq_raddr)
    input  wire [7:0]  shared_scale_x,  // Decoded X for this block
    input  wire [3:0]  act_pair_data,   // Reordered activation pair (2 bits each)
    input  wire [3:0]  wt_pair_data,    // Sequentially stored weight pair (2 bits each)
    
    // Outputs to Temporal Accumulator
    // out_m widened from 3 to 4 bits: fp2_processing_element now applies
    // the K in {0,1,2} multiplication factor (Section IV-E) to the base
    // F*W magnitude, and the K=2 case (both elements of the pair live on
    // both operands) needs one extra bit to avoid truncation.
    output wire        out_s,
    output wire [3:0]  out_m,
    output wire        is_zero,
    output wire [7:0]  block_x_out      // Pass-through scale for accumulation
);

    // Unpack 4-bit data into FP2 components (1 sign bit, 1 val bit, 2 flag bits)
    wire act_s       = act_pair_data[3];
    wire act_val     = act_pair_data[2];
    wire [1:0] act_c = act_pair_data[1:0];
    
    wire wt_s        = wt_pair_data[3];
    wire wt_val      = wt_pair_data[2];
    wire [1:0] wt_c  = wt_pair_data[1:0];

    // Instantiate the corrected PE logic name
    fp2_processing_element pe_unit_inst (
        .enc_type(enc_type),
        .act_s(act_s),
        .act_val(act_val),
        .act_c(act_c),
        .wt_s(wt_s),
        .wt_val(wt_val),
        .wt_c(wt_c),
        .out_s(out_s),
        .out_m(out_m),
        .is_zero(is_zero)
    );

    // Pass the block scale X directly to the accumulator logic
    assign block_x_out = shared_scale_x;

endmodule
