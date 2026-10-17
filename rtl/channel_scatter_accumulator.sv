`timescale 1ns / 1ps

// -----------------------------------------------------------------------
// channel_scatter_accumulator.sv
//
// This is the ONLY new hardware needed for Hessian-grouped weights, beyond
// what the paper already specifies. Everything it reuses is unchanged:
//   - fp2_block_decode_wrapper / fp2_processing_element: the paper's exact
//     pair-level FP2 encoding and multiply (Sec. IV-E/F) -- untouched.
//   - the sign-magnitude -> two's-complement conversion for spatial_terms:
//     copied verbatim from pe_node.sv.
//   - temporal_accumulator: the paper's exact dynamic-exponent-alignment
//     accumulation (Sec. IV-C) -- untouched, just instantiated once per
//     output channel (256x) instead of once per fixed PE grid position.
//
// What's new: instead of dsp_packed_spatial_adder always summing all 16
// pairs of a block into one PE-local value (valid only when all 16 belong
// to the same output channel -- true for spatial-mode weights, false for
// Hessian-grouped ones), each pair is routed to its OWN channel's
// accumulator based on channel_id[pair], computed offline and loaded via
// weight_lane_channel_map. Pairs that land on the same channel in the same
// cycle are summed together before hitting that channel's
// temporal_accumulator; pairs on different channels accumulate
// independently and in parallel, so throughput is unchanged (still one
// weight block processed per cycle).
// -----------------------------------------------------------------------
module channel_scatter_accumulator #(
    parameter FRACT_WIDTH   = 16,
    parameter EXP_WIDTH     = 8,
    parameter NUM_PAIRS     = 16,
    parameter NUM_CHANNELS  = 256,
    parameter CH_WIDTH      = 8
)(
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    input  wire enc_type,

    input  wire [63:0]          act_data_in, // 16 pairs * 4 bits (unchanged format)
    input  wire [EXP_WIDTH-1:0] act_x_in,    // block-shared activation scale (unchanged)
    input  wire [63:0]          wt_data_in,  // 16 pairs * 4 bits (unchanged format)
    input  wire [EXP_WIDTH-1:0] wt_x_in,     // block-shared weight scale (unchanged)

    // NEW: which channel each of the 16 pairs belongs to, from weight_lane_channel_map
    input  wire [CH_WIDTH-1:0]  channel_id [0:NUM_PAIRS-1],

    // Per-channel accumulated result (read out after the compute sweep completes,
    // same "settled value" pattern the existing testbench already uses)
    output wire [FRACT_WIDTH-1:0] accum_fract [0:NUM_CHANNELS-1],
    output wire [EXP_WIDTH-1:0]   accum_x     [0:NUM_CHANNELS-1],
    output wire                   accum_valid [0:NUM_CHANNELS-1]
);

    // -------------------------------------------------------------------
    // Unchanged: paper's per-pair FP2 decode (fp2_block_decode_wrapper),
    // copied verbatim from pe_node.sv including the spatial_terms
    // conversion and the +/-14-per-term / +/-224-total margin it relies on.
    // -------------------------------------------------------------------
    wire       unit_out_s   [0:NUM_PAIRS-1];
    wire [3:0] unit_out_m   [0:NUM_PAIRS-1];
    wire       unit_is_zero [0:NUM_PAIRS-1];
    wire signed [5:0] spatial_terms [0:NUM_PAIRS-1];

    genvar i;
    generate
        for (i = 0; i < NUM_PAIRS; i = i + 1) begin : lanes
            fp2_block_decode_wrapper fp2_lane (
                .clk(clk), .rst_n(rst_n), .enc_type(enc_type),
                .seq_raddr(8'h00),
                .shared_scale_x(act_x_in),
                .act_pair_data(act_data_in[i*4 +: 4]),
                .wt_pair_data(wt_data_in[i*4 +: 4]),
                .out_s(unit_out_s[i]), .out_m(unit_out_m[i]), .is_zero(unit_is_zero[i]),
                .block_x_out()
            );
            assign spatial_terms[i] = unit_is_zero[i] ? 6'sd0 :
                (unit_out_s[i] ? -$signed({2'b00, unit_out_m[i]}) : $signed({2'b00, unit_out_m[i]}));
        end
    endgenerate

    // Unchanged: paper's shared-scale addition (Sec. IV-B).
    wire [EXP_WIDTH-1:0] combined_x;
    localparam [EXP_WIDTH-1:0] EXP_BIAS = 1 << (EXP_WIDTH-1);
    assign combined_x = act_x_in + wt_x_in - EXP_BIAS;

    // -------------------------------------------------------------------
    // NEW: per-channel routing. For each of the NUM_CHANNELS possible
    // destinations, sum whichever of this cycle's 16 pairs (usually 0 or 1,
    // occasionally a few more) target that channel, then feed the result
    // into an unmodified temporal_accumulator for that channel.
    //
    // Width note: worst case all 16 pairs (each bounded +/-14, same margin
    // pe_node.sv already established) land on one channel in one cycle ->
    // max magnitude 224, same bound as the original full spatial_sum, so
    // match_sum reuses the identical 9-bit signed width -- no new margin
    // analysis needed.
    // -------------------------------------------------------------------
    genvar c, p;
    generate
        for (c = 0; c < NUM_CHANNELS; c = c + 1) begin : channel

            wire match [0:NUM_PAIRS-1];
            wire signed [5:0] matched_term [0:NUM_PAIRS-1];
            for (p = 0; p < NUM_PAIRS; p = p + 1) begin : match_gen
                assign match[p] = valid_in && (channel_id[p] == c[CH_WIDTH-1:0]);
                assign matched_term[p] = match[p] ? spatial_terms[p] : 6'sd0;
            end

            wire match_any = |{match[0], match[1], match[2], match[3], match[4], match[5], match[6], match[7],
                                match[8], match[9], match[10], match[11], match[12], match[13], match[14], match[15]};

            wire signed [8:0] match_sum =
                $signed(matched_term[0])  + $signed(matched_term[1])  + $signed(matched_term[2])  + $signed(matched_term[3]) +
                $signed(matched_term[4])  + $signed(matched_term[5])  + $signed(matched_term[6])  + $signed(matched_term[7]) +
                $signed(matched_term[8])  + $signed(matched_term[9])  + $signed(matched_term[10]) + $signed(matched_term[11]) +
                $signed(matched_term[12]) + $signed(matched_term[13]) + $signed(matched_term[14]) + $signed(matched_term[15]);

            wire [FRACT_WIDTH-1:0] sign_ext_match_sum = {{(FRACT_WIDTH-9){match_sum[8]}}, match_sum};

            // Unchanged: paper's exact temporal accumulator (Sec. IV-C),
            // just addressed by channel instead of fixed PE position.
            temporal_accumulator #(
                .FRACT_WIDTH(FRACT_WIDTH), .EXP_WIDTH(EXP_WIDTH)
            ) t_accum (
                .clk(clk), .rst_n(rst_n), .valid_in(match_any),
                .psum_fract(sign_ext_match_sum), .psum_x(combined_x),
                .accum_fract(accum_fract[c]), .accum_x(accum_x[c]), .valid_out(accum_valid[c])
            );
        end
    endgenerate

endmodule
