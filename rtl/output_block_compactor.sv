`timescale 1ns / 1ps

// -----------------------------------------------------------------------
// output_block_compactor.sv
//
// Paper Fig. 2 / Sec. III: "32 original floating-point numbers are grouped
// into a block, with each block sharing an 8-bit shared scale X." There is
// exactly ONE X per 32 elements -- nothing in the spec shares X at any
// finer granularity than that.
//
// Prior to this module, post_processing_unit produces 16 independent
// pairs (32 elements), each pair carrying its OWN locally-aligned exponent
// (aligned only against its own bias -- never against the other 15 pairs
// in its block). This module is the missing piece: it takes those 16
// pairs, finds the single maximum exponent across the whole block, and
// right-shifts every one of the 32 fractions down to that one shared
// scale, so the block leaving here matches the paper's format exactly
// (and is directly comparable to the golden *_output_x.hex files, which
// are one byte per 32-element block).
//
// NUM_PAIRS=16 (32 elements) is fixed by BLOCK_SIZE in the paper/python
// export script; left as a parameter only so the max-reduction tree below
// stays generic, not because BLOCK_SIZE is meant to vary.
// -----------------------------------------------------------------------
module output_block_compactor #(
    parameter FRACT_WIDTH = 16,
    parameter EXP_WIDTH   = 8,
    parameter NUM_PAIRS   = 16   // 16 pairs = 32 elements = one BLOCK_SIZE block
)(
    input  wire clk,
    input  wire rst_n,

    input  wire                          valid_in    [0:NUM_PAIRS-1],
    input  wire signed [FRACT_WIDTH-1:0] in_fract_0  [0:NUM_PAIRS-1],
    input  wire signed [FRACT_WIDTH-1:0] in_fract_1  [0:NUM_PAIRS-1],
    input  wire        [EXP_WIDTH-1:0]   in_x        [0:NUM_PAIRS-1], // each pair's own local exponent (from post_processing_unit)

    output reg  signed [FRACT_WIDTH-1:0] out_fract_0 [0:NUM_PAIRS-1],
    output reg  signed [FRACT_WIDTH-1:0] out_fract_1 [0:NUM_PAIRS-1],
    output reg         [EXP_WIDTH-1:0]   block_x,     // single shared exponent for the whole 32-element block
    output reg                           valid_out
);

    localparam SHIFT_W = $clog2(FRACT_WIDTH+1);
    localparam LOG2_NP = $clog2(NUM_PAIRS);

    // -------------------------------------------------------------------
    // Stage 0: block-wide max-exponent reduction tree. NUM_PAIRS is a
    // power of 2 (16), so a straightforward binary tree covers it exactly.
    // -------------------------------------------------------------------
    wire [EXP_WIDTH-1:0] max_level [0:LOG2_NP][0:NUM_PAIRS-1];

    genvar lvl, idx;
    generate
        for (idx = 0; idx < NUM_PAIRS; idx = idx + 1) begin : g_max_leaf
            assign max_level[0][idx] = in_x[idx];
        end
        for (lvl = 1; lvl <= LOG2_NP; lvl = lvl + 1) begin : g_max_level
            for (idx = 0; idx < (NUM_PAIRS >> lvl); idx = idx + 1) begin : g_max_cmp
                assign max_level[lvl][idx] =
                    (max_level[lvl-1][2*idx] >= max_level[lvl-1][2*idx+1])
                        ? max_level[lvl-1][2*idx] : max_level[lvl-1][2*idx+1];
            end
        end
    endgenerate

    wire [EXP_WIDTH-1:0] block_max_x = max_level[LOG2_NP][0];

    // -------------------------------------------------------------------
    // Stage 1: per-pair alignment down to block_max_x. Independent shifter
    // per pair -- this stage fires once per completed 32-element block
    // (once every full weight-block sweep), not once per MAC cycle, so
    // sharing one physical shifter across all 16 pairs (as pe_node/
    // temporal_accumulator do for their per-cycle datapath) buys silicon
    // area at the cost of serializing something that's already off the
    // critical per-cycle path. Kept parallel/independent for that reason.
    // -------------------------------------------------------------------
    genvar p;
    generate
        for (p = 0; p < NUM_PAIRS; p = p + 1) begin : g_align_pair

            reg [EXP_WIDTH:0]      exp_diff_p;
            reg [SHIFT_W-1:0]      shift_amt_p;
            reg [FRACT_WIDTH-1:0]  coarse_shifted_0, coarse_shifted_1;
            reg signed [FRACT_WIDTH-1:0] shifted_0, shifted_1;

            always @(*) begin
                // block_max_x >= in_x[p] always (it IS the max), so this is
                // a one-directional (right-only) shift -- no dynamic-sign
                // case needed here, unlike the two-sided aligners elsewhere.
                exp_diff_p  = {1'b0, block_max_x} - {1'b0, in_x[p]};
                shift_amt_p = (exp_diff_p > FRACT_WIDTH) ? SHIFT_W'(FRACT_WIDTH) : SHIFT_W'(exp_diff_p);

                case (shift_amt_p[SHIFT_W-1:2])
                    3'd0:    coarse_shifted_0 = in_fract_0[p];
                    3'd1:    coarse_shifted_0 = in_fract_0[p] >>> 4;
                    3'd2:    coarse_shifted_0 = in_fract_0[p] >>> 8;
                    3'd3:    coarse_shifted_0 = in_fract_0[p] >>> 12;
                    default: coarse_shifted_0 = in_fract_0[p] >>> 16;
                endcase
                case (shift_amt_p[SHIFT_W-1:2])
                    3'd0:    coarse_shifted_1 = in_fract_1[p];
                    3'd1:    coarse_shifted_1 = in_fract_1[p] >>> 4;
                    3'd2:    coarse_shifted_1 = in_fract_1[p] >>> 8;
                    3'd3:    coarse_shifted_1 = in_fract_1[p] >>> 12;
                    default: coarse_shifted_1 = in_fract_1[p] >>> 16;
                endcase

                case (shift_amt_p[1:0])
                    2'd0:    shifted_0 = coarse_shifted_0;
                    2'd1:    shifted_0 = coarse_shifted_0 >>> 1;
                    2'd2:    shifted_0 = coarse_shifted_0 >>> 2;
                    default: shifted_0 = coarse_shifted_0 >>> 3;
                endcase
                case (shift_amt_p[1:0])
                    2'd0:    shifted_1 = coarse_shifted_1;
                    2'd1:    shifted_1 = coarse_shifted_1 >>> 1;
                    2'd2:    shifted_1 = coarse_shifted_1 >>> 2;
                    default: shifted_1 = coarse_shifted_1 >>> 3;
                endcase
            end

            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    out_fract_0[p] <= {FRACT_WIDTH{1'b0}};
                    out_fract_1[p] <= {FRACT_WIDTH{1'b0}};
                end else begin
                    out_fract_0[p] <= shifted_0;
                    out_fract_1[p] <= shifted_1;
                end
            end
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            block_x   <= {EXP_WIDTH{1'b0}};
            valid_out <= 1'b0;
        end else begin
            block_x   <= block_max_x;
            valid_out <= valid_in[0]; // all 16 pairs in a block share one streaming schedule
        end
    end

endmodule
