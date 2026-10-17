`timescale 1ns / 1ps

module post_processing_unit #(
    parameter FRACT_WIDTH = 16,
    parameter EXP_WIDTH   = 8
)(
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   valid_in,

    // High-precision output from Temporal Accumulator (PARALLEL PAIR)
    // FIXED: each of the two PE columns feeding this pair has its OWN
    // independent temporal_accumulator, and therefore its own independently
    // -evolved accum_x -- they are NOT guaranteed to match. The previous
    // version only accepted a single psum_x (wired to column 0's exponent
    // only at the top level), silently discarding column 1's real exponent
    // and treating its value as if it shared column 0's scale. Now both are
    // taken in and aligned here, the same way bias alignment already works
    // below.
    input  wire signed [FRACT_WIDTH-1:0] psum_fract_0,
    input  wire signed [FRACT_WIDTH-1:0] psum_fract_1,
    input  wire        [EXP_WIDTH-1:0]   psum_x_0,
    input  wire        [EXP_WIDTH-1:0]   psum_x_1,

    // Layer Bias Inputs
    input  wire signed [FRACT_WIDTH-1:0] bias_fract,
    input  wire        [EXP_WIDTH-1:0]   bias_x,

    // Configuration
    input  wire                   en_relu, // 1 to enable ReLU, 0 to bypass

    // Post-processed output (PARALLEL PAIR)
    // pp_x is now the pair-local shared exponent (max of psum_x_0/psum_x_1,
    // further aligned against bias_x below) -- both pp_fract_0 and
    // pp_fract_1 are guaranteed aligned to this single value.
    output reg  signed [FRACT_WIDTH-1:0] pp_fract_0,
    output reg  signed [FRACT_WIDTH-1:0] pp_fract_1,
    output reg         [EXP_WIDTH-1:0]   pp_x,
    output reg                           valid_out
);

    localparam SHIFT_W = $clog2(FRACT_WIDTH+1);

    // -------------------------------------------------------------------------
    // Stage 0 (ADDED): align psum_fract_0/psum_fract_1 to a single pair-local
    // exponent before anything else touches them. Same coarse/fine shared
    // -shifter pattern as temporal_accumulator.sv, kept as two independent
    // shifters here (rather than one shared one) since this stage only fires
    // once per completed output pair, not once per MAC cycle -- unlike the
    // per-cycle datapath, area sharing isn't the binding constraint here.
    // -------------------------------------------------------------------------
    reg signed [EXP_WIDTH:0] pair_exp_diff;      // psum_x_0 - psum_x_1
    reg [SHIFT_W-1:0]        pair_shift_amt;
    reg                      shift_which;        // 1: shift fract_0, 0: shift fract_1
    reg [EXP_WIDTH-1:0]      pair_local_x;        // max(psum_x_0, psum_x_1)

    reg [FRACT_WIDTH-1:0]    pair_coarse_shifted;
    reg signed [FRACT_WIDTH-1:0] pair_shift_operand;
    reg signed [FRACT_WIDTH-1:0] pair_shifted_result;
    reg signed [FRACT_WIDTH-1:0] pair_aligned_fract_0;
    reg signed [FRACT_WIDTH-1:0] pair_aligned_fract_1;

    always @(*) begin
        pair_exp_diff = $signed({1'b0, psum_x_0}) - $signed({1'b0, psum_x_1});

        if (pair_exp_diff > 0) begin
            // Column 0's scale is larger -- shift column 1's fraction down to match.
            pair_shift_amt     = (pair_exp_diff > FRACT_WIDTH) ? SHIFT_W'(FRACT_WIDTH) : SHIFT_W'(pair_exp_diff);
            pair_shift_operand = psum_fract_1;
            shift_which        = 1'b0; // result routes to fract_1
            pair_local_x       = psum_x_0;
        end else if (pair_exp_diff < 0) begin
            // Column 1's scale is larger -- shift column 0's fraction down to match.
            pair_shift_amt     = ((-pair_exp_diff) > FRACT_WIDTH) ? SHIFT_W'(FRACT_WIDTH) : SHIFT_W'(-pair_exp_diff);
            pair_shift_operand = psum_fract_0;
            shift_which        = 1'b1; // result routes to fract_0
            pair_local_x       = psum_x_1;
        end else begin
            pair_shift_amt     = {SHIFT_W{1'b0}};
            pair_shift_operand = psum_fract_0; // shift_amt==0 -> passthrough either way
            shift_which        = 1'b1;
            pair_local_x       = psum_x_0;
        end

        case (pair_shift_amt[SHIFT_W-1:2])
            3'd0:    pair_coarse_shifted = pair_shift_operand;
            3'd1:    pair_coarse_shifted = pair_shift_operand >>> 4;
            3'd2:    pair_coarse_shifted = pair_shift_operand >>> 8;
            3'd3:    pair_coarse_shifted = pair_shift_operand >>> 12;
            default: pair_coarse_shifted = pair_shift_operand >>> 16;
        endcase

        case (pair_shift_amt[1:0])
            2'd0:    pair_shifted_result = pair_coarse_shifted;
            2'd1:    pair_shifted_result = pair_coarse_shifted >>> 1;
            2'd2:    pair_shifted_result = pair_coarse_shifted >>> 2;
            default: pair_shifted_result = pair_coarse_shifted >>> 3;
        endcase

        pair_aligned_fract_0 = shift_which ? pair_shifted_result : psum_fract_0;
        pair_aligned_fract_1 = shift_which ? psum_fract_1        : pair_shifted_result;
    end

    // -------------------------------------------------------------------------
    // Stage 1 (existing): align the now-pair-aligned pair against bias_x.
    // Unchanged logic, just fed from pair_aligned_fract_0/1 and pair_local_x
    // instead of the original single psum_fract_0/1/psum_x.
    // -------------------------------------------------------------------------
    reg signed [EXP_WIDTH:0]     exp_diff;
    reg        [SHIFT_W-1:0]     shift_amt;

    reg signed [FRACT_WIDTH-1:0] aligned_psum_0;
    reg signed [FRACT_WIDTH-1:0] aligned_psum_1;
    reg signed [FRACT_WIDTH-1:0] aligned_bias;
    reg        [EXP_WIDTH-1:0]   aligned_x;

    always @(*) begin
        exp_diff = $signed({1'b0, pair_local_x}) - $signed({1'b0, bias_x});

        if (exp_diff > 0) begin
            shift_amt      = (exp_diff > FRACT_WIDTH) ? SHIFT_W'(FRACT_WIDTH) : SHIFT_W'(exp_diff);
            aligned_bias   = bias_fract >>> shift_amt;
            aligned_psum_0 = pair_aligned_fract_0;
            aligned_psum_1 = pair_aligned_fract_1;
            aligned_x      = pair_local_x;
        end else if (exp_diff < 0) begin
            shift_amt      = ((-exp_diff) > FRACT_WIDTH) ? SHIFT_W'(FRACT_WIDTH) : SHIFT_W'(-exp_diff);
            aligned_bias   = bias_fract;
            aligned_psum_0 = pair_aligned_fract_0 >>> shift_amt;
            aligned_psum_1 = pair_aligned_fract_1 >>> shift_amt;
            aligned_x      = bias_x;
        end else begin
            aligned_bias   = bias_fract;
            aligned_psum_0 = pair_aligned_fract_0;
            aligned_psum_1 = pair_aligned_fract_1;
            aligned_x      = pair_local_x;
        end
    end

    // -------------------------------------------------------------------------
    // 2. Bias Addition
    // -------------------------------------------------------------------------
    wire signed [FRACT_WIDTH-1:0] biased_sum_0 = aligned_psum_0 + aligned_bias;
    wire signed [FRACT_WIDTH-1:0] biased_sum_1 = aligned_psum_1 + aligned_bias;

    // -------------------------------------------------------------------------
    // 3. ReLU & Sign Logic (Evaluated on the BIASED sum)
    // -------------------------------------------------------------------------
    wire is_negative_0 = biased_sum_0[FRACT_WIDTH-1];
    wire is_negative_1 = biased_sum_1[FRACT_WIDTH-1];

    wire signed [FRACT_WIDTH-1:0] relu_result_0 = (en_relu & is_negative_0) ? {FRACT_WIDTH{1'b0}} : biased_sum_0;
    wire signed [FRACT_WIDTH-1:0] relu_result_1 = (en_relu & is_negative_1) ? {FRACT_WIDTH{1'b0}} : biased_sum_1;

    // -------------------------------------------------------------------------
    // 4. Output Registers
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pp_fract_0 <= {FRACT_WIDTH{1'b0}};
            pp_fract_1 <= {FRACT_WIDTH{1'b0}};
            pp_x       <= {EXP_WIDTH{1'b0}};
            valid_out  <= 1'b0;
        end else begin
            pp_fract_0 <= relu_result_0;
            pp_fract_1 <= relu_result_1;
            pp_x       <= aligned_x;
            valid_out  <= valid_in;
        end
    end

endmodule
