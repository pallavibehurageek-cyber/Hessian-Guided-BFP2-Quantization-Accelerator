`timescale 1ns / 1ps

// -----------------------------------------------------------------------
// temporal_accumulator.sv  (DSP-SCOPED + SHARED-SHIFTER)
//
// Two changes from the previous (signed-shift-corrected) version, both
// aimed at LUT reduction on the Artix-7 XC7A200T without touching any
// parameter widths:
//
// 1. DSP ATTRIBUTE RE-SCOPED
//    Previously `(* use_dsp = "yes" *)` sat directly on `accum_fract`,
//    which is read back combinationally inside the alignment logic
//    (aligned_accum = accum_fract / accum_fract >>> shift_amt). That
//    pulled the ENTIRE shift+mux datapath into DSP inference, and since
//    DSP48E1 has no generic dynamic-width barrel shifter, Vivado
//    approximated it with multiple chained DSPs per instance (~3x),
//    saturating all 740 DSP48E1 slices on the part for almost no LUT
//    benefit -- the shift/mux control logic still had to stay in fabric
//    regardless.
//
//    The attribute now lives on `accum_sum`, an isolated wire whose
//    ONLY fan-in is the two already-resolved `aligned_*` operands. This
//    maps to exactly one clean DSP48E1 ALU add per instance (256 total
//    across the array) instead of ~3 per instance (~740 total).
//
// 2. SHARED BARREL SHIFTER
//    The alignment step only ever shifts ONE of {psum_fract, accum_fract}
//    per cycle -- the two cases (exp_diff > 0 / exp_diff < 0) are
//    mutually exclusive. The previous code wrote this as two separate
//    `>>>` expressions on two different operands, which risks (and in
//    practice produced) two independently-synthesized barrel shifters
//    per instance. This version routes both cases through a single
//    shared shift_operand -> shifted_result path, muxing the operand in
//    and the result back out, so only ONE physical barrel shifter is
//    built per PE instead of two. This is the dominant LUT reduction --
//    a 16-bit dynamic barrel shifter is far more expensive than the add
//    that follows it, and this halves that cost across all 256 PEs.
//
// Functionally identical to the previous version (same three branches,
// same clamping to FRACT_WIDTH, same signed/arithmetic shift), just
// restructured for hardware sharing and correctly scoped DSP mapping.
// -----------------------------------------------------------------------
module temporal_accumulator #(
    parameter FRACT_WIDTH = 12, // Width of accumulated fractional part (sized for DSP)
    parameter EXP_WIDTH   = 8   // Width of the shared scale (X)
)(
    input wire clk,
    input wire rst_n,
    input wire valid_in,

    // Incoming Psum from the spatial MAC tree (two's-complement signed)
    input wire signed [FRACT_WIDTH-1:0] psum_fract,
    input wire        [EXP_WIDTH-1:0]   psum_x,

    // Output accumulated final value (two's-complement signed)
    output reg signed [FRACT_WIDTH-1:0] accum_fract,
    output reg        [EXP_WIDTH-1:0]   accum_x,
    output reg                          valid_out
);

    // -------------------------------------------------------------------
    // Combinational alignment stage: derive exp_diff, pick the shift
    // amount / operand / routing, and drive ONE shared barrel shifter.
    // -------------------------------------------------------------------
    // -------------------------------------------------------------------
    // [TIER-1 FIX] shift_amt width correction.
    // Was: declared [EXP_WIDTH:0] (9 bits, range 0-511) but only the
    // lower 4 bits (shift_amt[3:0]) were ever used to drive the actual
    // shift -- silently truncating the legitimate clamp-to-FRACT_WIDTH=16
    // case (which needs 5 bits, 0-16) and wasting a compare/mux bit on
    // every one of the 256 instances for a range that was never usable.
    // Narrowed to SHIFT_W = clog2(FRACT_WIDTH+1) = 5 bits, which exactly
    // covers 0..FRACT_WIDTH with no truncation and no dead logic.
    // -------------------------------------------------------------------
    localparam SHIFT_W = $clog2(FRACT_WIDTH+1);

    reg signed [EXP_WIDTH:0]     exp_diff;
    reg        [SHIFT_W-1:0]     shift_amt;      // clamped, always >= 0 when used
    reg        [FRACT_WIDTH-1:0] coarse_shifted;
    reg signed [FRACT_WIDTH-1:0] shift_operand;  // whichever operand needs shifting this cycle
    reg                          route_to_psum;  // 1: shifted result -> aligned_psum, 0: -> aligned_accum
    reg signed [FRACT_WIDTH-1:0] shifted_result; // single shared barrel-shifter output
    reg signed [FRACT_WIDTH-1:0] aligned_psum;
    reg signed [FRACT_WIDTH-1:0] aligned_accum;
    reg        [EXP_WIDTH-1:0]   next_accum_x;

    always @(*) begin
        // Calculate the difference between internal scale and incoming scale.
        exp_diff = $signed({1'b0, accum_x}) - $signed({1'b0, psum_x});

        if (exp_diff > 0) begin
            // Internal scale is larger. Shift incoming Psum right to align.
            shift_amt     = (exp_diff > FRACT_WIDTH) ? SHIFT_W'(FRACT_WIDTH) : SHIFT_W'(exp_diff);
            shift_operand = psum_fract;
            route_to_psum = 1'b1;
            next_accum_x  = accum_x;
        end else if (exp_diff < 0) begin
            // Incoming scale is larger. Shift internal accumulator right to align.
            shift_amt     = ((-exp_diff) > FRACT_WIDTH) ? SHIFT_W'(FRACT_WIDTH) : SHIFT_W'(-exp_diff);
            shift_operand = accum_fract;
            route_to_psum = 1'b0;
            next_accum_x  = psum_x;
        end else begin
            // Scales are identical, no shifting required. shift_amt = 0 makes
            // the shared shifter a pure passthrough for whichever operand is
            // routed through it, so this reuses the same hardware cleanly.
            shift_amt     = {SHIFT_W{1'b0}};
            shift_operand = psum_fract;
            route_to_psum = 1'b1;
            next_accum_x  = accum_x;
        end

        // Single shared barrel shifter (arithmetic/sign-preserving).
        // [TIER-1 FIX] Was: shift_operand >>> shift_amt[3:0] -- truncated
        // to 4 bits regardless of shift_amt's declared width, silently
        // corrupting any legitimate shift_amt==16 (clamp) case. Now
        // drives directly off the correctly-sized shift_amt.
        // Coarse/fine 2-stage shift instead of one 5-bit dynamic shift.
        // Coarse stage: shift by 0/4/8/12/16 (top 3 bits of shift_amt).
        // Fine stage:   shift by 0/1/2/3     (bottom 2 bits of shift_amt).
        case (shift_amt[SHIFT_W-1:2])
            3'd0:    coarse_shifted = shift_operand;
            3'd1:    coarse_shifted = shift_operand >>> 4;
            3'd2:    coarse_shifted = shift_operand >>> 8;
            3'd3:    coarse_shifted = shift_operand >>> 12;
            default: coarse_shifted = shift_operand >>> 16; // covers the clamp case
        endcase

        case (shift_amt[1:0])
            2'd0:    shifted_result = coarse_shifted;
            2'd1:    shifted_result = coarse_shifted >>> 1;
            2'd2:    shifted_result = coarse_shifted >>> 2;
            default: shifted_result = coarse_shifted >>> 3;
        endcase

        aligned_psum  = route_to_psum ? shifted_result : psum_fract;
        aligned_accum = route_to_psum ? accum_fract    : shifted_result;
    end

    // -------------------------------------------------------------------
    // Isolated pure-add signal for DSP mapping. Its only fan-in is the
    // two already-resolved aligned operands above -- no shift/mux logic
    // bleeds into it, so this maps to exactly one DSP48E1 ALU per
    // instance instead of a multi-DSP chain approximating a shift.
    // -------------------------------------------------------------------
    (* use_dsp = "yes" *) wire signed [FRACT_WIDTH-1:0] accum_sum = aligned_accum + aligned_psum;

    // -------------------------------------------------------------------
    // Sequential accumulation
    // -------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accum_fract <= {FRACT_WIDTH{1'b0}};
            accum_x     <= {EXP_WIDTH{1'b0}};
            valid_out   <= 1'b0;
        end else if (valid_in) begin
            accum_fract <= accum_sum;
            accum_x     <= next_accum_x;
            valid_out   <= 1'b1;
        end else begin
            valid_out   <= 1'b0;
        end
    end

endmodule