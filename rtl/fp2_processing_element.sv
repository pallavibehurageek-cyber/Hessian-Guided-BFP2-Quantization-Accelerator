`timescale 1ns / 1ps

module fp2_processing_element (
    input wire       enc_type, // 0 for FP2-E1M0, 1 for FP2-E0M1

    input wire       act_s,    // Activation Sign (S)
    input wire       act_val,  // Activation Mantissa/Exponent bit
    input wire [1:0] act_c,    // Activation nonzero flags: {F1_nonzero, F2_nonzero} (or 00 for mismatched signs)

    input wire       wt_s,     // Weight Sign (S)
    input wire       wt_val,   // Weight Mantissa/Exponent bit
    input wire [1:0] wt_c,     // Weight nonzero flags: {W1_nonzero, W2_nonzero} (or 00 for mismatched signs)

    output wire       out_s,   // Result Sign
    output wire [3:0] out_m,   // Result Magnitude, K * (base F*W magnitude)
    output wire       is_zero  // Flag indicating Psum evaluates to zero
);

    // -------------------------------------------------------------------------
    // Step 1: Base Multiplication Logic F*W
    // -------------------------------------------------------------------------
    // Sign of the base product F*W
    wire raw_s = act_s ^ wt_s;

    // Base Magnitude computation (M1, M2, M3)
    wire m3_raw = act_val & wt_val;
    wire m2_raw = act_val ^ wt_val;
    
    wire m1_e0m1 = ~m3_raw;
    wire m1_e1m0 = ~(act_val | wt_val);
    wire m1_raw = (enc_type == 1'b1) ? m1_e0m1 : m1_e1m0;

    wire [2:0] base_m = {m1_raw, m2_raw, m3_raw};

    // -------------------------------------------------------------------------
    // Step 2: K-Factor Decoding (Table II exact implementation)
    // -------------------------------------------------------------------------
    // C flags translate to effective element vectors:
    // 11 -> ( F,  F)
    // 10 -> ( F,  0)
    // 01 -> ( 0,  F)
    // 00 -> ( F, -F)  // Mismatched sign case
    // K is the dot product of the two vectors.
    //
    // [TIER-1 FIX] Replaced the 16-entry `case({act_c,wt_c})` with direct
    // minimized boolean equations derived from the same truth table.
    // Functionally identical; expresses the mutual-exclusivity of the
    // K=0/K=-1/K=2/K=1 cases explicitly instead of relying on the
    // synthesis tool to re-derive it from the case statement, and is
    // replicated x4096 (16 lanes x 256 PEs) so even a small per-instance
    // reduction is worth taking.
    // -------------------------------------------------------------------
    wire is_zero_w  = (wt_c == ~act_c);                              // K = 0 case
    wire c_eq       = (act_c == wt_c);
    wire c_same_val = (act_c[1] == act_c[0]);                         // act_c is 00 or 11
    wire k2_cond    = c_eq & c_same_val & ~is_zero_w;                 // K = 2 case
    wire ksignflip  = (act_c == 2'b00 && wt_c == 2'b01)
                     | (act_c == 2'b01 && wt_c == 2'b00);             // K = -1 case

    wire [1:0] k_mag  = is_zero_w ? 2'd0 : (k2_cond ? 2'd2 : 2'd1);
    wire       k_sign = ksignflip;

    assign is_zero = is_zero_w;

    // -------------------------------------------------------------------------
    // Step 3: Apply K to the base magnitude and sign
    // -------------------------------------------------------------------------
    // Shift magnitude if K=2
    wire [3:0] m_k1 = {1'b0, base_m};
    wire [3:0] m_k2 = {base_m, 1'b0};

    assign out_m = is_zero          ? 4'b0000 :
                   (k_mag == 2'd2)  ? m_k2    :
                                      m_k1;

    // Apply the base sign, XOR'd with the k_sign flag to handle K = -1
    assign out_s = is_zero ? 1'b0 : (raw_s ^ k_sign);

endmodule