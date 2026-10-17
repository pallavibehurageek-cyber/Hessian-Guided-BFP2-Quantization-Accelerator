`timescale 1ns / 1ps

module output_compression_pairwise #(
    parameter FRACT_WIDTH = 16,
    parameter EXP_WIDTH   = 8
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          valid_in,
    input  wire                          enc_type,  // 0: FP2-E1M0, 1: FP2-E0M1

    // Inputs from Post Processing (PARALLEL PAIR)[cite: 4]
    input  wire signed [FRACT_WIDTH-1:0] in_fract_0,
    input  wire signed [FRACT_WIDTH-1:0] in_fract_1,
    input  wire        [EXP_WIDTH-1:0]   in_x,

    // Compressed pair output -- one valid_out pulse per single valid_in pulse
    output reg        pair_s,    // shared sign[cite: 4]
    output reg        pair_val,  // shared exponent-or-mantissa bit[cite: 4]
    output reg [1:0]  pair_c,    // shared zero-state flags[cite: 4]
    output reg [EXP_WIDTH-1:0] pair_x, // shared block scale, pass-through[cite: 4]
    output reg        valid_out
);

    localparam integer TOP_BIT  = FRACT_WIDTH - 2; 
    localparam integer LZ_WIDTH = $clog2(FRACT_WIDTH);

    // =========================================================================
    // COMBINATORIAL PATH FOR ELEMENT 0
    // =========================================================================
    wire sign_bit_0 = in_fract_0[FRACT_WIDTH-1];
    wire [FRACT_WIDTH-1:0] abs_fract_0 = sign_bit_0 ? (~in_fract_0 + 1'b1) : in_fract_0;
    wire is_zero_0  = (abs_fract_0 == {FRACT_WIDTH{1'b0}});

    reg [LZ_WIDTH-1:0] lead_pos_0;
    reg                found_0;
    integer li0;
    always @(*) begin
        lead_pos_0 = {LZ_WIDTH{1'b0}};
        found_0    = 1'b0;
        for (li0 = TOP_BIT; li0 >= 0; li0 = li0 - 1) begin
            if (!found_0 && abs_fract_0[li0]) begin
                lead_pos_0 = li0[LZ_WIDTH-1:0];
                found_0    = 1'b1;
            end
        end
    end

    wire [LZ_WIDTH-1:0] norm_shift_0 = found_0 ? (TOP_BIT[LZ_WIDTH-1:0] - lead_pos_0) : {LZ_WIDTH{1'b0}};
    // -------------------------------------------------------------------
    // [TIER-1 FIX] Was: full FRACT_WIDTH-bit variable left-shift
    // (abs_fract_0 << norm_shift_0) computed solely to read back ONE bit
    // (shifted_fract_0[TOP_BIT-1]). shifted_fract_0[TOP_BIT-1] is
    // mathematically identical to abs_fract_0[lead_pos_0-1] (the bit one
    // position below the detected leading one) -- so no shifter is
    // needed at all, only an indexed bit-select (15:1 mux instead of a
    // full barrel shifter). Removes ~256 full-width shifters overall
    // (128 pairs x 2 elements).
    // -------------------------------------------------------------------
    wire [LZ_WIDTH-1:0] bit_sel_0 = lead_pos_0 - 1'b1;
    wire val0 = (enc_type == 1'b0) ? (norm_shift_0 == 0)
              : (found_0 && (lead_pos_0 != 0)) ? abs_fract_0[bit_sel_0] : 1'b0;

    // =========================================================================
    // COMBINATORIAL PATH FOR ELEMENT 1
    // =========================================================================
    wire sign_bit_1 = in_fract_1[FRACT_WIDTH-1];
    wire [FRACT_WIDTH-1:0] abs_fract_1 = sign_bit_1 ? (~in_fract_1 + 1'b1) : in_fract_1;
    wire is_zero_1  = (abs_fract_1 == {FRACT_WIDTH{1'b0}});

    reg [LZ_WIDTH-1:0] lead_pos_1;
    reg                found_1;
    integer li1;
    always @(*) begin
        lead_pos_1 = {LZ_WIDTH{1'b0}};
        found_1    = 1'b0;
        for (li1 = TOP_BIT; li1 >= 0; li1 = li1 - 1) begin
            if (!found_1 && abs_fract_1[li1]) begin
                lead_pos_1 = li1[LZ_WIDTH-1:0];
                found_1    = 1'b1;
            end
        end
    end

    wire [LZ_WIDTH-1:0] norm_shift_1 = found_1 ? (TOP_BIT[LZ_WIDTH-1:0] - lead_pos_1) : {LZ_WIDTH{1'b0}};
    // [TIER-1 FIX] Same bit-select replacement as element 0 above.
    wire [LZ_WIDTH-1:0] bit_sel_1 = lead_pos_1 - 1'b1;
    wire val1 = (enc_type == 1'b0) ? (norm_shift_1 == 0)
              : (found_1 && (lead_pos_1 != 0)) ? abs_fract_1[bit_sel_1] : 1'b0;

    // =========================================================================
    // PAIR MERGE LOGIC (Clocked)
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pair_s    <= 1'b0;
            pair_val  <= 1'b0;
            pair_c    <= 2'b00;
            pair_x    <= {EXP_WIDTH{1'b0}};
            valid_out <= 1'b0;
        end else begin
            valid_out <= valid_in; 

            if (valid_in) begin
                // 11=both nonzero, 10=only val0 nonzero, 01=only val1 nonzero, 00=both zero[cite: 4]
                pair_c <= {~is_zero_0, ~is_zero_1};

                if (is_zero_0 && is_zero_1) begin
                    pair_s   <= 1'b0;
                    pair_val <= 1'b0;
                end else if (is_zero_0 && !is_zero_1) begin
                    pair_s   <= sign_bit_1;
                    pair_val <= val1;
                end else if (!is_zero_0 && is_zero_1) begin
                    pair_s   <= sign_bit_0;
                    pair_val <= val0;
                end else begin
                    // Both nonzero: sign of the first nonzero value[cite: 4]
                    pair_s   <= sign_bit_0;
                    
                    // shared bit merges per encoding type[cite: 4]
                    pair_val <= (enc_type == 1'b0) ? (val0 | val1) : ((val0 + val1) >= 1); 
                end

                pair_x <= in_x;
            end
        end
    end

endmodule