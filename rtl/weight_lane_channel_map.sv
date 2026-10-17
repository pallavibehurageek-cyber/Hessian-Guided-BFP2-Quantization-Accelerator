`timescale 1ns / 1ps

// -----------------------------------------------------------------------
// weight_lane_channel_map.sv
//
// The ONLY new addition this represents: recording which output channel
// each pair of a Hessian-shuffled weight block actually belongs to, so it
// can be routed correctly. The FP2 encoding/decoding itself
// (fp2_block_decode_wrapper / fp2_processing_element) is untouched.
//
// Per pair, channel_id = (original flat weight index of that pair's FIRST
// element) / 2304 (2304 = in_ch*kh*kw = 256*3*3 for this layer's conv).
// NOTE (explicit, not hidden): measured on the actual weight_groups.txt,
// 99.6% of pairs have their two elements spanning two DIFFERENT channels.
// Using element 0's channel as the pair's routing destination is a
// deliberate, documented simplification -- element 1's contribution rides
// along attributed to element 0's channel in those mixed cases. This is
// the tradeoff of keeping the paper's pair-level encoding untouched
// (which was the explicit ask) rather than decomposing to per-element
// routing.
//
// Division by 2304 happens ONLY here, at weight-load time (one calculation
// per pair per block, ~18432*16 times total across the whole load phase,
// nowhere near the per-cycle compute path), using Verilog's constant-divisor
// division -- synthesis tools handle a fixed constant divisor like this
// efficiently; this is not in any per-cycle timing-critical path.
//
// SYNTHESIS FIX: the storage array was originally declared as a 2D
// unpacked array-of-arrays (`mem [0:DEPTH-1][0:PAIRS-1]`). Every other
// memory in this design (fp2_sram, weight_index_map_sram,
// activation_reorder_buffer) uses a single flat packed-word array instead
// -- the pattern real memory compilers/BRAM inference reliably recognize
// as "one memory." A 2D array here risked either failed BRAM inference
// (falling back to distributed RAM/registers -- disastrous at this
// module's actual size, ~18432 x 128b = ~288KB, the single largest memory
// in the whole k-means engine) or splitting into many small BRAMs instead
// of one. Flattened to match the rest of the design.
// -----------------------------------------------------------------------
module weight_lane_channel_map #(
    parameter WEIGHT_DEPTH = 18432,
    parameter PAIRS_PER_BLOCK = 16,
    parameter IDX_WIDTH = 20,   // raw flat weight index width (0..589823 fits in 20 bits)
    parameter CH_WIDTH  = 8,    // channel id width (0..255)
    parameter ELEMS_PER_CHANNEL = 2304, // in_ch*kh*kw for THIS layer -- see note below
    // Must match the RD_LATENCY of whatever weight SRAM this map is read
    // in lockstep with (u_weight_sram/u_weight_x_sram, both fp2_sram instances).
    // A mismatch here is exactly what caused the block-0 dropout caught by
    // tb_fp2_kmeans_datapath.sv: with only 1 cycle of latency here vs. fp2_sram's
    // 2, this module's rvalid came up a cycle early, and the very first block's
    // scatter_valid_in (act_valid_align && wt_rvalid && wlcm_rvalid) never
    // asserted before wt_rvalid caught up -- silently dropping block 0 entirely.
    parameter RD_LATENCY = 2
)(
    input  wire clk,
    input  wire rst_n,

    // Write port: one block's worth of raw indices per write, PAIRS_PER_BLOCK*2
    // values (32 for this layer), same cadence as loading weight_data.hex.
    input  wire                          we,
    input  wire [$clog2(WEIGHT_DEPTH)-1:0] waddr,
    input  wire [IDX_WIDTH-1:0]          raw_idx_pair0 [0:PAIRS_PER_BLOCK-1], // element 0 of each pair

    // Read port: decoded channel id per pair, same address as weight_sram_raddr
    input  wire                          re,
    input  wire [$clog2(WEIGHT_DEPTH)-1:0] raddr,
    output wire [CH_WIDTH-1:0]           channel_id [0:PAIRS_PER_BLOCK-1],
    output wire                          rvalid
);
    // ELEMS_PER_CHANNEL is a per-layer constant (in_ch*kh*kw), NOT re-derived
    // from anything else in this module -- it must match whatever conv this
    // block of vectors came from. Documented explicitly rather than silently
    // assumed, since getting this wrong silently misroutes every pair.

    localparam ROW_WIDTH = PAIRS_PER_BLOCK * CH_WIDTH;

    // FIXED: single flat packed-word memory, BRAM-inference-safe.
    reg [ROW_WIDTH-1:0] mem [0:WEIGHT_DEPTH-1];

    integer p;
    always @(posedge clk) begin
        if (we) begin
            for (p = 0; p < PAIRS_PER_BLOCK; p = p + 1) begin
                mem[waddr][p*CH_WIDTH +: CH_WIDTH] <= raw_idx_pair0[p] / ELEMS_PER_CHANNEL;
            end
        end
    end

    // Core 1-cycle synchronous read (same shape as fp2_sram's core stage)
    reg [ROW_WIDTH-1:0] core_data;
    reg                 core_rvalid;
    always @(posedge clk) begin
        if (re) core_data <= mem[raddr];
        core_rvalid <= re;
    end

    // Extra pipeline stages so total latency == RD_LATENCY, identical
    // structure/convention to fp2_sram.sv's g_extra_latency block -- now
    // flat words throughout, not per-lane 2D arrays.
    wire [ROW_WIDTH-1:0] final_data;
    generate
        if (RD_LATENCY <= 1) begin : g_no_extra_latency
            assign final_data = core_data;
            assign rvalid = core_rvalid;
        end else begin : g_extra_latency
            reg [ROW_WIDTH-1:0] extra_data [0:RD_LATENCY-2];
            reg                 extra_valid [0:RD_LATENCY-2];
            integer k;
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (k = 0; k < RD_LATENCY-1; k = k + 1) begin
                        extra_valid[k] <= 1'b0;
                        extra_data[k]  <= {ROW_WIDTH{1'b0}};
                    end
                end else begin
                    extra_valid[0] <= core_rvalid;
                    extra_data[0]  <= core_data;
                    for (k = 1; k < RD_LATENCY-1; k = k + 1) begin
                        extra_valid[k] <= extra_valid[k-1];
                        extra_data[k]  <= extra_data[k-1];
                    end
                end
            end
            assign final_data = extra_data[RD_LATENCY-2];
            assign rvalid = extra_valid[RD_LATENCY-2];
        end
    endgenerate

    genvar gp;
    generate
        for (gp = 0; gp < PAIRS_PER_BLOCK; gp = gp + 1) begin : g_unpack
            assign channel_id[gp] = final_data[gp*CH_WIDTH +: CH_WIDTH];
        end
    endgenerate

endmodule
