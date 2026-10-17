`timescale 1ns / 1ps

// -----------------------------------------------------------------------
// hessian_bfp2_accelerator_top.sv
//
// Companion to fp2_accelerator_top.sv (the paper-faithful 16x16 systolic
// grid, unchanged, used for spatial/general-mode weights at full 256x
// parallelism). This top is for Hessian/k-means-grouped weight data
// specifically, where that parallelism is structurally unavailable (see
// design note in the accompanying message) -- it runs at the fastest rate
// that data's single shared, magnitude-sorted stream allows: one weight
// block (16 pairs) per cycle, every cycle.
//
// Everything except the channel-routing path is REUSED, UNCHANGED:
//   - fp2_block_decode_wrapper / fp2_processing_element: paper's exact
//     per-pair FP2 decode/multiply (Sec. IV-E/F)
//   - temporal_accumulator: paper's exact dynamic-alignment accumulation
//     (Sec. IV-C), now addressed by channel instead of fixed PE position
//   - main activation gather path (weight_index_map_sram [used for
//     activation permutation, per its original role] / weight_gather_ctrl /
//     activation_reorder_buffer / fp2_sram): unchanged, single instance
//     now instead of 16x row-replicated, since only one weight/activation
//     stream is meaningful for k-means data (see design note)
//   - post_processing_unit / output_block_compactor / output_compression_pairwise:
//     unchanged, paper-spec one-shared-exponent-per-32-element-block output path
//
// NEW (the k-means addition, nothing else):
//   - weight_lane_channel_map: load-time decode of each pair's original
//     output channel from the k-means permutation
//   - channel_scatter_accumulator: routes each pair's result to its real
//     channel instead of assuming all 16 pairs of a block share one
//
// KNOWN, DOCUMENTED LIMITATIONS (not silently glossed over):
//  1. Pair-level routing, not per-element: since fp2_processing_element's
//     pair encoding is kept unchanged (per explicit instruction), and 99.6%
//     of pairs span two different channels, each pair routes by its
//     element-0 channel; element-1's contribution in mixed pairs rides
//     along attributed to element 0's channel. Documented in
//     weight_lane_channel_map.sv.
//  2. Activation gather is still BLOCK-granularity (one packed 32-element
//     read per gather step, no sub-block lane selection, no zero-padding
//     injection) -- true im2col addressing for a real convolution is NOT
//     implemented here. This testbench/top can exercise the full
//     datapath and the new k-means routing correctly, but does not by
//     itself constitute a verified-correct convolution result against the
//     golden output -- that still requires the im2col engine discussed
//     earlier and not yet built.
// -----------------------------------------------------------------------
module hessian_bfp2_accelerator_top #(
    parameter FRACT_WIDTH      = 16,
    parameter EXP_WIDTH        = 8,
    parameter REDUCTION_LEN    = 32,
    parameter TILE_SIZE        = 32,
    parameter MEM_RD_LATENCY   = 2,
    parameter RAW_ACT_WIDTH    = 64,
    parameter WEIGHT_DEPTH     = 18432,
    parameter ADDR_WIDTH       = 16,
    parameter NUM_CHANNELS     = 256,
    parameter PAIRS_PER_BLOCK  = 16,
    parameter IDX_WIDTH        = 20   // raw weight-permutation index width
)(
    input  wire clk,
    input  wire rst_n,
    input  wire enc_type,
    input  wire en_relu,

    // Activation load (single instance -- see design note)
    input  wire                             idxmap_we,
    input  wire [$clog2(REDUCTION_LEN)-1:0] idxmap_waddr,
    input  wire [$clog2(REDUCTION_LEN)-1:0] idxmap_wdata,
    input  wire                             mact_we,
    input  wire [$clog2(REDUCTION_LEN)-1:0] mact_waddr,
    input  wire [RAW_ACT_WIDTH-1:0]         mact_wdata,
    input  wire [EXP_WIDTH-1:0]             mact_xdata,

    // Weight load (linear, unchanged)
    input  wire                             wt_we,
    input  wire [$clog2(WEIGHT_DEPTH)-1:0]  wt_waddr,
    input  wire [63:0]                      wt_wdata,
    input  wire                             wtx_we,
    input  wire [$clog2(WEIGHT_DEPTH)-1:0]  wtx_waddr,
    input  wire [EXP_WIDTH-1:0]             wtx_wdata,

    // NEW: weight-lane channel map load (the k-means permutation)
    input  wire                             wlcm_we,
    input  wire [$clog2(WEIGHT_DEPTH)-1:0]  wlcm_waddr,
    input  wire [IDX_WIDTH-1:0]             wlcm_raw_idx [0:PAIRS_PER_BLOCK-1],

    input  wire                       start_gather,
    input  wire                       compute_en,
    input  wire [ADDR_WIDTH-1:0]      base_compute_addr,
    output wire                       layer_gather_done,

    // Flat 128-pair output (channel 2k / 2k+1 -> pair k), matching the
    // existing golden-hex pairwise-block convention
    output wire                out_pair_s     [0:NUM_CHANNELS/2-1],
    output wire                out_pair_val   [0:NUM_CHANNELS/2-1],
    output wire [1:0]          out_pair_c     [0:NUM_CHANNELS/2-1],
    output wire [EXP_WIDTH-1:0] out_pair_x    [0:NUM_CHANNELS/2-1],
    output wire                out_pair_valid [0:NUM_CHANNELS/2-1]
);

    localparam IDX_ADDR_WIDTH  = $clog2(REDUCTION_LEN);
    localparam TILE_ADDR_WIDTH = $clog2(TILE_SIZE);
    localparam WT_ADDR_WIDTH   = $clog2(WEIGHT_DEPTH);
    localparam ACT_PAYLOAD_WIDTH = RAW_ACT_WIDTH + EXP_WIDTH;

    // -------------------------------------------------------------------
    // Activation gather path -- unchanged modules, single instance.
    // Identical structure to fp2_accelerator_top.sv's per-row generate
    // block, just not replicated 16x, since one shared stream is all
    // k-means weight data can use (see top-of-file note).
    // -------------------------------------------------------------------
    wire                      idx_map_re;
    wire [IDX_ADDR_WIDTH-1:0] idx_map_raddr;
    wire [IDX_ADDR_WIDTH-1:0] idx_map_rdata;

    weight_index_map_sram #(.DEPTH(REDUCTION_LEN)) u_idx_map (
        .clk(clk), .rst_n(rst_n),
        .we(idxmap_we), .waddr(idxmap_waddr), .wdata(idxmap_wdata),
        .re(idx_map_re), .raddr(idx_map_raddr), .rdata(idx_map_rdata)
    );

    wire                        mact_rd_en;
    wire [IDX_ADDR_WIDTH-1:0]   mact_raddr;
    wire [ACT_PAYLOAD_WIDTH-1:0] mact_rdata;
    wire                        mact_rvalid;

    fp2_sram #(.WIDTH(ACT_PAYLOAD_WIDTH), .DEPTH(REDUCTION_LEN), .RD_LATENCY(MEM_RD_LATENCY)) u_main_act_sram (
        .clk(clk), .rst_n(rst_n),
        .we(mact_we), .waddr(mact_waddr), .wdata({mact_xdata, mact_wdata}),
        .re(mact_rd_en), .raddr(mact_raddr), .rdata(mact_rdata), .rvalid(mact_rvalid)
    );

    wire                          bank_sel_wr;
    wire                          gather_done, layer_done;
    wire [TILE_ADDR_WIDTH-1:0]    reorder_waddr;
    wire                          reorder_we;
    wire [ACT_PAYLOAD_WIDTH-1:0]  reorder_wdata;
    wire                          tile_consumed_ack;

    weight_gather_ctrl #(
        .REDUCTION_LEN(REDUCTION_LEN), .TILE_SIZE(TILE_SIZE),
        .MEM_RD_LATENCY(MEM_RD_LATENCY), .DATA_WIDTH(ACT_PAYLOAD_WIDTH)
    ) u_gather_ctrl (
        .clk(clk), .rst_n(rst_n), .start_gather(start_gather),
        .tile_consumed_ack(tile_consumed_ack), .bank_sel_wr(bank_sel_wr),
        .gather_done(gather_done), .layer_done(layer_done),
        .idx_map_re(idx_map_re), .idx_map_raddr(idx_map_raddr), .idx_map_rdata(idx_map_rdata),
        .main_act_raddr(mact_raddr), .main_act_rd_en(mact_rd_en), .main_act_rdata(mact_rdata),
        .reorder_waddr(reorder_waddr), .reorder_we(reorder_we), .reorder_wdata(reorder_wdata)
    );
    assign layer_gather_done = layer_done;

    wire                          bank_sel_rd = ~bank_sel_wr;
    wire [ACT_PAYLOAD_WIDTH-1:0]  pack_rdata;
    wire                          weight_sram_rd_en;
    wire [ADDR_WIDTH-1:0]         weight_sram_raddr;
    wire                          reorder_buf_rd_en;
    wire [TILE_ADDR_WIDTH-1:0]    reorder_buf_raddr;

    memory_address_router #(
        .ADDR_WIDTH(ADDR_WIDTH), .TILE_SIZE(TILE_SIZE),
        .BLOCK_SIZE(TILE_SIZE), .MEM_RD_LATENCY(MEM_RD_LATENCY)
    ) u_addr_router (
        .clk(clk), .rst_n(rst_n), .compute_en(compute_en), .base_compute_addr(base_compute_addr),
        .gather_act_rd_en(mact_rd_en), .gather_act_raddr({{(ADDR_WIDTH-IDX_ADDR_WIDTH){1'b0}}, mact_raddr}),
        .weight_sram_rd_en(weight_sram_rd_en), .weight_sram_raddr(weight_sram_raddr),
        .reorder_buf_rd_en(reorder_buf_rd_en), .reorder_buf_raddr(reorder_buf_raddr)
    );

    activation_reorder_buffer #(.DEPTH(TILE_SIZE), .DATA_WIDTH(ACT_PAYLOAD_WIDTH)) u_reorder_buf (
        .clk(clk), .bank_sel_wr(bank_sel_wr), .bank_sel_rd(bank_sel_rd),
        .we(reorder_we), .waddr(reorder_waddr), .wdata(reorder_wdata),
        .raddr(reorder_buf_raddr), .rdata(pack_rdata)
    );

    reg [63:0]           act_data_align;
    reg [EXP_WIDTH-1:0]  act_x_align;
    reg                  act_valid_align;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            act_data_align <= 64'd0; act_x_align <= {EXP_WIDTH{1'b0}}; act_valid_align <= 1'b0;
        end else begin
            act_data_align  <= pack_rdata[RAW_ACT_WIDTH-1:0];
            act_x_align     <= pack_rdata[ACT_PAYLOAD_WIDTH-1:RAW_ACT_WIDTH];
            act_valid_align <= reorder_buf_rd_en;
        end
    end

    reg [TILE_ADDR_WIDTH-1:0] prev_raddr;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) prev_raddr <= 0; else prev_raddr <= reorder_buf_raddr;
    end
    assign tile_consumed_ack = reorder_buf_rd_en & (reorder_buf_raddr == 0) & (prev_raddr != 0);

    // -------------------------------------------------------------------
    // Weight path: linear read (unchanged) + NEW channel map (parallel read,
    // same address, same cadence)
    // -------------------------------------------------------------------
    wire [63:0]          wt_rdata;
    wire [EXP_WIDTH-1:0] wtx_rdata;
    wire                 wt_rvalid, wtx_rvalid;

    fp2_sram #(.WIDTH(64), .DEPTH(WEIGHT_DEPTH), .RD_LATENCY(MEM_RD_LATENCY)) u_weight_sram (
        .clk(clk), .rst_n(rst_n), .we(wt_we), .waddr(wt_waddr), .wdata(wt_wdata),
        .re(weight_sram_rd_en), .raddr(weight_sram_raddr[WT_ADDR_WIDTH-1:0]), .rdata(wt_rdata), .rvalid(wt_rvalid)
    );
    fp2_sram #(.WIDTH(EXP_WIDTH), .DEPTH(WEIGHT_DEPTH), .RD_LATENCY(MEM_RD_LATENCY)) u_weight_x_sram (
        .clk(clk), .rst_n(rst_n), .we(wtx_we), .waddr(wtx_waddr), .wdata(wtx_wdata),
        .re(weight_sram_rd_en), .raddr(weight_sram_raddr[WT_ADDR_WIDTH-1:0]), .rdata(wtx_rdata), .rvalid(wtx_rvalid)
    );

    wire [EXP_WIDTH-1:0] wlcm_channel_id [0:PAIRS_PER_BLOCK-1];
    wire                 wlcm_rvalid;

    weight_lane_channel_map #(
        .WEIGHT_DEPTH(WEIGHT_DEPTH), .PAIRS_PER_BLOCK(PAIRS_PER_BLOCK),
        .IDX_WIDTH(IDX_WIDTH), .CH_WIDTH(EXP_WIDTH), .RD_LATENCY(MEM_RD_LATENCY)
    ) u_wlcm (
        .clk(clk), .rst_n(rst_n),
        .we(wlcm_we), .waddr(wlcm_waddr), .raw_idx_pair0(wlcm_raw_idx),
        .re(weight_sram_rd_en), .raddr(weight_sram_raddr[WT_ADDR_WIDTH-1:0]),
        .channel_id(wlcm_channel_id), .rvalid(wlcm_rvalid)
    );

    // -------------------------------------------------------------------
    // NEW: the scatter-accumulate compute engine (replaces the systolic
    // grid for this data path). act_data_align/act_x_align/wt_rdata/
    // wtx_rdata/wlcm_channel_id all share the same MEM_RD_LATENCY-cycle
    // alignment already established by fp2_accelerator_top.sv.
    // -------------------------------------------------------------------
    wire [FRACT_WIDTH-1:0] accum_fract [0:NUM_CHANNELS-1];
    wire [EXP_WIDTH-1:0]   accum_x     [0:NUM_CHANNELS-1];
    wire                   accum_valid [0:NUM_CHANNELS-1];

    channel_scatter_accumulator #(
        .FRACT_WIDTH(FRACT_WIDTH), .EXP_WIDTH(EXP_WIDTH),
        .NUM_PAIRS(PAIRS_PER_BLOCK), .NUM_CHANNELS(NUM_CHANNELS), .CH_WIDTH(EXP_WIDTH)
    ) u_scatter (
        .clk(clk), .rst_n(rst_n), .valid_in(act_valid_align && wt_rvalid && wlcm_rvalid),
        .enc_type(enc_type),
        .act_data_in(act_data_align), .act_x_in(act_x_align),
        .wt_data_in(wt_rdata), .wt_x_in(wtx_rdata),
        .channel_id(wlcm_channel_id),
        .accum_fract(accum_fract), .accum_x(accum_x), .accum_valid(accum_valid)
    );

    // -------------------------------------------------------------------
    // Output path -- unchanged modules, reindexed from a flat 256-channel
    // array instead of a [16][16] grid. 128 pairs -> 8 groups of 16 pairs
    // (32 channels) for the paper-spec shared block exponent.
    // -------------------------------------------------------------------
    localparam BLOCK_PAIRS = 16;
    genvar g, pp, oc;
    generate
        for (g = 0; g < (NUM_CHANNELS/2) / BLOCK_PAIRS; g = g + 1) begin : out_group

            wire signed [FRACT_WIDTH-1:0] pp_fract_0 [0:BLOCK_PAIRS-1];
            wire signed [FRACT_WIDTH-1:0] pp_fract_1 [0:BLOCK_PAIRS-1];
            wire        [EXP_WIDTH-1:0]   pp_x_local [0:BLOCK_PAIRS-1];
            wire                          pp_valid   [0:BLOCK_PAIRS-1];
            wire signed [FRACT_WIDTH-1:0] cmp_fract_0 [0:BLOCK_PAIRS-1];
            wire signed [FRACT_WIDTH-1:0] cmp_fract_1 [0:BLOCK_PAIRS-1];
            wire [EXP_WIDTH-1:0]          block_x;
            wire                          block_valid;

            for (pp = 0; pp < BLOCK_PAIRS; pp = pp + 1) begin : pair
                localparam integer PAIR_IDX = g * BLOCK_PAIRS + pp; // 0..127
                localparam integer CH0 = 2 * PAIR_IDX;
                localparam integer CH1 = 2 * PAIR_IDX + 1;

                post_processing_unit #(.FRACT_WIDTH(FRACT_WIDTH), .EXP_WIDTH(EXP_WIDTH)) u_pp (
                    .clk(clk), .rst_n(rst_n), .valid_in(accum_valid[CH0]),
                    .psum_fract_0(accum_fract[CH0]), .psum_fract_1(accum_fract[CH1]),
                    .psum_x_0(accum_x[CH0]), .psum_x_1(accum_x[CH1]),
                    .en_relu(en_relu), .bias_fract({FRACT_WIDTH{1'b0}}), .bias_x({EXP_WIDTH{1'b0}}),
                    .pp_fract_0(pp_fract_0[pp]), .pp_fract_1(pp_fract_1[pp]),
                    .pp_x(pp_x_local[pp]), .valid_out(pp_valid[pp])
                );
            end

            output_block_compactor #(.FRACT_WIDTH(FRACT_WIDTH), .EXP_WIDTH(EXP_WIDTH), .NUM_PAIRS(BLOCK_PAIRS)) u_compactor (
                .clk(clk), .rst_n(rst_n),
                .valid_in(pp_valid), .in_fract_0(pp_fract_0), .in_fract_1(pp_fract_1), .in_x(pp_x_local),
                .out_fract_0(cmp_fract_0), .out_fract_1(cmp_fract_1), .block_x(block_x), .valid_out(block_valid)
            );

            for (oc = 0; oc < BLOCK_PAIRS; oc = oc + 1) begin : out_comp
                localparam integer PAIR_IDX2 = g * BLOCK_PAIRS + oc;
                output_compression_pairwise #(.FRACT_WIDTH(FRACT_WIDTH), .EXP_WIDTH(EXP_WIDTH)) u_out_comp (
                    .clk(clk), .rst_n(rst_n), .valid_in(block_valid), .enc_type(enc_type),
                    .in_fract_0(cmp_fract_0[oc]), .in_fract_1(cmp_fract_1[oc]), .in_x(block_x),
                    .pair_s(out_pair_s[PAIR_IDX2]), .pair_val(out_pair_val[PAIR_IDX2]),
                    .pair_c(out_pair_c[PAIR_IDX2]), .pair_x(out_pair_x[PAIR_IDX2]),
                    .valid_out(out_pair_valid[PAIR_IDX2])
                );
            end
        end
    endgenerate

endmodule
