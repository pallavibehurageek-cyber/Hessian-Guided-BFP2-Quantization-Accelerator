`timescale 1ns / 1ps

module memory_address_router #(
    parameter ADDR_WIDTH     = 16,
    parameter TILE_SIZE      = 256,
    parameter BLOCK_SIZE     = 32,   
    parameter MEM_RD_LATENCY = 2     
)(
    input  wire                         clk,
    input  wire                         rst_n,

    // --- Inputs from Global Control / Compute FSM ---
    input  wire                         compute_en,
    input  wire [ADDR_WIDTH-1:0]        base_compute_addr,

    // --- Inputs from Weight Gather Controller ---
    input  wire                         gather_act_rd_en,
    input  wire [ADDR_WIDTH-1:0]        gather_act_raddr,

    // --- Outputs to Physical SRAM Macros ---
    // With the act_block_packer bypassed, weights are issued immediately
    output wire                         weight_sram_rd_en,
    output wire [ADDR_WIDTH-1:0]        weight_sram_raddr,

    // 2. Main Activation SRAM (driven exclusively by the gather FSM)
    output wire                         main_act_sram_rd_en,
    output wire [ADDR_WIDTH-1:0]        main_act_sram_raddr,

    // --- Outputs to Activation Reorder Buffer ---
    output wire                         reorder_buf_rd_en,
    output wire [$clog2(TILE_SIZE)-1:0] reorder_buf_raddr
);

    // Issue weight read immediately
    assign weight_sram_rd_en = compute_en;
    assign weight_sram_raddr = base_compute_addr;

    // Activation Gather Path (K-Means Indirection) -- unchanged
    assign main_act_sram_rd_en  = gather_act_rd_en;
    assign main_act_sram_raddr  = gather_act_raddr;

    // Activation Compute Path -- reorder buffer sequential read
    assign reorder_buf_rd_en  = compute_en;
    assign reorder_buf_raddr  = base_compute_addr[$clog2(TILE_SIZE)-1:0];

endmodule