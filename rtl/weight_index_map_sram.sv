`timescale 1ns / 1ps

// -----------------------------------------------------------------------
// weight_index_map_sram.v  (CORRECTED)
//
// FIX: added a "configured" latch plus a simulation-only self-check.
// This directly targets the failure mode your own RTL plan flagged: a
// permutation port that's silently tied off or never loaded per layer --
// the same category of bug as the FFT project's stage_mask stuck at
// 11'h001. If this SRAM is ever READ before it has been WRITTEN at least
// once since reset, it now flags a warning in simulation instead of
// quietly handing back garbage/zero permutation data that then gets
// treated as a valid index by weight_gather_ctrl.
//
// Two new ports were required to support the check:
//   rst_n : needed to clear the "configured" latch at reset/layer boundary
//   re    : explicit read-enable, so "no read this cycle" can be told
//           apart from "reading garbage this cycle"
// weight_gather_ctrl.v (also corrected in this drop) now drives `re` via
// its new idx_map_re output.
// -----------------------------------------------------------------------
module weight_index_map_sram #(
    parameter DEPTH = 1024,                  // reduction_len
    parameter ADDR_WIDTH = $clog2(DEPTH)     // ceil(log2(reduction_len))
)(
    input  wire                  clk,
    input  wire                  rst_n,      // NEW: drives the self-check latch

    // Write Port: Used during layer configuration (static load)
    input  wire                  we,
    input  wire [ADDR_WIDTH-1:0] waddr,      // slot
    input  wire [ADDR_WIDTH-1:0] wdata,      // orig_reduction_idx

    // Read Port: Used dynamically by weight_gather_ctrl during compute
    input  wire                  re,         // NEW: explicit read-enable
    input  wire [ADDR_WIDTH-1:0] raddr,      // slot
    output reg  [ADDR_WIDTH-1:0] rdata       // orig_reduction_idx
);

    (* ram_style = "block" *)reg [ADDR_WIDTH-1:0] mem [0:DEPTH-1];

    // Latches high the first time a permutation table is loaded after
    // reset. This only guards the "never loaded even once" case -- if
    // rst_n is pulsed at each layer boundary in your top level, that's
    // exactly the case that matters (a new layer starting compute before
    // its own perm table has been staged).
    reg configured;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            configured <= 1'b0;
        end else if (we) begin
            configured <= 1'b1;
        end
    end

    always @(posedge clk) begin
        if (we) begin
            mem[waddr] <= wdata;
        end
        // Synchronous read
        rdata <= mem[raddr];
    end

    // ---------------------------------------------------------------
    // Simulation-only self-check. `ifndef SYNTHESIS is the conventional
    // guard most synthesis tools (Vivado, Genus, Yosys w/ appropriate
    // define) respect; $display is non-synthesizable regardless, so this
    // block disappears from the netlist even without the guard, but the
    // guard keeps lint/elaboration clean.
    // ---------------------------------------------------------------
    `ifndef SYNTHESIS
    always @(posedge clk) begin
        if (re && !configured) begin
            $display("%0t [ASSERT-FAIL] weight_index_map_sram: raddr=%0d read before any write since reset -- permutation table looks tied off or never loaded for this layer.", $time, raddr);
        end
    end
    `endif

endmodule
