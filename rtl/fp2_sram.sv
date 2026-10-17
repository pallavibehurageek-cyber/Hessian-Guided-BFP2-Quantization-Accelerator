`timescale 1ns / 1ps

// -----------------------------------------------------------------------
// fp2_sram.v
//
// Generic single-port-read / single-port-write synchronous SRAM model
// used everywhere in the FP2 accelerator that needs physical storage:
// weight banks, weight shared-scale (X) banks, main (un-reordered)
// activation storage, and the activation shared-scale storage.
// weight_index_map_sram.v and activation_reorder_buffer.v are already
// hand-written with their own dedicated port lists elsewhere in this
// project, so this module is deliberately NOT used for those two -- it
// fills the "everything else needs a physical memory and none exists
// yet" gap.
//
// RD_LATENCY is an explicit parameter (not an implicit pipeline depth)
// for the same reason weight_gather_ctrl.v made MEM_RD_LATENCY explicit:
// a mismatch between the *assumed* read latency in the controller that
// consumes `rdata` and the *actual* latency of the memory it reads from
// is exactly the bug class that caused repeated SQNR regressions in the
// FFT project (stage_mask / MEM_RD_LATENCY). Any module instantiating
// fp2_sram must pass RD_LATENCY through to whatever counts cycles
// waiting on `rdata` (see fp2_accelerator_top.v).
//
// This is a behavioral model for simulation and early synthesis
// exploration only. Before tapeout this should be swapped for a real
// hard macro (e.g. an OpenRAM-generated sram_<depth>x<width>_1rw.v, as
// was done for the FFT project's memory.v) with matching port names/
// timing re-verified against the vendor datasheet -- do not assume this
// behavioral model's timing survives that swap unexamined.
// -----------------------------------------------------------------------
module fp2_sram #(
    parameter WIDTH       = 64,
    parameter DEPTH       = 1024,
    parameter ADDR_WIDTH  = $clog2(DEPTH),
    parameter RD_LATENCY  = 1,      // cycles from `re` asserted to `rdata` valid
    parameter READ_FIRST  = 1'b1    // 1: read-old-data-on-collision, 0: write-first (new data forwarded to rdata)
)(
    input  wire                  clk,
    input  wire                  rst_n,

    // Write port
    input  wire                  we,
    input  wire [ADDR_WIDTH-1:0] waddr,
    input  wire [WIDTH-1:0]      wdata,

    // Read port
    input  wire                  re,
    input  wire [ADDR_WIDTH-1:0] raddr,
    output wire [WIDTH-1:0]      rdata,
    output wire                  rvalid    // re, delayed by RD_LATENCY -- companion valid strobe for rdata
);

    (* ram_style = "block" *)reg [WIDTH-1:0] mem [0:DEPTH-1];

    // ---------------------------------------------------------------
    // Core array access (1-cycle intrinsic synchronous read, like a
    // real single-port SRAM macro). Extra RD_LATENCY-1 stages are added
    // below via a pipeline to model deeper macros (matches the FFT
    // project precedent of 2-cycle vs 1-cycle macro latency).
    // ---------------------------------------------------------------
    reg [WIDTH-1:0] core_rdata;
    reg             core_rvalid;

    generate
        if (READ_FIRST) begin : g_read_first
            always @(posedge clk) begin
                if (we) mem[waddr] <= wdata;
                if (re) core_rdata <= mem[raddr];
                core_rvalid <= re;
            end
        end else begin : g_write_first
            always @(posedge clk) begin
                if (we) mem[waddr] <= wdata;
                if (re) begin
                    core_rdata <= (we && (waddr == raddr)) ? wdata : mem[raddr];
                end
                core_rvalid <= re;
            end
        end
    endgenerate

    // ---------------------------------------------------------------
    // Extra pipeline stages so total latency == RD_LATENCY. When
    // RD_LATENCY == 1 this generate block is empty and core_rdata /
    // core_rvalid drive the outputs directly.
    // ---------------------------------------------------------------
    generate
        if (RD_LATENCY <= 1) begin : g_no_extra_latency
            assign rdata  = core_rdata;
            assign rvalid = core_rvalid;
        end else begin : g_extra_latency
            reg [WIDTH-1:0] extra_data  [0:RD_LATENCY-2];
            reg             extra_valid [0:RD_LATENCY-2];
            integer k;
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (k = 0; k < RD_LATENCY-1; k = k + 1) begin
                        extra_data[k]  <= {WIDTH{1'b0}};
                        extra_valid[k] <= 1'b0;
                    end
                end else begin
                    extra_data[0]  <= core_rdata;
                    extra_valid[0] <= core_rvalid;
                    for (k = 1; k < RD_LATENCY-1; k = k + 1) begin
                        extra_data[k]  <= extra_data[k-1];
                        extra_valid[k] <= extra_valid[k-1];
                    end
                end
            end
            assign rdata  = extra_data[RD_LATENCY-2];
            assign rvalid = extra_valid[RD_LATENCY-2];
        end
    endgenerate

    // ---------------------------------------------------------------
    // Simulation-only self-check, same pattern as
    // weight_index_map_sram.v: flag a read of an address that was
    // never written since reset, instead of silently handing back
    // whatever garbage/zero the behavioral array happens to contain.
    // This guards the "SRAM tied off / never loaded for this layer"
    // failure mode at every instantiation site (weights, main
    // activations, shared-scale buffers) without needing a bespoke
    // check re-written per site.
    // ---------------------------------------------------------------
    `ifndef SYNTHESIS
    reg written_since_reset [0:DEPTH-1];
    integer j;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (j = 0; j < DEPTH; j = j + 1) written_since_reset[j] = 1'b0;
        end else begin
            if (we) written_since_reset[waddr] <= 1'b1;
            if (re && !written_since_reset[raddr]) begin
                $display("%0t [ASSERT-FAIL] fp2_sram(%m): raddr=%0d read before any write since reset.", $time, raddr);
            end
        end
    end
    `endif

endmodule
