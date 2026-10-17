`timescale 1ns / 1ps

module activation_reorder_buffer #(
    parameter DEPTH = 256,                 // Tile size / cluster size
    parameter ADDR_WIDTH = $clog2(DEPTH),
    parameter DATA_WIDTH = 16              // Width of FP4/FP2 activation block
)(
    input  wire                  clk,
    
    // Bank Selection (Ping-Pong Control)
    input  wire                  bank_sel_wr, // 0: Bank A, 1: Bank B
    input  wire                  bank_sel_rd, // 0: Bank A, 1: Bank B (should be ~bank_sel_wr)

    // Write Port: Driven by main activation SRAM after gather
    input  wire                  we,
    input  wire [ADDR_WIDTH-1:0] waddr,       // sequential cluster-order slot
    input  wire [DATA_WIDTH-1:0] wdata,       // gathered activation data
    
    // Read Port: Driven by the PE array (sequential burst)
    input  wire [ADDR_WIDTH-1:0] raddr,
    output reg  [DATA_WIDTH-1:0] rdata
);

    // Dual banks for ping-pong buffering
    (* ram_style = "block" *)reg [DATA_WIDTH-1:0] bank_a [0:DEPTH-1];
    (* ram_style = "block" *)reg [DATA_WIDTH-1:0] bank_b [0:DEPTH-1];

    // Write Logic
    always @(posedge clk) begin
        if (we) begin
            if (bank_sel_wr == 1'b0)
                bank_a[waddr] <= wdata;
            else
                bank_b[waddr] <= wdata;
        end
    end

    // Read Logic (Synchronous)
    always @(posedge clk) begin
        if (bank_sel_rd == 1'b0)
            rdata <= bank_a[raddr];
        else
            rdata <= bank_b[raddr];
    end

endmodule
