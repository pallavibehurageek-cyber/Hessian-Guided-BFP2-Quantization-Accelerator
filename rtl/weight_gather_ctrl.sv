`timescale 1ns / 1ps

// -----------------------------------------------------------------------
// weight_gather_ctrl.sv (CLOSED-LOOP BACKPRESSURE UPGRADE)
//
// 1. FIRE-AND-FORGET: 1-cycle `start_gather` trigger.
// 2. LATENCY DECOUPLING: Auto-computes TOTAL_LATENCY.
// 3. CLOSED-LOOP FLOW CONTROL: Uses `tile_consumed_ack` to track 
//    `available_slots` in the ping-pong buffer, pausing SRAM reads 
//    if the downstream Compute Array stalls.
// -----------------------------------------------------------------------
module weight_gather_ctrl #(
    parameter REDUCTION_LEN   = 1024,
    parameter TILE_SIZE       = 256,
    parameter MEM_RD_LATENCY  = 2,
    parameter DATA_WIDTH      = 16,
    parameter IDX_ADDR_WIDTH  = $clog2(REDUCTION_LEN),
    parameter TILE_ADDR_WIDTH = $clog2(TILE_SIZE)
)(
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       start_gather,      // 1-cycle trigger pulse
    input  wire                       tile_consumed_ack, // 1-cycle pulse from Compute Array when bank is freed

    // Status / Control
    output reg                        bank_sel_wr,
    output reg                        gather_done,       // pulses once per completed TILE_SIZE chunk
    output reg                        layer_done,        // pulses once REDUCTION_LEN elements are written

    // 1. Interface to weight_index_map_sram
    output reg                        idx_map_re,
    output reg  [IDX_ADDR_WIDTH-1:0]  idx_map_raddr,
    input  wire [IDX_ADDR_WIDTH-1:0]  idx_map_rdata,

    // 2. Interface to Main Activation SRAM
    output reg  [IDX_ADDR_WIDTH-1:0]  main_act_raddr,
    output reg                        main_act_rd_en,
    input  wire [DATA_WIDTH-1:0]      main_act_rdata, 

    // 3. Interface to Activation Reorder Buffer
    output reg  [TILE_ADDR_WIDTH-1:0] reorder_waddr,
    output reg                        reorder_we,
    output reg  [DATA_WIDTH-1:0]      reorder_wdata
);

    localparam TOTAL_LATENCY = MEM_RD_LATENCY + 1;
    localparam MAX_CAPACITY  = 2 * TILE_SIZE;

    reg [IDX_ADDR_WIDTH:0] counter; 
    reg is_gathering;

    // Credit-based flow control tracker
    reg [$clog2(MAX_CAPACITY):0] available_slots;

    // Pipeline Stages
    reg [IDX_ADDR_WIDTH-1:0] slot_pipe_stage0;
    reg                      valid_req;
    
    reg [IDX_ADDR_WIDTH-1:0] slot_pipe_stage1;
    reg                      valid_pipe_stage1;

    reg [TILE_ADDR_WIDTH-1:0] slot_delay  [0:TOTAL_LATENCY-1];
    reg                       valid_delay [0:TOTAL_LATENCY-1];

    reg [IDX_ADDR_WIDTH:0] write_count;
    integer d;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            idx_map_re        <= 1'b0;
            idx_map_raddr     <= {IDX_ADDR_WIDTH{1'b0}};
            main_act_raddr    <= {IDX_ADDR_WIDTH{1'b0}};
            main_act_rd_en    <= 1'b0;
            reorder_waddr     <= {TILE_ADDR_WIDTH{1'b0}};
            reorder_we        <= 1'b0;
            reorder_wdata     <= {DATA_WIDTH{1'b0}};
            
            counter           <= 0;
            write_count       <= 0;
            is_gathering      <= 1'b0;
            bank_sel_wr       <= 1'b0;
            gather_done       <= 1'b0;
            layer_done        <= 1'b0;
            
            available_slots   <= MAX_CAPACITY;

            slot_pipe_stage0  <= 0;
            valid_req         <= 1'b0;
            slot_pipe_stage1  <= 0;
            valid_pipe_stage1 <= 1'b0;

            for (d = 0; d < TOTAL_LATENCY; d = d + 1) begin
                slot_delay[d]  <= 0;
                valid_delay[d] <= 1'b0;
            end
        end else begin
            gather_done <= 1'b0; 
            layer_done  <= 1'b0; 

            // --- State Management ---
            if (start_gather) begin
                is_gathering <= 1'b1;
            end else if (counter == REDUCTION_LEN) begin
                is_gathering <= 1'b0; 
            end

            // --- Credit / Flow Control Management ---
            // valid_req consumes 1 slot. tile_consumed_ack frees TILE_SIZE slots.
            if (layer_done) begin
                available_slots <= MAX_CAPACITY; // Auto-reset for next layer
            end else begin
                case ({valid_req, tile_consumed_ack})
                    2'b10: available_slots <= available_slots - 1'b1;
                    2'b01: available_slots <= available_slots + TILE_SIZE;
                    2'b11: available_slots <= available_slots + TILE_SIZE - 1'b1;
                    default: available_slots <= available_slots;
                endcase
            end

            // --- Stage 0: Issue read to Index Map (WITH BACKPRESSURE) ---
            if ((start_gather || is_gathering) && counter < REDUCTION_LEN && available_slots > 0) begin
                idx_map_raddr    <= counter[IDX_ADDR_WIDTH-1:0];
                idx_map_re       <= 1'b1;
                slot_pipe_stage0 <= counter[IDX_ADDR_WIDTH-1:0];
                valid_req        <= 1'b1;
                counter          <= counter + 1;
            end else begin
                idx_map_re <= 1'b0;
                valid_req  <= 1'b0;
            end

            // --- Stage 1: Receive Index, Issue read to Main Act SRAM ---
            valid_pipe_stage1 <= valid_req;
            slot_pipe_stage1  <= slot_pipe_stage0;

            if (valid_pipe_stage1) begin
                main_act_raddr <= idx_map_rdata; 
                main_act_rd_en <= 1'b1;
            end else begin
                main_act_rd_en <= 1'b0;
            end

            // --- Delay line: wait EXACTLY for SRAM latency ---
            slot_delay[0]  <= slot_pipe_stage1[TILE_ADDR_WIDTH-1:0]; 
            valid_delay[0] <= valid_pipe_stage1;
            for (d = 1; d < TOTAL_LATENCY; d = d + 1) begin
                slot_delay[d]  <= slot_delay[d-1];
                valid_delay[d] <= valid_delay[d-1];
            end

            // --- Final stage: Write into reorder buffer ---
            reorder_we <= valid_delay[TOTAL_LATENCY-1];
            if (valid_delay[TOTAL_LATENCY-1]) begin
                reorder_waddr <= slot_delay[TOTAL_LATENCY-1];
                reorder_wdata <= main_act_rdata;
                write_count   <= write_count + 1;

                if (write_count == REDUCTION_LEN - 1) begin
                    layer_done  <= 1'b1;
                    write_count <= 0;
                    counter     <= 0;
                end
            end

            // --- Tile Boundary & Bank Flip (Fixed Timing) ---
            if (reorder_we && reorder_waddr == (TILE_SIZE[TILE_ADDR_WIDTH-1:0]-1'b1)) begin
                gather_done <= 1'b1;
                bank_sel_wr <= ~bank_sel_wr;
            end
        end
    end

endmodule
