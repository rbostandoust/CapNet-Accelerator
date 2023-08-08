module MAC(clk, rst, in1, in2, out, out_mode, sel_mux4, sel1, ld_reg, en_cnt, eq8, eq16, eq1152);
	parameter IN_SIZE = 19;
	parameter OUT_SIZE = 23;
	input clk;
	input rst;
	input [IN_SIZE:0] in1;
	input [IN_SIZE:0] in2;
	output [OUT_SIZE:0] out;
	input [1:0] out_mode, sel_mux4;
	input sel1;
	input ld_reg;
	input en_cnt;
	output eq8, eq16, eq1152;

	wire [2*IN_SIZE:0] mult_out;
	wire [OUT_SIZE:0] sum_out;
	reg [OUT_SIZE:0] reg_out;
	reg [OUT_SIZE:0] zeros = 0;
	reg [OUT_SIZE:0] b = 24'd1;
	reg [IN_SIZE:0] a = 20'd2;
	wire [OUT_SIZE:0] mux_2_out;
	wire [IN_SIZE:0] mux_1_out;
	wire [IN_SIZE:0] mux_3_out;
	wire [IN_SIZE:0] my_reg_out;
	// wire [15:0] my_cnt_out;

	//##################MUXs##################
	MUX_3 mux_a (0, 1, 2, 3, sel_mux4, a);
	MUX_3 mux_b (0, 1, 2, 3, sel_mux4, b);
	MUX mux_2 (reg_out, zeros, b, reg_out, out_mode, mux_2_out);
	MUX #(IN_SIZE) mux_1 (in2, my_reg_out, a, 1, out_mode, mux_1_out);
	MUX_2 mux_3 (in2, reg_out[IN_SIZE:0], sel1, mux_3_out);

	//##################Registers#############
	Register my_reg(clk, rst, mux_3_out, ld_reg, my_reg_out );

	//#################Counter################
	// (input clk, rst, en, output reg [15:0] out);
	// Counter my_counter(clk, rst, en_cnt, my_cnt_out);

	//################Comparators#############
	// assign eq8 = (my_cnt_out == 16'd8) ? 1 : 0;
	// assign eq16 = (my_cnt_out == 16'd16) ? 1 : 0;
	// assign eq1152 = (my_cnt_out == 16'd1152) ? 1 : 0;

	assign mult_out = in1* mux_1_out;
	assign sum_out = mux_2_out + mult_out ;
	assign out = reg_out;

	always @(posedge clk or posedge rst) begin
		if (rst) begin
			// reset
			reg_out = 0;
			
		end
		else if (clk) begin
			reg_out = sum_out;
			
		end
	end
endmodule