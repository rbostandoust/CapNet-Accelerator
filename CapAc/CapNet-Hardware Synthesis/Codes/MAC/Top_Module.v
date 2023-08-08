module Top_Module(input clk, rst, sq, sc, mat8, mat16, col_sum, input[19:0] in1, in2, output[23:0] out, output done);

	wire sel1, ld_reg, en_cnt, eq8, eq16, eq1152;
	wire [1:0] out_mode;
	Controller cu(clk, rst, sq, sc, mat8, mat16, col_sum, eq8, eq16, eq1152, out_mode, sel1, ld_reg, en_cnt, done);
	MAC dp(clk, rst, in1, in2, out, out_mode, sel1, ld_reg, en_cnt, eq8, eq16, eq1152);

endmodule