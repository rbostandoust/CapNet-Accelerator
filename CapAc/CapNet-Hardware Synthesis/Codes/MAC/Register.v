module Register#(parameter SIZE = 19) (input clk, rst, input[SIZE:0] in, input ld, output reg[SIZE:0] out);
	always @(posedge clk or posedge rst) begin
		if (rst) begin
			// reset
			out = 0;
		end
		else if (clk) begin
			if(ld)
				out = in;
		end
	end
endmodule