module Counter(input clk, rst, en, output reg [15:0] out);
	always @(posedge clk or posedge rst) begin
		if (rst) begin
			// reset
			out = 16'd0;
		end
		else if (clk) begin
			if(en)
				out = out + 1;
		end
	end
endmodule