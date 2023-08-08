module MUX_2 #(parameter SIZE = 19)(a, b, sel, out);
	input [SIZE:0] a, b;
	input sel;
	output reg [SIZE:0] out;
	always @(*) begin
		if(sel == 1'b0)
			out = a;
		else if (sel == 1'b1)
			out = b;
	end
endmodule