module MUX_3 #(parameter SIZE = 19)(a, b, c, d, sel, out);//MUX for Linear Regressioin
	// parameter SIZE = 23;
	input [SIZE:0] a, b, c, d;
	input[1:0] sel;
	output reg [SIZE:0] out;
	always @(*) begin
		if(sel == 2'b00)
			out = a;
		else if (sel == 2'b01)
			out = b;
		else if(sel == 2'b10)
			out = c;
		else if(sel == 2'b11)
			out = d;
	end
endmodule