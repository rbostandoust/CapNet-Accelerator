module MAC_tb();
	parameter IN_SIZE = 19;
	parameter OUT_SIZE = 23;
	reg clk;
	reg rst;
	reg [IN_SIZE:0] in1;
	reg [IN_SIZE:0] in2;
	reg sq, sc, mat8, mat16, col_sum;
	wire [OUT_SIZE:0] out;
	wire done;
	// MAC my_mac(clk, rst, in1, in2, out, out_mode);
	Top_Module top_mod(clk, rst, sq, sc, mat8, mat16, col_sum, in1, in2,  out, done);
	initial begin
	    clk = 1'b0;
	    rst = 1'b1;
	    repeat(4) #10 clk = ~clk;
	    rst = 1'b0;
	    mat8 = 1'b1;
	    repeat(4) #10 clk = ~clk;
	    mat8 = 1'b0;
	    forever #10 clk = ~clk;
  	end
	initial begin
	    in1 = 20'd2; // initial value
	    in2 = 20'd2;
	    @(negedge rst); // wait for reset
	    // compare = 8'd128;
	    repeat(256) @(posedge clk) begin in1 = in1 + 1; in2 = in2 + 1; end
	    $stop;
  	end
endmodule

// module Top_Module(input clk, rst, sq, sc, mat8, mat16, col_sum, input[19:0] in1, in2, output[23:0] out, output done);