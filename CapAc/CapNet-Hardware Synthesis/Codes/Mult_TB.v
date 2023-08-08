module Mult_TB ();
  reg[7:0][7:0] a, b;
  reg cmd;
  wire[7:0][15:0] out;
  Mult_Test mt(.in_A(a), .in_B(b), .cmd(cmd), .vec_out(out));
  initial begin
    a = 8'd4;
    b = 8'd2;
  end
endmodule // Mult_TB
