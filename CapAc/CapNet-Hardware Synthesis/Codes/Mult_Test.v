module Mult_Test ( 
  );
  reg [7:0] a [7:0];
  reg [7:0] b [7:0];
  reg [15:0] out [7:0];
  reg [15:0] out2 = 16'd0;
  reg cmd = 1'b1;
  integer i;
  initial begin
    for(i=0; i<8; i=i+1) begin
      a[i] = i;
      b[i] = i+10;
      out[i] = a[i] * b[i];
    end  
    if(cmd == 1) begin
      for (i=0; i<8; i=i+1) 
        out2 = out2 + out[i];
    end
  end
endmodule // multiplier


// module des ();
//   reg [7:0]  mem1;               // reg vector 8-bit wide
//   reg [7:0]  mem2 [0:3];         // 8-bit wide vector array with depth=4
//   reg [15:0] mem3 [0:3][0:1];   // 16-bit wide vector 2D array with rows=4,cols=2
 
//   initial begin
//     int i;
 
//     mem1 = 8'ha9;
//     $display ("mem1 = 0x%0h", mem1);
 
//     mem2[0] = 8'haa;
//     mem2[1] = 8'hbb;
//     mem2[2] = 8'hcc;
//     mem2[3] = 8'hdd;
//     for(i = 0; i < 4; i = i+1) begin
//       $display("mem2[%0d] = 0x%0h", i, mem2[i]);
//     end
 
//     for(int i = 0; i < 4; i += 1) begin
//       for(int j = 0; j < 2; j += 1) begin
//         mem3[i][j] = i + j;
//         $display("mem3[%0d][%0d] = 0x%0h", i, j, mem3[i][j]);
//       end
//     end
//   end
// endmodule
