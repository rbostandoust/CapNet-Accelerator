// out_mode, sel1, ld_reg, en_cnt
module Controller(input clk, rst, sq, sc, mat8, mat16, col_sum, eq8, eq16, eq1152, output reg[1:0] out_mode, output reg sel1, ld_reg, en_cnt, done);
	reg [4:0] state, ns;
  	parameter [4:0] idle = 0, sq1 = 1, sq2 = 2, sq3 = 3, sq4 = 4,
  					sc1 = 5, sc2 = 6, sc3 = 7,
  					mat8_1 = 8, mat8_2 = 9,
  					mat16_1 = 10, mat16_2 = 11,
  					col_sum1 = 12, col_sum2 = 13;
  	always @(state) begin
	    { out_mode, sel1, ld_reg, en_cnt } = 5'b0;
	    case (state)
	    ///rstcnt 
	      sq1 : begin out_mode = 2'b10; end
	      sq2 : begin sel1=1; ld_reg=1; end
	      sq3 : begin en_cnt=1; out_mode=2'b01; end
	      sq4 : begin done=1; end 
	      sc1 : begin ld_reg = 1; end
	      sc2 : begin out_mode = 2'b01; en_cnt = 1; end
	      sc3 : begin done = 1; end
	      mat8_1 : begin en_cnt = 1; end
	      mat8_2 : begin done = 1; end
	      mat16_1 : begin en_cnt = 1; end
	      mat16_2 : begin done = 1; end
	      col_sum1 : begin out_mode = 2'b11; en_cnt = 1; end
	      col_sum2 : begin done = 1; end
	      default:  { out_mode, sel1, ld_reg, en_cnt } = 5'b0;
	    endcase
  	end

  	always@(sq, sc, mat8, mat16, col_sum, state, clk) begin
	    ns = idle;
	    case (state)
	      idle: begin if(sq == 1)ns = sq1;else if(sc == 1) ns = sc1;else if(mat8 == 1) ns = mat8_1; else if(mat16 == 1) ns = mat16_1; else if(col_sum==1) ns = col_sum1; else ns = idle; end
	      sq1 :	begin ns = sq2; end
	      sq2 : ns = sq3;
	      sq3 : begin if(eq8 == 1)ns = sq4; else ns= sq3;end
	      sq4 : ns = idle;
	      sc1 : ns = sc2;
	      sc2 : begin if(eq16) ns = sc3; else ns = sc2; end
	      sc3 : ns = idle; 
	      mat8_1 : begin if(eq8) ns = mat8_2; else ns = mat8_1; end
	      mat8_2 : ns = idle;
	      mat16_1 : begin if(eq16) ns = mat8_2; else ns = mat8_1; end
	      mat16_2 : ns = idle;
	      col_sum1: begin if(eq1152) ns = col_sum2; else ns = col_sum1; end
	      col_sum2 : ns = idle;
	    endcase
  	end

  	always@( posedge clk) begin
   		state <= ns;
  	end

endmodule