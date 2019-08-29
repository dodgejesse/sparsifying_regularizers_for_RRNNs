FOURGRAM_RRNN_SEMIRING = """
            
extern "C" {
     __global__ void rrnn_semiring_fwd(
                const float * __restrict__ u, 
                const float * __restrict__ c1_init,
                const float * __restrict__ c2_init,
                const float * __restrict__ c3_init,
                const float * __restrict__ c4_init,
                const int len, 
                const int batch,
                const int dim,
                const int k,
                float * __restrict__ c1,
                float * __restrict__ c2,
                float * __restrict__ c3,
                float * __restrict__ c4,
                int semiring_type) {
        int ncols = batch*dim;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;
        int ncols_u = ncols*k;

        const float *up = u + (col*k);
        float *c1p = c1 + col;
        float *c2p = c2 + col;
        float *c3p = c3 + col;
        float *c4p = c4 + col;
        float cur_c1 = *(c1_init + col);
        float cur_c2 = *(c2_init + col);
        float cur_c3 = *(c3_init + col);
        float cur_c4 = *(c4_init + col);
        
        for (int row = 0; row < len; ++row) {
            float u1 = *(up);
            float u2 = *(up+1);
            float u3 = *(up+2);
            float u4 = *(up+3);

            float forget1 = *(up+4);
            float forget2 = *(up+5);
            float forget3 = *(up+6);
            float forget4 = *(up+7);
            
            float prev_c1 = cur_c1;
            float prev_c2 = cur_c2;
            float prev_c3 = cur_c3;
            
            // cur_c1 = cur_c1 * forget1 + u1;
            float op1 = times_forward(semiring_type, cur_c1, forget1);
            cur_c1 = plus_forward(semiring_type, op1, u1);
            
            // cur_c2 = cur_c2 * forget2 + prev_c1 * u2;
            float op2 = times_forward(semiring_type, cur_c2, forget2);
            float op3 = times_forward(semiring_type, prev_c1, u2);
            cur_c2 = plus_forward(semiring_type, op2, op3);
            
            // cur_c3 = cur_c3 * forget3 + prev_c2 * u3;
            float op4 = times_forward(semiring_type, cur_c3, forget3);
            float op5 = times_forward(semiring_type, prev_c2, u3);
            cur_c3 = plus_forward(semiring_type, op4, op5);
            
            // cur_c4 = cur_c4 * forget4 + prev_c3 * u4;            
            float op6 = times_forward(semiring_type, cur_c4, forget4);
            float op7 = times_forward(semiring_type, prev_c3, u4);
            cur_c4 = plus_forward(semiring_type, op6, op7);
            
            *c1p = cur_c1;
            *c2p = cur_c2;
            *c3p = cur_c3;
            *c4p = cur_c4;
            
            up += ncols_u;
            c1p += ncols;
            c2p += ncols;
            c3p += ncols;
            c4p += ncols;
        }
    }
    
    __global__ void rrnn_semiring_bwd(
                const float * __restrict__ u, 
                const float * __restrict__ c1_init,
                const float * __restrict__ c2_init,
                const float * __restrict__ c3_init,
                const float * __restrict__ c4_init,
                const float * __restrict__ c1,
                const float * __restrict__ c2,
                const float * __restrict__ c3,
                const float * __restrict__ c4,
                const float * __restrict__ grad_c1, 
                const float * __restrict__ grad_c2, 
                const float * __restrict__ grad_c3, 
                const float * __restrict__ grad_c4, 
                const float * __restrict__ grad_last_c1,
                const float * __restrict__ grad_last_c2,
                const float * __restrict__ grad_last_c3,
                const float * __restrict__ grad_last_c4,
                const int len, 
                const int batch, 
                const int dim, 
                const int k,
                float * __restrict__ grad_u, 
                float * __restrict__ grad_c1_init,
                float * __restrict__ grad_c2_init,
                float * __restrict__ grad_c3_init,
                float * __restrict__ grad_c4_init,
                int semiring_type) {
        int ncols = batch*dim;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;

        int ncols_u = ncols*k;

        float cur_c1 = *(grad_last_c1 + col);
        float cur_c2 = *(grad_last_c2 + col);
        float cur_c3 = *(grad_last_c3 + col);
        float cur_c4 = *(grad_last_c4 + col);

        const float *up = u + (col*k) + (len-1)*ncols_u;
        const float *c1p = c1 + col + (len-1)*ncols;
        const float *c2p = c2 + col + (len-1)*ncols;
        const float *c3p = c3 + col + (len-1)*ncols;
        const float *c4p = c4 + col + (len-1)*ncols;
        
        const float *gc1p = grad_c1 + col + (len-1)*ncols;
        const float *gc2p = grad_c2 + col + (len-1)*ncols;
        const float *gc3p = grad_c3 + col + (len-1)*ncols;
        const float *gc4p = grad_c4 + col + (len-1)*ncols;
        float *gup = grad_u + (col*k) + (len-1)*ncols_u;
        
        for (int row = len-1; row >= 0; --row) {
            float u1 = *(up);
            float u2 = *(up+1);
            float u3 = *(up+2);
            float u4 = *(up+3);
            float forget1 = *(up+4);
            float forget2 = *(up+5);
            float forget3 = *(up+6);
            float forget4 = *(up+7);
            
            const float prev_c1 = (row>0) ? (*(c1p-ncols)) : (*(c1_init+col));
            const float prev_c2 = (row>0) ? (*(c2p-ncols)) : (*(c2_init+col));
            const float prev_c3 = (row>0) ? (*(c3p-ncols)) : (*(c3_init+col));
            const float prev_c4 = (row>0) ? (*(c4p-ncols)) : (*(c4_init+col));
            
            const float gc1 = *(gc1p) + cur_c1;
            const float gc2 = *(gc2p) + cur_c2;
            const float gc3 = *(gc3p) + cur_c3;
            const float gc4 = *(gc4p) + cur_c4;
            
            cur_c1 = cur_c2 = cur_c3 = cur_c4 = 0.f;
            
            // cur_c1 = cur_c1 * forget1 + u1;
            // float op1 = times_forward(semiring_type, cur_c1, forget1);
            // cur_c1 = plus_forward(semiring_type, op1, u1);
            float op1 = times_forward(semiring_type, prev_c1, forget1);
            float gop1 = 0.f, gu1 = 0.f;
            plus_backward(semiring_type, op1, u1, gc1, gop1, gu1);
            float gprev_c1 = 0.f, gprev_c2 = 0.f, gprev_c3 = 0.f, gprev_c4 = 0.f, gforget1 = 0.f;
            times_backward(semiring_type, prev_c1, forget1, gop1, gprev_c1, gforget1);
            *(gup) = gu1;
            *(gup+4) = gforget1;
            cur_c1 += gprev_c1; 
            
            
            //float op2 = times_forward(semiring_type, cur_c2, forget2);
            //float op3 = times_forward(semiring_type, prev_c1, u2);
            //cur_c2 = plus_forward(semiring_type, op2, op3);
            
            float op2 = times_forward(semiring_type, prev_c2, forget2);
            float op3 = times_forward(semiring_type, prev_c1, u2);
            float gop2 = 0.f, gop3 = 0.f;
            plus_backward(semiring_type, op2, op3, gc2, gop2, gop3);

            float gu2 = 0.f, gforget2 = 0.f;
            times_backward(semiring_type, prev_c2, forget2, gop2, gprev_c2, gforget2);
            times_backward(semiring_type, prev_c1, u2, gop3, gprev_c1, gu2);
            *(gup+1) = gu2;
            *(gup+5) = gforget2;
			cur_c2 += gprev_c2; 
            
			// cur_c3 = cur_c3 * forget3 + prev_c2 * u3;
            // float op4 = times_forward(semiring_type, cur_c3, forget3);
            // float op5 = times_forward(semiring_type, prev_c2, u3);
            // cur_c3 = plus_forward(semiring_type, op4, op5);
            
            float op4 = times_forward(semiring_type, prev_c3, forget3);
            float op5 = times_forward(semiring_type, prev_c2, u3);
            float gop4 = 0.f, gop5 = 0.f;
            plus_backward(semiring_type, op4, op5, gc3, gop4, gop5);

            float gu3 = 0.f, gforget3 = 0.f;
            times_backward(semiring_type, prev_c3, forget3, gop4, gprev_c3, gforget3);
            times_backward(semiring_type, prev_c2, u3, gop5, gprev_c2, gu3);
            *(gup+2) = gu3;
            *(gup+6) = gforget3;
			cur_c3 += gprev_c3; 
            
            // float op6 = times_forward(semiring_type, cur_c4, forget4);
            // float op7 = times_forward(semiring_type, prev_c3, u4);
            // cur_c4 = plus_forward(semiring_type, op6, op7);
            float op6 = times_forward(semiring_type, prev_c4, forget4);
            float op7 = times_forward(semiring_type, prev_c3, u4);
            float gop6 = 0.f, gop7 = 0.f;
            plus_backward(semiring_type, op6, op7, gc4, gop6, gop7);

            float gu4 = 0.f, gforget4 = 0.f;
            times_backward(semiring_type, prev_c4, forget4, gop6, gprev_c4, gforget4);
            times_backward(semiring_type, prev_c3, u4, gop7, gprev_c3, gu4);
            *(gup+3) = gu4;
            *(gup+7) = gforget4;
            
            cur_c1 += gprev_c1;
            cur_c2 += gprev_c2;
            cur_c3 += gprev_c3;
            cur_c4 += gprev_c4;

            up -= ncols_u; 
            c1p -= ncols;
            c2p -= ncols;
            c3p -= ncols;
            c4p -= ncols;
            gup -= ncols_u;
            gc1p -= ncols;
            gc2p -= ncols;
            gc3p -= ncols;
            gc4p -= ncols;
        }
        
        *(grad_c1_init + col) = cur_c1;
        *(grad_c2_init + col) = cur_c2;
        *(grad_c3_init + col) = cur_c3;
        *(grad_c4_init + col) = cur_c4;
    }
}
"""
