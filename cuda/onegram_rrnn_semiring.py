ONEGRAM_RRNN_SEMIRING = """
            
extern "C" {
     __global__ void rrnn_semiring_fwd(
                const float * __restrict__ u, 
                const float * __restrict__ c1_init,
                const int len, 
                const int batch,
                const int dim,
                const int k,
                float * __restrict__ c1,
                int semiring_type) {
        int ncols = batch*dim;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;
        int ncols_u = ncols*k;

        const float *up = u + (col*k);
        float *c1p = c1 + col;
        float cur_c1 = *(c1_init + col);
        
        for (int row = 0; row < len; ++row) {
            float u1 = *(up);
            float forget1 = *(up+1);
            
            float op1 = times_forward(semiring_type, cur_c1, forget1);
            cur_c1 = plus_forward(semiring_type, op1, u1);
            
            *c1p = cur_c1;
            
            up += ncols_u;
            c1p += ncols;
        }
    }
    
    __global__ void rrnn_semiring_bwd(
                const float * __restrict__ u, 
                const float * __restrict__ c1_init,
                const float * __restrict__ c1,
                const float * __restrict__ grad_c1, 
                const float * __restrict__ grad_last_c1,
                const int len, 
                const int batch, 
                const int dim, 
                const int k,
                float * __restrict__ grad_u, 
                float * __restrict__ grad_c1_init,
                int semiring_type) {
        int ncols = batch*dim;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;

        int ncols_u = ncols*k;

        float cur_c1 = *(grad_last_c1 + col);

        const float *up = u + (col*k) + (len-1)*ncols_u;
        const float *c1p = c1 + col + (len-1)*ncols;
        
        const float *gc1p = grad_c1 + col + (len-1)*ncols;
        float *gup = grad_u + (col*k) + (len-1)*ncols_u;
        
        
        // cur_c1 = cur_c1 * forget1 + u1;
            
        for (int row = len-1; row >= 0; --row) {
            float u1 = *(up);
            float forget1 = *(up+1);
            
            const float prev_c1 = (row>0) ? (*(c1p-ncols)) : (*(c1_init+col));
            
            const float gc1 = *(gc1p) + cur_c1;
            
            
            cur_c1 = 0.f;
            float op1 = times_forward(semiring_type, prev_c1, forget1);
            float gop1 = 0.f, gu1 = 0.f;
            plus_backward(semiring_type, op1, u1, gc1, gop1, gu1);
            float gprev_c1 = 0.f, gforget1=0.f;
            times_backward(semiring_type, prev_c1, forget1, gop1, gprev_c1, gforget1);
            *(gup) = gu1;
            *(gup+1) = gforget1;
            cur_c1 += gprev_c1; 

            up -= ncols_u; 
            c1p -= ncols;
            gup -= ncols_u;
            gc1p -= ncols;
        }
        
        *(grad_c1_init + col) = cur_c1;
    }
}
"""
