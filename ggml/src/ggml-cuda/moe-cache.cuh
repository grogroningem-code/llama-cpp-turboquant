#pragma once

// MoE expert cache registration entry point, called from ggml_backend_cuda_reg().
// Registers the provider API with the backend registry (see ggml-backend-moe-cache.h).

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void ggml_cuda_moe_cache_register(void * reg);

// Surrender the device's cache VRAM under allocator pressure; returns bytes freed.
size_t ggml_moe_cache_trim(int device);

// T10.3 prefill-borrow: lend a contiguous run of whole pool slots as prompt
// compute scratch (bytes==0 selects auto sizing: half of the largest pool's
// free bytes). Returns bytes lent (0 = refused); *out_ptr is slab interior
// and is never owned by the caller - return it with ggml_moe_cache_return
// before the slab can be freed.
size_t ggml_moe_cache_lend(int device, size_t bytes, void ** out_ptr);
// Return a range previously lent. Returns true when a loan matched.
bool ggml_moe_cache_return(int device, void * ptr);

#ifdef __cplusplus
}
#endif
