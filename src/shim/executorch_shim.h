/*
 * extern "C" bridge over ExecuTorch's C++ Module API, so Zig can call it.
 * v1 scope: float32, int64, int32, bool tensors; portable CPU kernels only.
 */
#ifndef LIB_EXECUTORCH_SHIM_H
#define LIB_EXECUTORCH_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ETModule ETModule;
typedef struct ETResult ETResult;

/* Mirrors the subset of executorch::aten::ScalarType this shim supports. */
typedef enum {
  ET_DTYPE_FLOAT = 0,
  ET_DTYPE_INT64 = 1,
  ET_DTYPE_INT32 = 2,
  ET_DTYPE_BOOL = 3,
} ETScalarType;

typedef struct {
  /* Points to caller-owned, tightly-packed elements of the type named by
   * `dtype` (float / int64_t / int32_t / bool). Not copied; must outlive the
   * et_module_forward() call. */
  const void* data;
  const int32_t* sizes;
  int32_t rank;
  int32_t dtype; /* ETScalarType */
} ETTensor;

/* Call before any other et_* call. Creates the global thread pool on the
 * first call: num_threads == 0 keeps ExecuTorch's default (performance
 * cores), otherwise exactly num_threads. XNNPACK keeps the pool pointer from
 * load() on, so the count is fixed after the first call: a later call with a
 * different nonzero count returns InvalidState (0x02), one with 0 or the same
 * count returns 0. Builds without the thread pool only accept 0 or 1 and
 * return NotSupported (0x10) otherwise. Thread safe. */
int32_t et_runtime_init(uint32_t num_threads);

/* Thread count of the global pool (1 in builds without it). Only valid after
 * a successful et_runtime_init(). */
uint32_t et_runtime_num_threads(void);

/* path must outlive the call (copied internally). Never null. */
ETModule* et_module_create(const char* path);
/* Same, reading the program from memory (BufferDataLoader, no copy): data
 * must stay valid and unchanged until et_module_destroy(). Never null. */
ETModule* et_module_create_from_buffer(const void* data, size_t len);
void et_module_destroy(ETModule* m);

/* Returns an ExecuTorch Error code; 0 is Ok. */
int32_t et_module_load(ETModule* m);

/* Validates `inputs` against the "forward" method's expected input count,
 * rank, and dtype (queried from the loaded program's metadata) without
 * running the method (loads the program first if needed). Returns 0 if all
 * inputs match, a positive ExecuTorch Error code if the program/method
 * metadata can't be loaded, or a negative sentinel on mismatch:
 *   -2: n_inputs does not match the method's expected input count
 *   -3: an input's tag is not Tensor
 *   -4: an input's rank or dtype does not match
 * Call this before et_module_forward() to convert malformed input shape/dtype
 * into a caught error instead of undefined behavior inside the kernel. */
int32_t et_module_validate_inputs(ETModule* m, const ETTensor* inputs, int32_t n_inputs);

/* Metadata of the "forward" method's inputs, read from the program (loads it
 * first if needed). Both return an ExecuTorch Error code; 0 is Ok.
 * et_module_input_info: InvalidArgument for an out-of-range index,
 * InvalidType for a non-tensor input. *dtype is -1 for a tensor of a type
 * outside ETScalarType. *sizes points into the program and stays valid until
 * et_module_destroy(). */
int32_t et_module_num_inputs(ETModule* m, int32_t* n_inputs);
int32_t et_module_input_info(ETModule* m, int32_t index, int32_t* dtype, const int32_t** sizes, int32_t* rank);

/* inputs/n_inputs describe caller-owned tensors (not copied, must outlive the
 * call). Returns an opaque result; check et_result_error() before reading
 * outputs. Must be freed with et_result_destroy(). */
ETResult* et_module_forward(ETModule* m, const ETTensor* inputs, int32_t n_inputs);

int32_t et_result_error(const ETResult* r);
int32_t et_result_num_outputs(const ETResult* r);
/* For an out-of-range index or a non-tensor output: rank -1, dtype -1,
 * sizes/data null. dtype is also -1 for a tensor of an unsupported type. */
int32_t et_result_output_rank(const ETResult* r, int32_t index);
const int32_t* et_result_output_sizes(const ETResult* r, int32_t index);
int32_t et_result_output_dtype(const ETResult* r, int32_t index); /* ETScalarType */
const void* et_result_output_data(const ETResult* r, int32_t index);
void et_result_destroy(ETResult* r);

#ifdef __cplusplus
}
#endif

#endif
