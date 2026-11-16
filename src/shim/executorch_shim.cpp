#include "executorch_shim.h"

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/platform/runtime.h>

#ifdef ET_USE_THREADPOOL
#include <executorch/extension/threadpool/threadpool.h>
#endif

#include <cstring>
#include <mutex>
#include <string>
#include <vector>

/* Header-only FPU helpers (get/set_fpu_state, disable_fpu_denormals). Kept
 * last: it defines global min()/max() helpers. Included by relative path so
 * builds without XNNPACK (no pthreadpool include dir) still find it. */
#include "../../pthreadpool-fork/src/threadpool-utils.h"

using executorch::aten::ScalarType;
using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::runtime::EValue;
using executorch::runtime::Error;
using executorch::runtime::Result;
using executorch::runtime::Tag;
using executorch::runtime::runtime_init;

namespace {

/* Returns Ok on a supported/recognized dtype; the underlying kernels only
 * exercise these four in this shim's v1 scope. */
Result<ScalarType> to_scalar_type(int32_t dtype) {
  switch (dtype) {
    case ET_DTYPE_FLOAT:
      return ScalarType::Float;
    case ET_DTYPE_INT64:
      return ScalarType::Long;
    case ET_DTYPE_INT32:
      return ScalarType::Int;
    case ET_DTYPE_BOOL:
      return ScalarType::Bool;
    default:
      return Error::InvalidArgument;
  }
}

int32_t from_scalar_type(ScalarType type) {
  switch (type) {
    case ScalarType::Float:
      return ET_DTYPE_FLOAT;
    case ScalarType::Long:
      return ET_DTYPE_INT64;
    case ScalarType::Int:
      return ET_DTYPE_INT32;
    case ScalarType::Bool:
      return ET_DTYPE_BOOL;
    default:
      /* Unreachable for outputs of a v1-scope model; no negative dtype
       * value exists, so surface a sentinel a Zig caller can never mistake
       * for a supported type. */
      return -1;
  }
}

}  // namespace

struct ETModule {
  Module module;
  explicit ETModule(const char* path) : module(std::string(path)) {}
  ETModule(const void* data, size_t len)
      : module(std::make_unique<executorch::extension::BufferDataLoader>(data, len)) {}
};

struct ETResult {
  Result<std::vector<EValue>> result;
  explicit ETResult(Result<std::vector<EValue>>&& r) : result(std::move(r)) {}
};

namespace {
std::mutex runtime_init_mutex;
uint32_t runtime_threads = 0; /* 0 until the first successful init. */
}  // namespace

int32_t et_runtime_init(uint32_t num_threads) {
  std::lock_guard<std::mutex> lock{runtime_init_mutex};
  if (runtime_threads != 0) {
    if (num_threads != 0 && num_threads != runtime_threads) return static_cast<int32_t>(Error::InvalidState);
    return 0;
  }
  runtime_init();
#ifdef ET_USE_THREADPOOL
  /* No Module exists yet (the Zig side needs this call's token to make one),
   * so nothing holds the pool pointer and resetting it here is safe. */
  auto* pool = executorch::extension::threadpool::get_threadpool();
  if (pool == nullptr) return static_cast<int32_t>(Error::Internal);
  if (num_threads != 0) pool->_unsafe_reset_threadpool(num_threads);
  runtime_threads = static_cast<uint32_t>(pool->get_thread_count());
#else
  if (num_threads > 1) return static_cast<int32_t>(Error::NotSupported);
  runtime_threads = 1;
#endif
  return 0;
}

uint32_t et_runtime_num_threads(void) { return runtime_threads; }

ETModule* et_module_create(const char* path) { return new ETModule(path); }

ETModule* et_module_create_from_buffer(const void* data, size_t len) { return new ETModule(data, len); }

void et_module_destroy(ETModule* m) { delete m; }

int32_t et_module_load(ETModule* m) {
  return static_cast<int32_t>(
      m->module.load(executorch::runtime::Program::Verification::InternalConsistency));
}

int32_t et_module_validate_inputs(ETModule* m, const ETTensor* inputs, int32_t n_inputs) {
  auto meta = m->module.method_meta("forward");
  if (!meta.ok()) return static_cast<int32_t>(meta.error());
  if (meta->num_inputs() != static_cast<size_t>(n_inputs)) return -2;
  for (int32_t i = 0; i < n_inputs; ++i) {
    auto tag = meta->input_tag(static_cast<size_t>(i));
    if (!tag.ok() || *tag != Tag::Tensor) return -3;
    auto info = meta->input_tensor_meta(static_cast<size_t>(i));
    if (!info.ok()) return -3;
    auto want_type = to_scalar_type(inputs[i].dtype);
    if (!want_type.ok()) return -4;
    if (info->scalar_type() != *want_type) return -4;
    if (static_cast<int32_t>(info->sizes().size()) != inputs[i].rank) return -4;
    for (int32_t d = 0; d < inputs[i].rank; ++d) {
      if (info->sizes()[d] != inputs[i].sizes[d]) return -4;
    }
  }
  return 0;
}

int32_t et_module_num_inputs(ETModule* m, int32_t* n_inputs) {
  auto meta = m->module.method_meta("forward");
  if (!meta.ok()) return static_cast<int32_t>(meta.error());
  *n_inputs = static_cast<int32_t>(meta->num_inputs());
  return 0;
}

int32_t et_module_input_info(ETModule* m, int32_t index, int32_t* dtype, const int32_t** sizes, int32_t* rank) {
  auto meta = m->module.method_meta("forward");
  if (!meta.ok()) return static_cast<int32_t>(meta.error());
  if (index < 0) return static_cast<int32_t>(Error::InvalidArgument);
  auto tag = meta->input_tag(static_cast<size_t>(index));
  if (!tag.ok()) return static_cast<int32_t>(tag.error());
  if (*tag != Tag::Tensor) return static_cast<int32_t>(Error::InvalidType);
  auto info = meta->input_tensor_meta(static_cast<size_t>(index));
  if (!info.ok()) return static_cast<int32_t>(info.error());
  *dtype = from_scalar_type(info->scalar_type());
  *sizes = info->sizes().data();
  *rank = static_cast<int32_t>(info->sizes().size());
  return 0;
}

ETResult* et_module_forward(ETModule* m, const ETTensor* inputs, int32_t n_inputs) {
  std::vector<EValue> args;
  args.reserve(static_cast<size_t>(n_inputs));
  // TensorPtrs must outlive the forward() call; keep them alive in `holders`.
  std::vector<executorch::extension::TensorPtr> holders;
  holders.reserve(static_cast<size_t>(n_inputs));
  for (int32_t i = 0; i < n_inputs; ++i) {
    std::vector<executorch::aten::SizesType> sizes(
        inputs[i].sizes, inputs[i].sizes + inputs[i].rank);
    auto type = to_scalar_type(inputs[i].dtype);
    if (!type.ok()) return new ETResult(Error::InvalidArgument);
    auto t = from_blob(const_cast<void*>(inputs[i].data), std::move(sizes), *type);
    args.emplace_back(*t);
    holders.push_back(std::move(t));
  }
  /* Denormals make portable kernels orders of magnitude slower (decaying
   * audio tails hit them constantly). Flush them for this call only and give
   * the caller's thread its FPU state back. XNNPACK's pool workers already
   * run with PTHREADPOOL_FLAG_DISABLE_DENORMALS. */
  const struct fpu_state saved = get_fpu_state();
  disable_fpu_denormals();
  auto result = m->module.forward(args);
  set_fpu_state(saved);
  return new ETResult(std::move(result));
}

int32_t et_result_error(const ETResult* r) {
  return static_cast<int32_t>(r->result.error());
}

int32_t et_result_num_outputs(const ETResult* r) {
  if (!r->result.ok()) return 0;
  return static_cast<int32_t>(r->result->size());
}

namespace {
/* Null for an out-of-range index or a non-tensor output (a program may
 * return ints/bools/lists, and the caller can't know without asking). */
const executorch::aten::Tensor* output_tensor(const ETResult* r, int32_t index) {
  if (!r->result.ok() || index < 0 || static_cast<size_t>(index) >= r->result->size()) return nullptr;
  const EValue& v = (*r->result)[static_cast<size_t>(index)];
  return v.isTensor() ? &v.toTensor() : nullptr;
}
}  // namespace

int32_t et_result_output_rank(const ETResult* r, int32_t index) {
  const auto* t = output_tensor(r, index);
  return t ? static_cast<int32_t>(t->dim()) : -1;
}

const int32_t* et_result_output_sizes(const ETResult* r, int32_t index) {
  const auto* t = output_tensor(r, index);
  return t ? t->sizes().data() : nullptr;
}

int32_t et_result_output_dtype(const ETResult* r, int32_t index) {
  const auto* t = output_tensor(r, index);
  return t ? from_scalar_type(t->scalar_type()) : -1;
}

const void* et_result_output_data(const ETResult* r, int32_t index) {
  const auto* t = output_tensor(r, index);
  return t ? t->const_data_ptr() : nullptr;
}

void et_result_destroy(ETResult* r) { delete r; }
