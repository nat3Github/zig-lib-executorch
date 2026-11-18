//! Zig bindings for the ExecuTorch C++ runtime, via a thin extern "C" shim
//! (src/shim/executorch_shim.cpp) over ExecuTorch's `Module` API.
//!
//! v1 scope: float32/int64/int32/bool tensors, portable (reference) CPU kernels.
const std = @import("std");

const c = struct {
    const ETModule = opaque {};
    const ETResult = opaque {};

    const ET_DTYPE_FLOAT = 0;
    const ET_DTYPE_INT64 = 1;
    const ET_DTYPE_INT32 = 2;
    const ET_DTYPE_BOOL = 3;

    const ETTensor = extern struct {
        data: ?*const anyopaque,
        sizes: [*c]const i32,
        rank: i32,
        dtype: i32,
    };

    extern fn et_runtime_init(num_threads: u32) i32;
    extern fn et_runtime_num_threads() u32;
    extern fn et_module_create(path: [*:0]const u8) ?*ETModule;
    extern fn et_module_create_from_buffer(data: [*]const u8, len: usize) ?*ETModule;
    extern fn et_module_destroy(m: *ETModule) void;
    extern fn et_module_load(m: *ETModule) i32;
    extern fn et_module_num_inputs(m: *ETModule, n_inputs: *i32) i32;
    extern fn et_module_input_info(m: *ETModule, index: i32, dtype: *i32, sizes: *[*c]const i32, rank: *i32) i32;
    extern fn et_module_validate_inputs(m: *ETModule, inputs: [*]const ETTensor, n_inputs: i32) i32;
    extern fn et_module_forward(m: *ETModule, inputs: [*]const ETTensor, n_inputs: i32) ?*ETResult;
    extern fn et_result_error(r: *const ETResult) i32;
    extern fn et_result_num_outputs(r: *const ETResult) i32;
    extern fn et_result_output_rank(r: *const ETResult, index: i32) i32;
    extern fn et_result_output_sizes(r: *const ETResult, index: i32) [*c]const i32;
    extern fn et_result_output_dtype(r: *const ETResult, index: i32) i32;
    extern fn et_result_output_data(r: *const ETResult, index: i32) ?*const anyopaque;
    extern fn et_result_destroy(r: *ETResult) void;
};

/// Mirrors executorch::runtime::Error (runtime/core/error.h), named 1:1 so a
/// caller can match on the specific failure instead of one opaque bucket.
pub const EtError = error{
    Internal,
    InvalidState,
    EndOfMethod,
    AlreadyLoaded,
    NotSupported,
    NotImplemented,
    InvalidArgument,
    InvalidType,
    OperatorMissing,
    RegistrationExceedingMaxKernels,
    RegistrationAlreadyRegistered,
    NotFound,
    MemoryAllocationFailed,
    AccessFailed,
    InvalidProgram,
    InvalidExternalData,
    OutOfResources,
    DelegateInvalidCompatibility,
    DelegateMemoryAllocationFailed,
    DelegateInvalidHandle,
    /// A nonzero code came back that isn't in runtime/core/error.h as
    /// vendored; keeps this mapping from silently misreporting a future
    /// ExecuTorch error code as one of the above.
    UnknownExecutorchError,
};

/// Maps a raw ExecuTorch `Error` code (0 == Ok) to EtError. Never call with 0.
fn mapEtError(code: i32) EtError {
    return switch (code) {
        0x01 => EtError.Internal,
        0x02 => EtError.InvalidState,
        0x03 => EtError.EndOfMethod,
        0x04 => EtError.AlreadyLoaded,
        0x10 => EtError.NotSupported,
        0x11 => EtError.NotImplemented,
        0x12 => EtError.InvalidArgument,
        0x13 => EtError.InvalidType,
        0x14 => EtError.OperatorMissing,
        0x15 => EtError.RegistrationExceedingMaxKernels,
        0x16 => EtError.RegistrationAlreadyRegistered,
        0x20 => EtError.NotFound,
        0x21 => EtError.MemoryAllocationFailed,
        0x22 => EtError.AccessFailed,
        0x23 => EtError.InvalidProgram,
        0x24 => EtError.InvalidExternalData,
        0x25 => EtError.OutOfResources,
        0x30 => EtError.DelegateInvalidCompatibility,
        0x31 => EtError.DelegateMemoryAllocationFailed,
        0x32 => EtError.DelegateInvalidHandle,
        else => EtError.UnknownExecutorchError,
    };
}

pub const Error = EtError || error{
    /// The shim couldn't allocate the module/result handle.
    ExecutorchError,
    /// Input count/rank/dtype didn't match what the loaded method expects.
    InvalidInput,
};

/// Mirrors ETScalarType / the subset of executorch::aten::ScalarType this
/// shim supports.
pub const DType = enum(i32) {
    f32 = c.ET_DTYPE_FLOAT,
    i64 = c.ET_DTYPE_INT64,
    i32 = c.ET_DTYPE_INT32,
    bool = c.ET_DTYPE_BOOL,
};

fn dtypeOf(comptime T: type) DType {
    return switch (T) {
        f32 => .f32,
        i64 => .i64,
        i32 => .i32,
        bool => .bool,
        else => @compileError("unsupported ExecuTorch tensor element type: " ++ @typeName(T)),
    };
}

pub const RuntimeOptions = struct {
    /// Size of the process-wide thread pool shared by XNNPACK and the
    /// portable kernels. null keeps ExecuTorch's default (performance cores).
    num_threads: ?u32 = null,
};

/// Proof that runtimeInit() ran. Module.init/initFromBuffer require it, so
/// the thread count is always fixed before the first load(), which is when
/// XNNPACK takes the pool pointer for good.
pub const Runtime = struct {
    /// Thread count of the global pool (1 in builds without XNNPACK).
    pub fn numThreads(_: Runtime) u32 {
        const n = c.et_runtime_num_threads();
        std.debug.assert(n > 0);
        return n;
    }
};

/// Initializes the runtime and its global thread pool. Safe to call again
/// and from several threads: later calls with null or the same num_threads
/// return the token, a different num_threads gives error.InvalidState since
/// loaded modules may already hold the pool. num_threads = 0 gives
/// error.InvalidArgument; more than 1 in a build without XNNPACK gives
/// error.NotSupported.
pub fn runtimeInit(options: RuntimeOptions) EtError!Runtime {
    const n = options.num_threads orelse 0;
    if (options.num_threads != null and n == 0) return EtError.InvalidArgument;
    const code = c.et_runtime_init(n);
    if (code != 0) return mapEtError(code);
    return .{};
}

pub const Tensor = struct {
    data: *const anyopaque,
    sizes: []const i32,
    dtype: DType,

    /// data/sizes must outlive the forward() call this Tensor is passed to.
    pub fn from(comptime T: type, data: []const T, sizes: []const i32) Tensor {
        return .{ .data = data.ptr, .sizes = sizes, .dtype = dtypeOf(T) };
    }

    fn toC(self: Tensor) c.ETTensor {
        return .{
            .data = self.data,
            .sizes = self.sizes.ptr,
            .rank = @intCast(self.sizes.len),
            .dtype = @backingInt(self.dtype),
        };
    }
};

pub const ForwardResult = struct {
    handle: *c.ETResult,

    pub fn deinit(self: ForwardResult) void {
        c.et_result_destroy(self.handle);
    }

    pub fn numOutputs(self: ForwardResult) usize {
        return @intCast(c.et_result_num_outputs(self.handle));
    }

    /// Sizes slice is valid only until deinit() is called. Empty for a 0-dim
    /// tensor and for a non-tensor output.
    pub fn outputSizes(self: ForwardResult, index: usize) []const i32 {
        std.debug.assert(index < self.numOutputs());
        const rank = c.et_result_output_rank(self.handle, @intCast(index));
        if (rank <= 0) return &.{};
        const ptr = c.et_result_output_sizes(self.handle, @intCast(index));
        return ptr[0..@intCast(rank)];
    }

    /// null for a non-tensor output or a dtype outside `DType`.
    pub fn outputDType(self: ForwardResult, index: usize) ?DType {
        std.debug.assert(index < self.numOutputs());
        return std.enums.fromInt(DType, c.et_result_output_dtype(self.handle, @intCast(index)));
    }

    /// Data slice is valid only until deinit() is called. Returns null if T
    /// doesn't match the output's actual dtype (query outputDType() first
    /// if unsure).
    pub fn outputData(self: ForwardResult, comptime T: type, index: usize) ?[]const T {
        if (self.outputDType(index) != dtypeOf(T)) return null;
        const sizes = self.outputSizes(index);
        var numel: usize = 1;
        for (sizes) |s| numel *= @intCast(s);
        if (numel == 0) return &.{};
        const ptr: [*]const T = @ptrCast(@alignCast(c.et_result_output_data(self.handle, @intCast(index)).?));
        return ptr[0..numel];
    }
};

/// Metadata of one input of the "forward" method, as recorded in the program.
pub const InputInfo = struct {
    /// null for a dtype outside `DType`.
    dtype: ?DType,
    /// Valid until the owning Module is deinit()ed.
    sizes: []const i32,
};

pub const Module = struct {
    handle: *c.ETModule,

    /// path must outlive this call (it's copied internally by ExecuTorch).
    pub fn init(_: Runtime, path: [:0]const u8) Error!Module {
        const handle = c.et_module_create(path.ptr) orelse return Error.ExecutorchError;
        return .{ .handle = handle };
    }

    /// Reads the program from memory without copying: `data` must stay valid
    /// and unchanged until deinit().
    pub fn initFromBuffer(_: Runtime, data: []const u8) Error!Module {
        const handle = c.et_module_create_from_buffer(data.ptr, data.len) orelse return Error.ExecutorchError;
        return .{ .handle = handle };
    }

    pub fn deinit(self: Module) void {
        c.et_module_destroy(self.handle);
    }

    pub fn load(self: Module) EtError!void {
        const code = c.et_module_load(self.handle);
        if (code != 0) return mapEtError(code);
    }

    /// Number of inputs of the "forward" method. Loads the program if needed.
    pub fn numInputs(self: Module) EtError!usize {
        var n: i32 = undefined;
        const code = c.et_module_num_inputs(self.handle, &n);
        if (code != 0) return mapEtError(code);
        std.debug.assert(n >= 0);
        return @intCast(n);
    }

    /// Metadata of input `index` of the "forward" method. Loads the program
    /// if needed. error.InvalidType for a non-tensor input.
    pub fn inputInfo(self: Module, index: usize) EtError!InputInfo {
        var dtype: i32 = undefined;
        var sizes: [*c]const i32 = undefined;
        var rank: i32 = undefined;
        const code = c.et_module_input_info(self.handle, @intCast(index), &dtype, &sizes, &rank);
        if (code != 0) return mapEtError(code);
        std.debug.assert(rank >= 0);
        return .{
            .dtype = std.enums.fromInt(DType, dtype),
            .sizes = if (rank == 0) &.{} else sizes[0..@intCast(rank)],
        };
    }

    /// inputs must outlive the call (not copied). Caller must deinit() the result.
    /// Validates inputs against the loaded method's expected count/rank/dtype
    /// and caps every dim at the size recorded in the program (the upper bound
    /// of a dynamic dim), so a mismatch surfaces as error.InvalidInput instead
    /// of undefined behavior. A smaller dim of a static shape is rejected by
    /// ExecuTorch with error.NotSupported.
    pub fn forward(self: Module, inputs: []const Tensor, arena: std.mem.Allocator) !ForwardResult {
        const c_inputs = try arena.alloc(c.ETTensor, inputs.len);
        for (inputs, 0..) |t, i| c_inputs[i] = t.toC();

        const check = c.et_module_validate_inputs(self.handle, c_inputs.ptr, @intCast(c_inputs.len));
        if (check > 0) return mapEtError(check);
        if (check < 0) return Error.InvalidInput;

        const handle = c.et_module_forward(self.handle, c_inputs.ptr, @intCast(c_inputs.len)) orelse
            return Error.ExecutorchError;
        const result: ForwardResult = .{ .handle = handle };
        const code = c.et_result_error(handle);
        if (code != 0) {
            result.deinit();
            return mapEtError(code);
        }
        return result;
    }
};
