# zig-lib-executorch

Zig bindings for the [ExecuTorch](https://github.com/pytorch/executorch) runtime (v1.3.1) with the XNNPACK backend. Builds with `zig build` alone: the ExecuTorch codegen output is vendored, so no CMake, Python or flatc is needed.

Requires Zig 0.17.1. Tested on macOS and Linux (x86_64, aarch64) and on Windows x86_64. Android compiles, iOS is blocked by zig's libc++ build.

```sh
zig fetch --save git+https://github.com/nat3Github/zig-lib-executorch
```

```zig
const executorch = b.dependency("lib_executorch", .{ .target = target, .optimize = optimize });
exe.root_module.addImport("lib_executorch", executorch.module("lib_executorch"));
```

```zig
const et = @import("lib_executorch");

const rt = try et.runtimeInit(.{});
const module = try et.Module.init(rt, "model.pte");
defer module.deinit();
try module.load();

const x = [_]f32{ 1, 2, 3, 4 };
const result = try module.forward(&.{et.Tensor.from(f32, &x, &.{ 2, 2 })}, arena);
defer result.deinit();
const out = result.outputData(f32, 0).?;
```

## License

BSD-3-Clause. Vendored ExecuTorch: `vendor/LICENSE-executorch` (BSD-3-Clause). XNNPACK, cpuinfo and pthreadpool are fetched from BSD-licensed forks.
