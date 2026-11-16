const std = @import("std");

/// ExecuTorch's own sources use `#include <executorch/...>`, so the include
/// root must be the directory containing `vendor/executorch`, i.e. `vendor/`.
const vendor_dir = "vendor";
const executorch_src_dir = vendor_dir ++ "/executorch";
const generated_dir = vendor_dir ++ "/generated/portable_ops_lib";
const flatbuffers_include_dir = vendor_dir ++ "/third-party/flatbuffers/include";
const c10_shim_include_dir = executorch_src_dir ++ "/runtime/core/portable_type/c10";

const cxx_flags = &[_][]const u8{
    "-std=c++17",
    "-fno-exceptions",
    "-fno-rtti",
    "-DC10_USING_CUSTOM_GENERATED_MACROS",
    // Enables Program::Verification::InternalConsistency (pointer/consistency
    // checks on the flatbuffer), used by the shim before parsing any .pte.
    // Without this, that verification level silently degrades to Minimal.
    "-DET_ENABLE_PROGRAM_VERIFICATION=1",
};

// Only defined when the XNNPACK backend is wired in: switches
// runtime/kernel/thread_parallel_interface.h from its header-only,
// no-threadpool fallback (get_thread_num/set_thread_num defined inline in
// the header) to declarations satisfied by extension/threadpool's real
// definitions. Must be applied uniformly to every TU in `mod` — portable
// kernels include this header too, and a mismatched view of it across TUs
// (some seeing `inline` definitions, others seeing bare declarations) is an
// ODR violation, not just a XNNPACK-local concern.
const et_use_threadpool_flag = "-DET_USE_THREADPOOL";

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const extra_cxx_flags = b.option([]const []const u8, "extra_cxx_flags", "Appended to every vendored/shim C++ compile (e.g. -fsanitize-coverage for fuzzing)") orelse &.{};
    _ = addLibModule(b, target, optimize, extra_cxx_flags);
}

/// The library module: root.zig plus every vendored ExecuTorch/XNNPACK
/// source.
fn addLibModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, extra_cxx_flags: []const []const u8) *std.Build.Module {
    // ExecuTorch's operator-registration translation units (portable ops,
    // prim ops) work via static-initializer side effects: a global object's
    // constructor calls into the operator registry. No symbol from those
    // TUs is ever referenced directly, so if they're archived into a static
    // library, the linker drops them as "unused" and kernel lookups fail at
    // runtime ("kernel 'aten::add.out' not found"). Compiling every source
    // file directly into the public module (rather than a separate static
    // lib. Compile step) means they always ship as individual object files
    // in the final link, sidestepping the archive-member-pruning problem
    // without needing platform-specific whole-archive/force_load flags.
    const mod = b.addModule("lib_executorch", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libcpp = true,
    });
    mod.addIncludePath(b.path(vendor_dir));
    mod.addIncludePath(b.path(flatbuffers_include_dir));
    mod.addIncludePath(b.path(c10_shim_include_dir));
    mod.addIncludePath(b.path(generated_dir));
    mod.addIncludePath(b.path("src/shim"));

    // Unlike macOS (built natively on this host, so zig auto-detects the
    // system SDK), iOS is always a foreign cross-target here: zig ships no
    // bundled iOS libc/libc++ headers or frameworks, so without an explicit
    // SDK sysroot every libc++ TU that reaches <pthread.h>/<sched.h> etc.
    // fails to find them. Locate the right SDK via `xcrun` (simulator vs.
    // device ABI) and wire its include/framework/library paths onto every
    // module built for this target.
    if (target.result.os.tag == .ios) addIosSdkPaths(b, mod, target.result);

    // ExecuTorch's mman.h expects a hand-rolled mman_windows.h on Windows
    // (CreateFileMapping-based mmap/munmap/mlock/munlock); upstream doesn't
    // ship one and it's deliberately excluded from vendoring, so provide our
    // own via an extra include dir rather than patching it into vendor/.
    if (target.result.os.tag == .windows) {
        mod.addIncludePath(b.path("src/shim/win32_compat"));
    }

    // XNNPACK delegate: real perf backend, wired into the default module for every OS/arch combination
    // xnnpack-fork/cpuinfo/pthreadpool-fork actually support. Registers
    // itself into ExecuTorch's backend registry via the same static-init
    // pattern as the portable kernels (see addSourcesRecursive's doc comment)
    // — compiling straight into `mod` (never an intermediate static lib)
    // keeps that registration from being pruned, no different than the
    // existing portable-kernel gotcha.
    const xnnpack_enabled = switch (target.result.os.tag) {
        .macos, .linux, .windows => switch (target.result.cpu.arch) {
            .x86_64, .aarch64 => true,
            else => false,
        },
        else => false,
    };
    const threadpool_flags: []const []const u8 = if (xnnpack_enabled) &.{et_use_threadpool_flag} else &.{};
    const active_cxx_flags = std.mem.concat(b.allocator, []const u8, &.{ cxx_flags, threadpool_flags, extra_cxx_flags }) catch @panic("OOM");
    if (xnnpack_enabled) addXnnpackBackend(b, mod, target, optimize);

    addSourcesRecursive(b, b.allocator, mod, executorch_src_dir, target.result.os.tag, active_cxx_flags);
    mod.addCSourceFile(.{
        .file = b.path(generated_dir ++ "/RegisterCodegenUnboxedKernelsEverything.cpp"),
        .flags = active_cxx_flags,
    });
    mod.addCSourceFile(.{
        .file = b.path("src/shim/executorch_shim.cpp"),
        .flags = active_cxx_flags,
    });
    return mod;
}

/// Wires an iOS SDK's headers/frameworks onto `mod`. Runs `xcrun` at configure time to locate the
/// right SDK for the target's ABI (device vs. simulator) — this only works
/// on a macOS host with Xcode installed, same as building for macOS
/// itself, just without zig's usual auto-detection since iOS is never the
/// zig build's own host target.
///
/// Since zig 0.17 build.zig can no longer set the sysroot, so pass
/// `--sysroot "$(xcrun --sdk <sdk> --show-sdk-path)"` on the command line.
/// Both pieces are needed: the sysroot is what zig's own internal libc++
/// build (triggered by `link_libcpp`) consults for a Darwin non-macOS
/// target, but our own module's C/C++ TUs still need explicit include
/// paths. The sysroot *also* makes zig auto-derive a `usr/lib` library
/// search path, so don't add that one again here — doing so double-prefixes
/// the sysroot and breaks linking ("unable to open library directory
/// '<sdk>/<sdk>/usr/lib'").
///
/// Still fails on zig 0.17.1, now with `mbstate_t` undefined inside zig's
/// libc++ build. Original 0.16 analysis:
/// `link_libcpp` currently fails to cross-compile for iOS on zig 0.16: zig's
/// own internal libc++/libc++abi build (a static lib it builds once per
/// target, independent of any of this function's search paths) hits "no
/// member named 'memcpy'/'memset' in the global namespace" inside
/// libcxxabi/src/cxa_exception.cpp — the libc++
/// `using ::memcpy _LIBCPP_USING_IF_EXISTS;` pattern silently swallows a
/// header that didn't actually resolve for this target. This is the same
/// class of bug as https://github.com/ziglang/zig/issues/16352 (closed, but
/// that fix only covered a pure-C/Foundation example with `link_libc`, not
/// `link_libcpp`) and unblocked by zig/zig#17470 (open, "unplanned": zig has
/// no flag to build libcxx/libcxxabi with `-fno-exceptions` — confirmed via
/// upstream issue tracker, so that workaround isn't reachable without
/// patching zig's own compiler source) — an upstream toolchain gap with no
/// workaround through zig's own libc++ build.
///
/// Tried instead: disabling `link_libcpp` for iOS and linking Apple's own
/// system libc++ directly (its SDK ships full `usr/include/c++/v1` headers
/// and a `libc++.tbd` stub — `linkSystemLibrary("c++")` doesn't work for
/// this since zig special-cases that name back to its own broken build, but
/// `addObjectFile` on the `.tbd` bypasses that). That gets past the
/// memcpy/memset error, but then hits a libc++ header-search-ordering
/// problem (`<cstddef>`'s `#include_next` chain needs a specific
/// resource-dir/system-header order that zig's per-module
/// `addSystemIncludePath` ordering doesn't reproduce reliably) — not
/// resolved; not something fixable from this file without visibility into
/// zig's internal Module include-dir handling.
fn addIosSdkPaths(b: *std.Build, mod: *std.Build.Module, t: std.Target) void {
    const sdk_name = if (t.abi == .simulator) "iphonesimulator" else "iphoneos";
    const sdk_path = std.mem.trim(u8, b.run(&.{ "xcrun", "--sdk", sdk_name, "--show-sdk-path" }), " \n\r\t");
    mod.addSystemIncludePath(b.graph.cwdRelativePath(b.pathJoin(&.{ sdk_path, "usr/include" })));
    mod.addSystemFrameworkPath(b.graph.cwdRelativePath(b.pathJoin(&.{ sdk_path, "System/Library/Frameworks" })));
}

// ---------------------------------------------------------------------------
// XNNPACK delegate backend — the ExecuTorch-side glue (backends/xnnpack/
// runtime/*.cpp + extension/threadpool) that turns the vendored xnnpack
// package into ExecuTorch's real "XnnpackBackend" delegate, not just a
// standalone microkernel/subgraph PoC.
// ---------------------------------------------------------------------------

/// Wires the XNNPACK delegate backend into `mod` so any `.pte` with an
/// XNNPACK-partitioned subgraph runs through real microkernels instead of
/// falling back to (or failing without) portable kernels. ExecuTorch's
/// executor discovers registered backends generically at `Program`/`Method`
/// load time by the delegate's `backend_id` string baked into the `.pte` —
/// nothing in src/root.zig or the shim needs to change for this to take
/// effect. The backend/threadpool `.cpp` sources themselves don't need
/// adding here: they live under `vendor/executorch/` (backends/xnnpack/
/// runtime, extension/threadpool) alongside the portable kernels, so
/// `addSourcesRecursive`'s tree walk already compiles them — this function
/// only wires the extra import + include paths those TUs need.
fn addXnnpackBackend(b: *std.Build, mod: *std.Build.Module, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    const xnnpack_dep = b.dependency("xnnpack", .{ .target = target, .optimize = optimize });
    mod.addImport("xnnpack", xnnpack_dep.module("xnnpack"));

    // Backend/threadpool TUs use the public `<xnnpack.h>`/`<pthreadpool.h>`/
    // `<cpuinfo.h>` include style (angle brackets, no `include/` prefix) —
    // distinct from xnnpack-fork's own internal sources, which include their
    // own public header as `"include/xnnpack.h"` relative to the package
    // root (see xnnpack-fork/build.zig's addModule). addImport doesn't
    // propagate C include paths across a module boundary, so these need to
    // be added directly to `mod` too.
    mod.addIncludePath(xnnpack_dep.path("include"));
    mod.addIncludePath(b.dependency("pthreadpool", .{}).path("include"));
    mod.addIncludePath(b.dependency("cpuinfo", .{}).path("include"));
}

// POSIX-only TUs (pread, mmap, no mman_windows.h shim in this vendored
// snapshot) that fail to compile for Windows targets. Unreferenced by our
// shim (nothing in src/ uses FileDescriptorDataLoader/MmapDataLoader), so
// skipping them for Windows is safe.
const windows_incompatible_srcs = &[_][]const u8{
    // fd:// URI loader: uses pread()/VLA-in-C++ in a way not covered by the
    // compat_unistd.h shim, and unreferenced by module.cpp (only
    // FileDataLoader/MmapDataLoader are), so just drop it for Windows.
    "extension/data_loader/file_descriptor_data_loader.cpp",
};

/// Recursively finds and adds every `.cpp` file under `dir_path` (relative to
/// the build root) as a C++ source file compiled into `mod`. The vendored
/// ExecuTorch tree has already been pruned to only the files needed for a
/// portable-kernel, non-ATen, POSIX build (see vendor/EXECUTORCH_VERSION.txt),
/// so no further filtering is needed here.
fn addSourcesRecursive(b: *std.Build, arena: std.mem.Allocator, mod: *std.Build.Module, dir_path: []const u8, os_tag: std.Target.Os.Tag, flags: []const []const u8) void {
    const io = b.graph.io;
    const root_path = b.root.join(arena, dir_path) catch @panic("OOM");
    var dir = root_path.root_dir.handle.openDir(io, root_path.sub_path, .{ .iterate = true }) catch |err| {
        std.debug.panic("failed to open {s}: {s}", .{ dir_path, @errorName(err) });
    };
    defer dir.close(io);

    var walker = dir.walk(arena) catch @panic("OOM");
    defer walker.deinit();

    while (walker.next(io) catch @panic("walk error")) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.basename, ".cpp")) continue;
        if (os_tag == .windows) {
            var skip = false;
            for (windows_incompatible_srcs) |skip_suffix| {
                if (std.mem.endsWith(u8, entry.path, skip_suffix)) {
                    skip = true;
                    break;
                }
            }
            if (skip) continue;
        }
        const full_path = std.fs.path.join(arena, &.{ dir_path, entry.path }) catch @panic("OOM");
        mod.addCSourceFile(.{ .file = b.path(full_path), .flags = flags });
    }
}
