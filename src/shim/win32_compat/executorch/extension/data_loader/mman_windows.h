/*
 * Minimal POSIX mman.h shim for Windows, backing
 * vendor/executorch/extension/data_loader/mman.h's `#else` (Windows) branch.
 * Deliberately excluded from vendoring since ExecuTorch
 * upstream doesn't ship one; only implements what mmap_data_loader.cpp
 * actually calls: read-only mmap/munmap and mlock/munlock.
 */
#pragma once

#include <io.h>
#include <windows.h>

#define PROT_READ 0x1
#define PROT_WRITE 0x2
#define MAP_SHARED 0x01
#define MAP_PRIVATE 0x02
#define MAP_FAILED ((void*)-1)

inline void* mmap(
    void* /*addr*/,
    size_t len,
    int prot,
    int /*flags*/,
    int fd,
    uint64_t offset) {
  HANDLE file = (HANDLE)_get_osfhandle(fd);
  if (file == INVALID_HANDLE_VALUE) {
    return MAP_FAILED;
  }
  DWORD protect = (prot & PROT_WRITE) ? PAGE_READWRITE : PAGE_READONLY;
  HANDLE mapping = CreateFileMappingA(file, nullptr, protect, 0, 0, nullptr);
  if (mapping == nullptr) {
    return MAP_FAILED;
  }
  DWORD access = (prot & PROT_WRITE) ? FILE_MAP_WRITE : FILE_MAP_READ;
  LARGE_INTEGER off;
  off.QuadPart = static_cast<LONGLONG>(offset);
  void* p =
      MapViewOfFile(mapping, access, off.HighPart, off.LowPart, len);
  // The mapping handle isn't needed once a view is created; the view keeps
  // the underlying mapping alive until UnmapViewOfFile.
  CloseHandle(mapping);
  if (p == nullptr) {
    return MAP_FAILED;
  }
  return p;
}

inline int munmap(void* addr, size_t /*len*/) {
  return UnmapViewOfFile(addr) ? 0 : -1;
}

inline int mlock(const void* addr, size_t len) {
  return VirtualLock(const_cast<void*>(addr), len) ? 0 : -1;
}

inline int munlock(const void* addr, size_t len) {
  return VirtualUnlock(const_cast<void*>(addr), len) ? 0 : -1;
}
