// quill: minimal OS layer (Windows / POSIX) for positional file reads and page allocations.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#endif

namespace ingine::platform {

#ifdef _WIN32
using FileHandle = HANDLE;
inline const FileHandle kInvalidFile = INVALID_HANDLE_VALUE;

inline std::string last_error() {
    const DWORD e = GetLastError();
    char buf[512] = {0};
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, e, 0, buf, sizeof(buf), nullptr);
    return "error " + std::to_string(e) + ": " + buf;
}

inline FileHandle open_read(const std::filesystem::path& p) {
    return CreateFileW(p.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                       FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
}
inline std::uint64_t file_size(FileHandle h) {
    LARGE_INTEGER s;
    if (!GetFileSizeEx(h, &s)) throw std::runtime_error("GetFileSizeEx failed: " + last_error());
    return std::uint64_t(s.QuadPart);
}
// Reads up to `bytes` at `offset`; returns bytes read (0 at EOF), -1 on error.
inline std::int64_t read_at(FileHandle h, void* dst, std::size_t bytes, std::uint64_t offset) {
    OVERLAPPED ov{};
    ov.Offset = DWORD(offset & 0xffffffffu);
    ov.OffsetHigh = DWORD(offset >> 32);
    DWORD got = 0;
    const DWORD want = DWORD(bytes > (1u << 30) ? (1u << 30) : bytes);
    if (!ReadFile(h, dst, want, &got, &ov)) {
        if (GetLastError() == ERROR_HANDLE_EOF) return 0;
        return -1;
    }
    return std::int64_t(got);
}
inline void close_file(FileHandle h) {
    if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
}
inline void* page_alloc(std::size_t bytes) { return VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE); }
inline void page_free(void* p, std::size_t) {
    if (p) VirtualFree(p, 0, MEM_RELEASE);
}

// quill: D3DKMTSetProcessSchedulingPriorityClass(GetCurrentProcess(), cls) (gdi32, dynamic load; no admin
// below Realtime). cls: 3 AboveNormal, 4 High. Lifts the engine's CUDA context ahead of the desktop's own
// GPU users (dwm, the browser), the GPU-side counterpart of the launcher's process priority High.
inline bool set_gpu_scheduling_priority(int cls) {
    using Fn = long(__stdcall*)(void*, int);
    const HMODULE g = GetModuleHandleW(L"gdi32.dll");
    if (!g) return false;
    const auto fn = reinterpret_cast<Fn>(GetProcAddress(g, "D3DKMTSetProcessSchedulingPriorityClass"));
    return fn && fn(GetCurrentProcess(), cls) == 0;  // NTSTATUS 0 = success
}
#else
using FileHandle = int;
inline const FileHandle kInvalidFile = -1;
inline std::string last_error() { return std::strerror(errno); }
inline FileHandle open_read(const std::filesystem::path& p) { return ::open(p.c_str(), O_RDONLY | O_CLOEXEC); }
inline std::uint64_t file_size(FileHandle h) {
    struct stat st {};
    if (::fstat(h, &st) != 0) throw std::runtime_error("fstat failed");
    return std::uint64_t(st.st_size);
}
inline std::int64_t read_at(FileHandle h, void* dst, std::size_t bytes, std::uint64_t offset) {
    for (;;) {
        const ssize_t got = ::pread(h, dst, bytes, off_t(offset));
        if (got < 0 && errno == EINTR) continue;
        return got;
    }
}
inline void close_file(FileHandle h) {
    if (h >= 0) ::close(h);
}
inline void* page_alloc(std::size_t bytes) {
    void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return p == MAP_FAILED ? nullptr : p;
}
inline void page_free(void* p, std::size_t bytes) {
    if (p) ::munmap(p, bytes);
}
#endif

}  // namespace ingine::platform
