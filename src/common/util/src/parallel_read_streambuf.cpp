// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/parallel_read_streambuf.hpp"

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <fcntl.h>  // posix_fadvise
#endif

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#include "openvino/util/file_util.hpp"
#include "openvino/util/parallel_io.hpp"

namespace ov::util {

ParallelReadStreamBuf::ParallelReadStreamBuf(const std::filesystem::path& path,
                                             std::streamoff header_offset,
                                             size_t threshold)
    : m_path(path),
      m_file_offset(header_offset),
      m_header_offset(header_offset),
      m_threshold(threshold) {
    get_file_handle_and_size(path, m_file_offset, m_handle, m_file_size);

    // Advise the kernel to prefetch the blob region into pagecache
    // asynchronously.  This makes subsequent small pread() calls hit warm
    // cache instead of blocking on disk I/O.
    const auto advise_offset = static_cast<size_t>(m_header_offset);
    const auto advise_len = static_cast<size_t>(m_file_size - m_header_offset);
    if (advise_len > 0) {
#ifndef _WIN32
        // POSIX_FADV_SEQUENTIAL doubles the default readahead window.
        // POSIX_FADV_WILLNEED starts async readahead for the whole region.
        // Both are advisory — failure is non-fatal.
        (void)posix_fadvise(m_handle, static_cast<off_t>(advise_offset),
                            static_cast<off_t>(advise_len), POSIX_FADV_SEQUENTIAL);
        (void)posix_fadvise(m_handle, static_cast<off_t>(advise_offset),
                            static_cast<off_t>(advise_len), POSIX_FADV_WILLNEED);
#endif
    }
}

ParallelReadStreamBuf::~ParallelReadStreamBuf() {
    close_file_handle(m_handle);
}

// Serve bytes from the read-ahead buffer.  Updates dst/n in-place.
std::streamsize ParallelReadStreamBuf::serve_from_readahead(char_type*& dst, std::streamsize& n) {
    const auto off = static_cast<size_t>(m_file_offset);
    if (off < m_ra_buf_start || off >= m_ra_buf_end)
        return 0;
    const size_t buf_avail = m_ra_buf_end - off;
    const size_t to_copy = (std::min)(buf_avail, static_cast<size_t>(n));
    std::memcpy(dst, m_ra_buf.get() + (off - m_ra_buf_start), to_copy);
    m_file_offset += static_cast<std::streamoff>(to_copy);
    dst += to_copy;
    n -= static_cast<std::streamsize>(to_copy);
    return static_cast<std::streamsize>(to_copy);
}

// Refill the read-ahead buffer starting at m_file_offset.
bool ParallelReadStreamBuf::refill_readahead() {
    if (!m_ra_buf) {
        m_ra_buf = std::make_unique<char_type[]>(default_parallel_io_readahead_size);
    }
    const size_t off = static_cast<size_t>(m_file_offset);
    const size_t remaining = static_cast<size_t>(m_file_size - m_file_offset);
    const size_t to_read = (std::min)(default_parallel_io_readahead_size, remaining);
    if (to_read == 0)
        return false;
    if (!single_read(m_ra_buf.get(), to_read, off))
        return false;
    m_ra_buf_start = off;
    m_ra_buf_end = off + to_read;
    return true;
}

// xsgetn: main hot path - called by sgetn() for all bulk reads
std::streamsize ParallelReadStreamBuf::xsgetn(char_type* dst, std::streamsize n) {
    if (n <= 0)
        return 0;

    std::streamsize total = 0;

    // 1. Drain any chars in the streambuf get-area (set by underflow)
    if (gptr() != nullptr && gptr() < egptr()) {
        const std::streamsize avail = static_cast<std::streamsize>(egptr() - gptr());
        const std::streamsize from_buf = (std::min)(n, avail);
        std::memcpy(dst, gptr(), static_cast<size_t>(from_buf));
        static_assert(default_parallel_io_readahead_size <= static_cast<size_t>((std::numeric_limits<int>::max)()),
                      "default_parallel_io_readahead_size must fit in int for gbump()");
        gbump(static_cast<int>(from_buf));
        total += from_buf;
        dst += from_buf;
        n -= from_buf;
    }

    if (n <= 0 || m_file_offset >= m_file_size) {
        return total;
    }

    const std::streamoff remaining = m_file_size - m_file_offset;
    n = static_cast<std::streamsize>((std::min)(static_cast<std::streamoff>(n), remaining));

    // 2. Large reads: bypass the read-ahead buffer entirely
    if (static_cast<size_t>(n) >= m_threshold) {
        const size_t bytes = static_cast<size_t>(n);
        const size_t offset = static_cast<size_t>(m_file_offset);
        bool ok = parallel_read(dst, bytes, offset);
        if (ok) {
            m_file_offset += n;
            total += n;
        }
        // Invalidate the read-ahead buffer since we jumped past it
        m_ra_buf_start = m_ra_buf_end = 0;
        return total;
    }

    // 3. Small reads: serve from the read-ahead buffer, refilling as needed
    while (n > 0 && m_file_offset < m_file_size) {
        // Try the existing read-ahead window first
        std::streamsize served = serve_from_readahead(dst, n);
        total += served;
        if (n <= 0)
            break;
        // Buffer miss — refill
        if (!refill_readahead())
            break;
    }

    return total;
}

// underflow: called for single-char peek / non-bulk reads (e.g. std::getline)
ParallelReadStreamBuf::int_type ParallelReadStreamBuf::underflow() {
    if (m_file_offset >= m_file_size) {
        return traits_type::eof();
    }
    // Reuse the read-ahead buffer: if current offset falls within it,
    // expose the remaining window as the get-area.  Otherwise refill.
    const auto off = static_cast<size_t>(m_file_offset);
    if (off < m_ra_buf_start || off >= m_ra_buf_end) {
        if (!refill_readahead())
            return traits_type::eof();
    }
    // Expose the portion from current offset to end of read-ahead buffer.
    char_type* base = m_ra_buf.get();
    const size_t buf_off = off - m_ra_buf_start;
    const size_t buf_avail = m_ra_buf_end - off;
    // Advance m_file_offset past the bytes we expose in the get area.
    // seekoff(0, cur) formula: logical_pos = m_file_offset - (egptr - gptr).
    m_file_offset += static_cast<std::streamoff>(buf_avail);
    setg(base + buf_off, base + buf_off, base + buf_off + buf_avail);
    return traits_type::to_int_type(*(base + buf_off));
}

ParallelReadStreamBuf::pos_type ParallelReadStreamBuf::seekoff(off_type off,
                                                               std::ios_base::seekdir way,
                                                               std::ios_base::openmode /* which */) {
    // All internal positions (m_file_offset, m_file_size, m_header_offset) are
    // absolute byte offsets from the start of the file.  The public-facing
    // stream positions are *logical* offsets: 0 == header_offset in the file.
    std::streamoff new_pos = 0;
    if (way == std::ios_base::beg) {
        // off is a logical offset; translate to absolute file offset.
        new_pos = m_header_offset + off;
    } else if (way == std::ios_base::cur) {
        // Account for the buffered chars from underflow() not yet consumed.
        const std::streamsize ahead = (gptr() != nullptr) ? static_cast<std::streamsize>(egptr() - gptr()) : 0;
        new_pos = m_file_offset - ahead + off;  // stays absolute
        // Pure tell (off == 0): return current position without any side effects
        // on the get area or m_file_offset.  Discarding the underflow buffer on a
        // tell would force the next read to re-issue a pread for data that is
        // already buffered, breaking interleaved getline()+tellg() patterns.
        if (off == 0) {
            if (new_pos < m_header_offset || new_pos > m_file_size)
                return pos_type(off_type(-1));
            return pos_type(new_pos - m_header_offset);
        }
    } else {
        new_pos = m_file_size + off;  // stays absolute
    }

    // Reject seeks before the logical stream start or past the file end.
    if (new_pos < m_header_offset || new_pos > m_file_size) {
        return pos_type(off_type(-1));
    }

    setg(nullptr, nullptr, nullptr);  // invalidate get-area
    m_ra_buf_start = m_ra_buf_end = 0;  // invalidate read-ahead buffer
    m_file_offset = new_pos;
    // Return the logical position (0 == start of exposed stream).
    return pos_type(m_file_offset - m_header_offset);
}

ParallelReadStreamBuf::pos_type ParallelReadStreamBuf::seekpos(pos_type pos, std::ios_base::openmode /* which */) {
    return seekoff(off_type(pos), std::ios_base::beg, std::ios_base::in);
}

std::streamsize ParallelReadStreamBuf::showmanyc() {
    // Report both buffered characters (in the get area) and remaining
    // bytes in the underlying file.
    // Per [streambuf.virt.get]/6: return -1 when the next call to
    // underflow() would return traits_type::eof() (i.e. stream truly
    // exhausted).  Return 0 when availability is unknown.  Here both
    // the file and the get-area are fully accounted for, so -1 at
    // total==0 is correct — underflow() would indeed return EOF.
    std::streamsize buffered = 0;
    if (gptr() != nullptr && egptr() != nullptr && egptr() > gptr()) {
        buffered = static_cast<std::streamsize>(egptr() - gptr());
    }
    std::streamoff remaining_off = m_file_size - m_file_offset;
    if (remaining_off < 0) {
        remaining_off = 0;
    }
    const std::streamsize remaining = remaining_off > 0 ? static_cast<std::streamsize>(remaining_off) : 0;
    const std::streamsize total = buffered + remaining;
    return total > 0 ? total : static_cast<std::streamsize>(-1);
}

// Single-threaded positional read
bool ParallelReadStreamBuf::single_read(char* dst, size_t size, size_t file_offset) {
    return positional_read(m_handle, dst, size, file_offset);
}

// Parallel positional read
bool ParallelReadStreamBuf::parallel_read(char* dst, size_t size, size_t file_offset) {
    const size_t hw_threads = (std::max)(size_t{1}, static_cast<size_t>(std::thread::hardware_concurrency()));
    const size_t max_by_chunk = size / default_parallel_io_min_chunk;
    const size_t num_threads = (std::max)(size_t{1}, (std::min)(hw_threads, max_by_chunk));

    if (num_threads == 1) {
        return single_read(dst, size, file_offset);
    }

    // Round chunk_size UP to the next 4 KiB boundary so that every thread's
    // start offset is page-aligned (better I/O coalescing on NVMe/direct I/O).
    // Because rounding up means num_threads * chunk_size >= size, two extra
    // guards are required:
    //   1. Non-last threads: cap read to min(chunk_size, size - cur_offset) so
    //      they never stride past EOF when the aligned chunk extends beyond it.
    //   2. Last thread: use (size - cur_offset) to capture every remaining byte
    //      including the fragment that lies beyond (num_threads-1) * chunk_size
    //      but before size.  Using chunk_size here would silently drop those bytes.
    size_t chunk_size = size / num_threads;
    chunk_size = (chunk_size + 4095u) & ~size_t{4095u};

    std::atomic<bool> success{true};
    // Each worker opens its own file descriptor so that the kernel's per-file-
    // description readahead state (file_ra_state / f_ra) is independent per
    // thread.  Sharing a single fd causes concurrent pread() calls to corrupt
    // each other's sequential readahead prediction, collapsing throughput from
    // ~3.5 GB/s sequential to ~0.5 GB/s.
    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    for (size_t ithr = 0; ithr < num_threads; ++ithr) {
        try {
            workers.emplace_back([&, ithr]() {
                // Wrap the entire thread body in a try-catch so that
                // any unexpected exception (e.g. std::bad_alloc from the OS path,
                // or a future code change) sets the error flag instead of calling
                // std::terminate() and killing the process.
                try {
                    const size_t cur_offset = ithr * chunk_size;
                    if (cur_offset >= size) {
                        return;  // page-alignment rounding created a surplus worker slot
                    }
                    const size_t read_size =
                        (ithr == num_threads - 1) ? (size - cur_offset) : (std::min)(chunk_size, size - cur_offset);
                    char* const ptr = dst + cur_offset;
                    const size_t thread_file_offset = file_offset + cur_offset;

                    FileHandle t_handle = open_file_for_read(m_path);
                    if (t_handle == INVALID_HANDLE_VALUE) {
                        success = false;
                        return;
                    }
                    if (!positional_read(t_handle, ptr, read_size, thread_file_offset)) {
                        success = false;
                    }
                    close_file_handle(t_handle);
                } catch (...) {
                    success = false;
                }
            });  // workers.emplace_back
        } catch (...) {
            success = false;
            break;
        }
    }
    for (auto& t : workers) {
        t.join();
    }
    return success.load();
}

}  // namespace ov::util
