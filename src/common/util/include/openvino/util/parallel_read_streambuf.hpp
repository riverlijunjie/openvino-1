// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for definition of a parallel I/O streambuf for file-based reads.
 * @file parallel_read_streambuf.hpp
 */

#pragma once

#include <filesystem>
#include <memory>
#include <streambuf>

#include "openvino/util/parallel_io.hpp"

namespace ov::util {

/**
 * @brief A std::streambuf that reads from a file using parallel I/O for large
 *        reads, bypassing the OS page cache pressure that mmap+memcpy incurs.
 *
 * For reads >= threshold bytes, the read is split across N threads where each
 * thread issues its own independent positional read operation using
 * platform-specific APIs. Smaller reads fall through to a single positional call.
 *
 * Usage:
 * @code
 *   ParallelReadStreamBuf buf(cache_path, blob_offset_in_file);
 *   std::istream stream(&buf);
 *   cldnn::BinaryInputBuffer ib(stream, engine);
 *   ib >> ...;
 * @endcode
 */
class ParallelReadStreamBuf : public std::streambuf {
public:
    /**
     * @param path           Path to the file to read.
     * @param header_offset  Initial file position (absolute offset from the start
     *                       of the file; the stream starts reading from here).
     * @param threshold      Minimum read size to trigger parallel I/O.
     */
    explicit ParallelReadStreamBuf(const std::filesystem::path& path,
                                   std::streamoff header_offset = 0,
                                   size_t threshold = default_parallel_io_threshold);

    ~ParallelReadStreamBuf() override;

    ParallelReadStreamBuf(const ParallelReadStreamBuf&) = delete;
    ParallelReadStreamBuf& operator=(const ParallelReadStreamBuf&) = delete;

protected:
    std::streamsize xsgetn(char_type* dst, std::streamsize n) override;
    int_type underflow() override;
    pos_type seekoff(off_type off, std::ios_base::seekdir way, std::ios_base::openmode which) override;
    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override;
    std::streamsize showmanyc() override;

private:
    bool single_read(char* dst, size_t size, size_t file_offset);
    bool parallel_read(char* dst, size_t size, size_t file_offset);
    /// Try to serve @p n bytes from the read-ahead buffer.  Returns the number
    /// of bytes copied (may be 0 if the buffer doesn't cover the current offset).
    std::streamsize serve_from_readahead(char_type*& dst, std::streamsize& n);
    /// Refill the read-ahead buffer starting at m_file_offset.
    bool refill_readahead();

    std::filesystem::path m_path;
    FileHandle m_handle;  ///< platform file handle

    std::streamoff m_file_offset = 0;    ///< absolute file offset of next byte to read
    std::streamoff m_header_offset = 0;  ///< absolute file offset of logical stream start
    std::streamoff m_file_size = 0;
    size_t m_threshold = default_parallel_io_threshold;

    std::unique_ptr<char_type[]> m_ra_buf;  ///< lazily allocated read-ahead buffer
    size_t m_ra_buf_start = 0;  ///< absolute file offset of first byte in m_ra_buf
    size_t m_ra_buf_end = 0;    ///< absolute file offset one past last valid byte
};

}  // namespace ov::util