#ifndef ZISAMEMORY_COPY_BYTES_HPP
#define ZISAMEMORY_COPY_BYTES_HPP

#include <zisa/config.hpp>
#include <zisa/memory/device_type.hpp>

namespace zisa {

void copy_bytes(void * dst, const device_type &dst_loc, void * src, const device_type &src_loc, std::size_t n_bytes);

}

#endif // ZISAMEMORY_COPY_BYTES_HPP
