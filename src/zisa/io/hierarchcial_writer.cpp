// SPDX-License-Identifier: MIT
// Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

#include <zisa/io/hierarchical_writer.hpp>

namespace zisa {

void HierarchicalWriter::write_array(const void *data,
                                     const DataType &data_type,
                                     const std::string &tag,
                                     int rank,
                                     const std::size_t *dims) {
  do_write_array(data, data_type, tag, rank, dims);
}

void HierarchicalWriter::write_scalar(const void *addr,
                                      const DataType &data_type,
                                      const std::string &tag) {
  do_write_scalar(addr, data_type, tag);
}

void HierarchicalWriter::write_string(const std::string &data,
                                      const std::string &tag) {
  do_write_string(data, tag);
}

} // namespace zisa
