// SPDX-License-Identifier: MIT
// Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

#ifndef NETCDF_WRITER_HPP_BCEJSJGO
#define NETCDF_WRITER_HPP_BCEJSJGO

#include <zisa/io/hierarchical_writer.hpp>
#include <zisa/io/netcdf_file.hpp>

namespace zisa {

class NetCDFWriter : public virtual NetCDFFile,
                     public virtual HierarchicalWriter {

public:
  using NetCDFFile::NetCDFFile;

protected:
  // Redirects `HierarchicalWriter::do_write_array` to the NetCDF
  // implementation.
  virtual void do_write_array(void const *data,
                              const DataType & /* data_type */,
                              const std::string &tag,
                              int /* rank */,
                              std::size_t const * /* dims */) override;

  // Redirects `HierarchicalWriter::do_write_array` to the NetCDF
  // implementation.
  virtual void do_write_scalar(void const *addr,
                               const DataType & /* data_type */,
                               const std::string &tag) override;

  // Provides the interface in the NetCDF world which needs to be implemented.
  virtual void do_write_array(void const *data, const std::string &tag) = 0;

  // Provides the interface in the NetCDF world which needs to be implemented.
  virtual void do_write_scalar(void const *addr, const std::string &tag) = 0;

  virtual void do_write_string(const std::string &data,
                               const std::string &tag) override;
};

}

#endif /* end of include guard: NETCDF_WRITER_HPP_BCEJSJGO */
