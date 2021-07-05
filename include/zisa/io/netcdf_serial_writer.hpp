// SPDX-License-Identifier: MIT
// Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

#ifndef NETCDF_SERIAL_WRITER_HPP_XWYKVGMU
#define NETCDF_SERIAL_WRITER_HPP_XWYKVGMU

#include <zisa/io/netcdf_writer.hpp>

namespace zisa {

class NetCDFSerialWriter : public virtual NetCDFWriter {
public:
  using NetCDFWriter::NetCDFWriter;

protected:
  virtual void do_write_array(void const *data,
                              const std::string &tag) override;

  virtual void do_write_scalar(void const *addr,
                               const std::string &tag) override;
};

}
#endif /* end of include guard: NETCDF_SERIAL_WRITER_HPP_XWYKVGMU */
