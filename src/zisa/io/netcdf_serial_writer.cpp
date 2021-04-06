#if ZISA_HAS_NETCDF == 1

#include <zisa/io/netcdf_serial_writer.hpp>

namespace zisa {

void NetCDFSerialWriter::do_write_array(void const *data,
                                        const std::string &tag) {
  auto id = file_ids_.top();
  auto var_id = var_ids_.at(tag);
  zisa::io::netcdf::put_var(id, var_id, data);
}

void NetCDFSerialWriter::do_write_scalar(void const *addr,
                                         const std::string &tag) {
  do_write_array(addr, tag);
}

}

#endif
