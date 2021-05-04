#if ZISA_HAS_NETCDF

#include <zisa/io/netcdf_writer.hpp>

namespace zisa {

void NetCDFWriter::do_write_array(void const *data,
                                  const DataType & /* data_type */,
                                  const std::string &tag,
                                  int /* rank */,
                                  std::size_t const * /* dims */) {
  do_write_array(data, tag);
}

void NetCDFWriter::do_write_scalar(void const *addr,
                                   const DataType & /* data_type */,
                                   const std::string &tag) {
  do_write_scalar(addr, tag);
}
void NetCDFWriter::do_write_string(const std::string & /* data */,
                                   const std::string & /* tag */) {
  LOG_ERR("Implement first.");
}
}

#endif
