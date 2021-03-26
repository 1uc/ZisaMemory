#include <zisa/io/hierarchical_reader.hpp>

namespace zisa {

void HierarchicalReader::read_array(void *data,
                                    const DataType &data_type,
                                    const std::string &tag) const {
  do_read_array(data, data_type, tag);
}
void HierarchicalReader::read_scalar(void *data,
                                     const DataType &data_type,
                                     const std::string &tag) const {
  do_read_scalar(data, data_type, tag);
}

std::string HierarchicalReader::read_string(const std::string &tag) const {
  return do_read_string(tag);
}

}
