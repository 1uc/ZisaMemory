#include <zisa/io/hierarchical_file.hpp>

namespace zisa {

void HierarchicalFile::open_group(const std::string &group_name) {
  return do_open_group(group_name);
}

void HierarchicalFile::close_group() { do_close_group(); }

void HierarchicalFile::switch_group(const std::string &group_name) {
  do_switch_group(group_name);
}

bool HierarchicalFile::group_exists(const std::string &group_name) const {
  return do_group_exists(group_name);
}

std::string HierarchicalFile::hierarchy() const { return do_hierarchy(); }

void HierarchicalFile::unlink(const std::string &tag) { do_unlink(tag); }

}
