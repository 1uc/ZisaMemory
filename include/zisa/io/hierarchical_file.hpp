#ifndef HIERARCHICAL_FILE_HPP_VNUEIW
#define HIERARCHICAL_FILE_HPP_VNUEIW

#include <string>
#include <vector>

namespace zisa {

enum class ErasedBasicDataType { DOUBLE, FLOAT, INT, UNSIGNED_LONG, CHAR };

namespace detail {

template <class T>
struct ErasedBasicDataTypeFactory {};

#define ZISA_CREATE_DATA_TYPE_FACTORY(type, TYPE)                              \
  template <>                                                                  \
  struct ErasedBasicDataTypeFactory<type> {                                    \
    static ErasedBasicDataType create() { return ErasedBasicDataType::TYPE; }  \
  };

ZISA_CREATE_DATA_TYPE_FACTORY(double, DOUBLE);
ZISA_CREATE_DATA_TYPE_FACTORY(float, FLOAT);
ZISA_CREATE_DATA_TYPE_FACTORY(int, INT);
ZISA_CREATE_DATA_TYPE_FACTORY(unsigned long, UNSIGNED_LONG);
ZISA_CREATE_DATA_TYPE_FACTORY(char, CHAR);

#undef ZISA_CREATE_DATA_TYPE_FACTORY
}

template <class T>
ErasedBasicDataType erase_data_type() {
  return detail::ErasedBasicDataTypeFactory<T>::create();
}

class HierarchicalFile {
public:
  using DataType = ErasedBasicDataType;

public:
  virtual ~HierarchicalFile() = default;

  /// Open a group.
  void open_group(const std::string &group_name);

  /// Close a group.
  void close_group();

  /// Close the current group and open another group
  void switch_group(const std::string &group_name);

  /// Does this group exist in the file?
  bool group_exists(const std::string &group_name) const;

  /// Human readable description of the current hierarchy.
  std::string hierarchy() const;

  /// Unlink a dataset.
  void unlink(const std::string &tag);

protected:
  virtual void do_open_group(const std::string &group_name) = 0;
  virtual void do_close_group() = 0;
  virtual void do_switch_group(const std::string &group_name) = 0;
  virtual bool do_group_exists(const std::string &group_name) const = 0;
  virtual std::string do_hierarchy() const = 0;
  virtual void do_unlink(const std::string &tag) = 0;
};

}

#endif // HIERARCHICAL_FILE_HPP
