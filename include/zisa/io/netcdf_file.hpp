#ifndef NETCDF_FILE_HPP_UQNOIU
#define NETCDF_FILE_HPP_UQNOIU

#include <map>
#include <string>
#include <vector>

#include <zisa/config.hpp>
#include <zisa/io/hierarchical_file.hpp>
#include <zisa/utils/has_key.hpp>

#include <zisa/io/netcdf.hpp>

namespace zisa {

/// The structure of a NetCDF file.
/** NetCDF requires that the structure of the file be defined before
 *  writing any data to the file. This class abstracts a simpified view
 *  of the structure.
 */
class NetCDFFileStructure {
private:
  using vector_dims_t = std::vector<std::tuple<std::string, int>>;
  using vector_vars_t = std::vector<
      std::tuple<std::string, std::vector<std::string>, ErasedBasicDataType>>;

public:
  NetCDFFileStructure(const vector_dims_t &dims, const vector_vars_t &vars) {
    add_dims(dims);
    add_vars(vars);
  }

  /// Does the structure appear to be plausible?
  bool is_valid() const;

  const auto &dims() const { return dims_; }
  const auto &vars() const { return vars_; }

private:
  void add_dim(const std::string &name, int extent) {
    LOG_ERR_IF(has_key(dims_, name), "Redefining a dimension.");
    dims_[name] = extent;
  }

  void add_dims(const vector_dims_t &dims) {
    for (const auto &[name, extent] : dims) {
      add_dim(name, extent);
    }
  }

  void add_vars(const vector_vars_t &vars) {
    for (const auto &[name, params] : vars) {
      add_var(name, std::get<0>(params), std::get<1>(params));
    }
  }

  void add_var(const std::string &name, const std::vector<std::string> &dims,
               const ErasedBasicDataType &data_type) {

    LOG_ERR_IF(has_key(vars_, name), "Redefining a variable.");
    vars_[name] = std::tuple{dims, data_type};
  }

private:
  std::map<std::string, int> dims_;
  std::map<std::string,
           std::tuple<std::vector<std::string>, ErasedBasicDataType>>
      vars_;
};

class NetCDFFile {
public:
  NetCDFFile(const std::string &filename,
             const NetCDFFileStructure &structure) {
    auto file_id = zisa::io::netcdf::create(filename, NC_CLOBBER | NC_NETCDF4);

    const auto &dims = structure.dims();
    for (const auto &[name, extent] : dims) {
      zisa::io::netcdf::def_dim(file_id, name, extent);
    }

    const auto &vars = structure.vars();
    for (const auto &[name, dim] : vars) {
      zisa::io::netcdf::def_var(file_id, name, extent);
    }
  }
};

} // namespace zisa
#endif // NETCDF_FILE_HPP
