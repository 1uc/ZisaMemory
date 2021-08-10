// SPDX-License-Identifier: MIT
// Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

#ifndef NETCDF_FILE_HPP_UQNOIU
#define NETCDF_FILE_HPP_UQNOIU

#include <map>
#include <stack>
#include <string>
#include <vector>

#include <zisa/config.hpp>
#include <zisa/io/concatenate.hpp>
#include <zisa/io/hierarchical_file.hpp>
#include <zisa/io/hierarchical_writer.hpp>
#include <zisa/utils/has_key.hpp>

#include <zisa/io/netcdf.hpp>

namespace zisa {

int make_netcdf_data_type(const ErasedDataType &data_type);

/// The structure of a NetCDF file.
/** NetCDF requires that the structure of the file be defined before
 *  writing any data to the file. This class abstracts a simplified view
 *  of the structure.
 */
class NetCDFFileStructure {
private:
  using dim_t = std::tuple<std::string, std::size_t>;
  using vector_dims_t = std::vector<dim_t>;

  using var_t
      = std::tuple<std::string, std::vector<std::string>, ErasedDataType>;
  using vector_vars_t = std::vector<var_t>;

public:
  NetCDFFileStructure(const vector_dims_t &dims, const vector_vars_t &vars);

  /// Does the structure appear to be plausible?
  bool is_valid() const;

  const auto &dims() const;
  const auto &vars() const;

private:
  void add_dim(const dim_t &dim);
  void add_dims(const vector_dims_t &dims);

  void add_vars(const vector_vars_t &vars);
  void add_var(const var_t &var);

private:
  vector_dims_t dims_;
  vector_vars_t vars_;
};

class NetCDFFile : public virtual HierarchicalFile {
public:
  NetCDFFile(const std::string &filename, const NetCDFFileStructure &structure);

  virtual ~NetCDFFile() override;

protected:
  virtual void do_open_group(const std::string & /* group_name */) override;
  virtual void do_close_group() override;
  virtual void do_switch_group(const std::string &group_name) override;

  virtual bool
  do_group_exists(const std::string & /* group_name */) const override;

  virtual std::string do_hierarchy() const override;
  virtual void do_unlink(const std::string &tag) override;

private:
  std::vector<int>
  extract_dim_ids(const std::vector<std::string> &dim_names) const;

protected:
  std::map<std::string, int> dim_ids_; ///< NetCDF ID for a given dimension.
  std::map<std::string, int> var_ids_; ///< NetCDF ID for a given variable.

  std::stack<int> file_ids_;      ///< NetCDF file/group identifiers (branch)
  std::vector<std::string> path_; ///< NetCDF path
};

} // namespace zisa
#endif // NETCDF_FILE_HPP
