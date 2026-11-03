// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <mdspan/mdspan.hpp>
#include <mdspan/experimental/shared_accessor.hpp>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

// This is a minimal test for a "shared mdspan"
// The idea of this feature is that you get an memory-safe mdspan,
// where from the point of construction and allocation everything is safe,
// and no out-of-bounds memory access can ever happen.
//
// Components:
// - shared_accessor: defines a checked_shared_ptr as data_handle_type
//   - can wrap a nested_accessor, but only if its stateless and has ptr as
//   data_handle
// - checked_shared_ptr: like a shared_ptr of array with tracked size
//   - operator[] can check index is valid
//   - stores also an offset ptr, enabling submdspan
// - make_shared_mdspan: construct and mdspan with shared_accessor
//   - takes constructor arguments for the mapping and forwards them
//   - construct checked_shared_ptr using the required span size of the mapping

namespace {

constexpr size_t dyn = Kokkos::dynamic_extent;

TEST(TestSharedMdspan, make_shared_mdspan) {
  auto mds = Kokkos::experimental::make_shared_mdspan<
      float, Kokkos::extents<int, dyn, 4>, Kokkos::layout_right>(
      Kokkos::extents<int, dyn, 4>(10));

  using mdspan_t = decltype(mds);
  using expected_mdspan_t =
      Kokkos::mdspan<float, Kokkos::extents<int, dyn, 4>, Kokkos::layout_right,
                     Kokkos::experimental::shared_accessor<float>>;

  static_assert(std::is_same_v<mdspan_t, expected_mdspan_t>);

  ASSERT_EQ(mds.extent(0), 10);
  ASSERT_EQ(mds.data_handle().use_count(), 1);
  ASSERT_EQ(mds.data_handle().size(), 40);
  {
    auto mds2 = mds;
    ASSERT_EQ(mds.data_handle().use_count(), 2);
    ASSERT_EQ(mds2.data_handle(), mds.data_handle());
  }
  ASSERT_EQ(mds.data_handle().use_count(), 1);
  {
    auto dh = mds.data_handle();
    ASSERT_EQ(mds.data_handle().use_count(), 2);
    ASSERT_EQ(dh, mds.data_handle());
  }
  ASSERT_EQ(mds.data_handle().use_count(), 1);
  {
    auto mds_sub = Kokkos::submdspan(mds, 3, std::pair(1, 3));
    ASSERT_EQ(mds.data_handle().use_count(), 2);
    ASSERT_NE(mds_sub.data_handle(), mds.data_handle());
    ASSERT_EQ(&mds(3, 1), mds_sub.data_handle().data());
    ASSERT_EQ(size_t(40 - 3 * 4 - 1), mds_sub.data_handle().size());
    ASSERT_EQ(&mds(3, 1), &mds_sub(0));
  }
  ASSERT_EQ(mds.data_handle().use_count(), 1);
}

}  // namespace
