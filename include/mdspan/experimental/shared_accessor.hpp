// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include "../impl/mdspan.hpp"

#include <cstddef>  // size_t
#include <type_traits>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace experimental {

template <class ElementType,
          class NestedAccessor = default_accessor<ElementType>>
struct shared_accessor;

template <class ElementType>
class checked_shared_ptr {
  using base_ptr_t   = std::shared_ptr<ElementType>;
  using offset_ptr_t = ElementType*;
  base_ptr_t m_base_ptr;
  ElementType* m_ptr;
  size_t m_size;

  template <class OtherElementType, class NestedAccessor>
  friend struct shared_accessor;

 public:
  using element_type = ElementType;

  checked_shared_ptr(size_t N)
    requires(!std::is_const_v<ElementType>)
  {
    m_base_ptr = base_ptr_t(new ElementType[N]{});
    m_ptr      = m_base_ptr.get();
    m_size     = N;
  }

  template <class OtherElementType>
    requires(std::is_convertible_v<OtherElementType (*)[], element_type (*)[]>)
  checked_shared_ptr(const checked_shared_ptr<OtherElementType>& src)
      : m_base_ptr(src.m_base_ptr), m_ptr(src.m_ptr), m_size(src.m_size) {}

  template <class OtherElementType>
    requires(std::is_convertible_v<OtherElementType (*)[], element_type (*)[]>)
  checked_shared_ptr(const checked_shared_ptr<OtherElementType>& src,
                     size_t offset)
      : m_base_ptr(src.m_base_ptr) {
    MDSPAN_IMPL_PRECONDITION(offset > m_size && "Too large offset");

    m_ptr = src.m_ptr + offset, m_size = src.m_size - offset;
  }

  ElementType& operator[](size_t i) const {
    MDSPAN_IMPL_PRECONDITION(i < m_size && "Out of bounds access");
    return m_ptr[i];
  }

  checked_shared_ptr operator+(size_t offset) const {
    return checked_shared_ptr(*this, offset);
  }

  ElementType* data() const { return m_ptr; }
  size_t size() const { return m_size; }
  size_t use_count() const { return m_base_ptr.use_count(); }

  template <class OtherElementType>
  bool operator==(const checked_shared_ptr<OtherElementType>& other) const {
    return (other.m_base_ptr == m_base_ptr) && (other.m_ptr == m_ptr) &&
           (other.m_size == m_size);
  }
};

template <class ElementType, class NestedAccessor>
struct shared_accessor {
  static_assert(
      std::is_same_v<typename NestedAccessor::data_handle_type, ElementType*>);
  static_assert(std::is_empty_v<NestedAccessor>);

  using offset_policy =
      shared_accessor<ElementType, typename NestedAccessor::offset_policy>;
  using element_type     = ElementType;
  using reference        = typename NestedAccessor::reference;
  using data_handle_type = checked_shared_ptr<ElementType>;

  MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr shared_accessor() noexcept =
      default;

  template <class OtherElementType, class OtherNestedAccessor>
    requires(std::is_convertible_v<OtherNestedAccessor, NestedAccessor>)
  MDSPAN_INLINE_FUNCTION constexpr shared_accessor(
      shared_accessor<OtherElementType, OtherNestedAccessor>) noexcept {}

  MDSPAN_INLINE_FUNCTION
  constexpr data_handle_type offset(const data_handle_type& p,
                                    size_t i) const noexcept {
    // invalid offset will be checked by checked_shared_ptr, so we don't have to
    // here
    return p + i;
  }

  MDSPAN_FORCE_INLINE_FUNCTION
  constexpr reference access(const data_handle_type& p,
                             size_t i) const noexcept {
    MDSPAN_IMPL_PRECONDITION(i < p.size() && "Out of bounds access");
    return nested_accessor.access(p.m_ptr, i);
  }

 private:
  NestedAccessor nested_accessor{};
};

template <class ElementType, class Extents, class LayoutType,
          class... MappingArgs>
auto make_shared_mdspan(const MappingArgs&... args) {
  using mdspan_t =
      mdspan<ElementType, Extents, LayoutType, shared_accessor<ElementType>>;
  using map_t         = typename mdspan_t::mapping_type;
  using data_handle_t = typename mdspan_t::data_handle_type;

  static_assert(std::is_constructible_v<map_t, MappingArgs...>);

  map_t map(args...);
  return mdspan_t(data_handle_t(map.required_span_size()), map);
}

}  // end namespace experimental
}  // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE
