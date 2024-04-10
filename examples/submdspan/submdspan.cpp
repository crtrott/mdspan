#include<mdspan/mdspan.hpp>
#include<cassert>
#include<vector>
#include<utility>

int main(int argc, char* argv[]) {

  using ext_t = Kokkos::dextents<int, 6>;
  Kokkos::layout_left::mapping<ext_t> map(ext_t(10,10,10,10,10,10));

  std::vector<int> data(map.required_span_size());
  Kokkos::mdspan a(data.data(), map);

  for(int i=0; i<data.size(); i++) data[i] = i;
  
  int counter = 0;
  for(int i5=0; i5<a.extent(0); i5++)
  for(int i4=0; i4<a.extent(0); i4++)
  for(int i3=0; i3<a.extent(0); i3++)
  for(int i2=0; i2<a.extent(0); i2++)
  for(int i1=0; i1<a.extent(0); i1++)
  for(int i0=0; i0<a.extent(0); i0++) {
    assert((a[i0,i1,i2,i3,i4,i5] == i0 + i1*10 + i2*100 + i3*1000 + i4*10000 + i5*100000));
  }

  {
    auto as = Kokkos::submdspan(a, Kokkos::full_extent, Kokkos::full_extent, std::pair{1,4}, 2, 3, 4);
    assert((as[0,0,0] == 432100));
    assert((as[1,1,1] == 432211));
    static_assert(std::is_same_v<decltype(as)::layout_type,Kokkos::layout_left>);
  }
  {
    auto as = Kokkos::submdspan(a, std::pair{1,9}, Kokkos::full_extent, std::pair{1,4}, 2, 3, 4);
    assert((as[0,0,0] == 432101));
    assert((as[1,1,1] == 432212));
    static_assert(std::is_same_v<decltype(as)::layout_type,Kokkos::Experimental::layout_left_padded<Kokkos::dynamic_extent>>);
  }
  {
    auto as = Kokkos::submdspan(a, std::pair{1,9}, 2, Kokkos::full_extent, std::pair{1,4}, 3, 4);
    assert((as[0,0,0] == 431021));
    assert((as[1,1,1] == 432122));
    static_assert(std::is_same_v<decltype(as)::layout_type,Kokkos::Experimental::layout_left_padded<Kokkos::dynamic_extent>>);
  }
  {
    auto as = Kokkos::submdspan(a, Kokkos::full_extent, 2, Kokkos::full_extent, std::pair{1,4}, 3, 4);
    assert((as[0,0,0] == 431020));
    assert((as[1,1,1] == 432121));
    static_assert(std::is_same_v<decltype(as)::layout_type,Kokkos::Experimental::layout_left_padded<Kokkos::dynamic_extent>>);
  }
  {
    auto as = Kokkos::submdspan(a, Kokkos::full_extent, 2, 3, Kokkos::full_extent, std::pair{1,4}, 4);
    assert((as[0,0,0] == 410320));
    assert((as[1,1,1] == 421321));
    static_assert(std::is_same_v<decltype(as)::layout_type,Kokkos::Experimental::layout_left_padded<Kokkos::dynamic_extent>>);
  }
  {
    auto as = Kokkos::submdspan(a, std::pair{1,9}, 2, std::pair{7,9}, std::pair{1,4}, 3, 4);
    assert((as[0,0,0] == 431721));
    assert((as[1,1,1] == 432822));
    static_assert(std::is_same_v<decltype(as)::layout_type,Kokkos::layout_stride>);
  }
  {
    auto as = Kokkos::submdspan(a, Kokkos::full_extent, 2, std::pair{7,9}, std::pair{1,4}, 3, 4);
    assert((as[0,0,0] == 431720));
    assert((as[1,1,1] == 432821));
    static_assert(std::is_same_v<decltype(as)::layout_type,Kokkos::layout_stride>);
  }
  return 0;
}
