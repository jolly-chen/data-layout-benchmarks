#ifndef LAYOUT_HELPERS_H
#define LAYOUT_HELPERS_H

// Helpers and concepts for the containers that generate_datastructures.py
// emits into namespace layouts::<Struct>:
// - AlignSize, StorageDeleter: used by the generated containers
// - LayoutOf<C, T>: the interface the benchmark driver expects from a container
// - ForEachLayout<T>(f): calls f for every generated container of T

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <meta>
#include <new>
#include <span>
#include <string>
#include <vector>

/* Round size up to the next multiple of alignment. */
constexpr size_t AlignSize(size_t size, size_t alignment) {
  return (size + alignment - 1) / alignment * alignment;
}

/* Deleter for memory obtained from std::aligned_alloc. */
template <typename T>
struct StorageDeleter {
  void operator()(T *ptr) const { std::free(ptr); }
};

/* The interface the benchmark driver expects from a container that stores n
 * elements of T in some memory layout. Any layout backend that satisfies it
 * can be benchmarked; the containers emitted by generate_datastructures.py are
 * one such backend.
 *
 * - C(n, alignment) allocates n elements, each partition aligned to alignment.
 * - c[i] returns a proxy reference whose members are named like those of T,
 *   and that can be assigned from a T.
 * - C::bytes_for_all(n, alignment) is the number of bytes C(n, alignment)
 *   allocates, including alignment padding.
 * - C::to_string() names the layout in benchmark names.
 */
template <class C, class T>
concept LayoutOf =
    std::same_as<typename C::value_type, T> &&
    std::constructible_from<C, size_t, size_t> &&
    requires(C c, const T &t, size_t i) {
      c[i] = t;
      { c.size() } -> std::convertible_to<size_t>;
      { C::bytes_for_all(i, i) } -> std::convertible_to<size_t>;
      { C::to_string() } -> std::convertible_to<std::string>;
    };

/* The generated containers of each struct are in namespace layouts::<Struct>. */
namespace layouts {}

namespace layout_helpers_detail {
/* The namespaces in namespace layouts, one per struct with generated layouts.
 * Evaluated where ForEachLayout<T> is instantiated, so it sees every
 * layouts_<Struct>.h included before that point. */
template <class T>
consteval std::vector<std::meta::info> LayoutNamespaces() {
  std::vector<std::meta::info> namespaces;
  for (auto m : std::meta::members_of(^^layouts, std::meta::access_context::unchecked())) {
    if (std::meta::is_namespace(m)) { namespaces.push_back(m); }
  }
  return namespaces;
}
} // namespace layout_helpers_detail

/* Call f.template operator()<C>() for every generated container C that stores
 * elements of T, i.e., every C in a namespace layouts::<Struct> with
 * LayoutOf<C, T>. Containers for other structs are skipped. */
template <class T, class F>
void ForEachLayout(F &&f) {
  template for (constexpr std::meta::info ns :
                std::define_static_array(layout_helpers_detail::LayoutNamespaces<T>())) {
    template for (constexpr std::meta::info c : std::define_static_array(
                      members_of(ns, std::meta::access_context::unchecked()))) {
      if constexpr (std::meta::is_type(c)) {
        if constexpr (LayoutOf<typename[:c:], T>) {
          f.template operator()<typename[:c:]>();
        }
      }
    }
  }
}

#endif // LAYOUT_HELPERS_H
