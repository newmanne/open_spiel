#ifndef OPEN_SPIEL_GAMES_CARTESIAN_H
#define OPEN_SPIEL_GAMES_CARTESIAN_H

//  :copyright: (c) 2017 Alex Huszagh.
//  :license: MIT.
/**
 *  Cartesian product algorithm.
 *
 *  Lazily calculates the cartesian product from a container of containers,
 *  of either linear (array, vector, set, unordered_set) or associative
 *  (map, unordered_map, multimap, unordered_multimap) containers.
 *
 *  The code can be used as follows:
 *
 *      std::vector<std::vector<int>> aa = {{1, 2, 3}, {4, 5, 6}};
 *      product(aa, [](const auto &i) {
 *          std::cout << "[";
 *          for (auto j : i) {
 *              std::cout << j << ", ";
 *          }
 *          std::cout << "]" << std::endl;
 *      });
 *
 *  The above example would print:
 *      [1, 4, ]
 *      [1, 5, ]
 *      [1, 6, ]
 *      [2, 4, ]
 *      [2, 5, ]
 *      [2, 6, ]
 *      [3, 4, ]
 *      [3, 5, ]
 *      [3, 6, ]
 *
 *  Permission is hereby granted, free of charge, to any person
 *  obtaining a copy of this software and associated documentation
 *  files (the “Software”), to deal in the Software without
 *  restriction, including without limitation the rights to use,
 *  copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the
 *  Software is furnished to do so, subject to the following
 *  conditions:
 *
 *  The above copyright notice and this permission notice shall be
 *  included in all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
 *  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 *  OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 *  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 *  HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 *  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 *  OTHER DEALINGS IN THE SOFTWARE.
 */

#include <algorithm>
#include <functional>
#include <vector>

// SFINAE
// ------

template <typename T>
struct is_reference_wrapper: std::false_type
{};

template <typename T>
struct is_reference_wrapper<std::reference_wrapper<T>>: std::true_type
{};

template <typename T>
constexpr bool is_reference_wrapper_v = is_reference_wrapper<T>::value;

template <bool B, typename T, typename F>
using conditional_t = typename std::conditional<B, T, F>::type;

template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

template <typename T>
using type_t = typename T::type;

template <typename T>
using value_t = typename T::value_type;

template <typename T>
using iterator_t = typename T::iterator;

template <typename T>
using const_iterator_t = typename T::const_iterator;

template <typename T>
using iterator_value_t = value_t<std::iterator_traits<T>>;

template <typename T>
using reference_t = typename T::reference_type;

template <typename T>
using mapped_t = typename T::mapped_type;

template <typename T>
using list_t = typename T::list_type;

template <typename T>
using matrix_t = typename T::matrix_type;

#define HAS_MEMBER_TYPE(member, name)                                        \
    template<typename T>                                                     \
    class name                                                               \
    {                                                                        \
    protected:                                                               \
        struct U {};                                                         \
        typedef conditional_t<std::is_class<T>::value, T, U> V;              \
                                                                             \
        struct Fallback                                                      \
        {                                                                    \
            struct member {};                                                \
        };                                                                   \
                                                                             \
        struct Derived: V, Fallback                                          \
        {};                                                                  \
                                                                             \
        template <typename C> static char &test(typename C::member*);        \
        template <typename C> static long &test(...);                        \
                                                                             \
    public:                                                                  \
        enum {                                                               \
            value = sizeof(test<Derived>(0)) == sizeof(long)                 \
        };                                                                   \
    };

HAS_MEMBER_TYPE(mapped_type, has_mapped_type);

template <typename T>
constexpr bool has_mapped_type_v = has_mapped_type<T>::value;

// FUNCTIONS
// ---------

namespace detail
{
// PRODUCT
// -------

/** \brief Helper demangle and product reference types.
 *
 *  Due to the various levels of indirection with our wrapper classes,
 *  we can have the inner container be a reference, or a value, and the
 *  `value_type` of that container be a reference, or a value.
 *
 *  Since we want the output to be a vector of reference wrappers,
 *  appropriately detect when we are using references and when we
 *  are using values.
 */
template <typename T>
struct ReferenceHelper
{
    typedef conditional_t<is_reference_wrapper_v<T>, T, std::reference_wrapper<const T>> type;
    typedef type_t<type> container_type;
    typedef value_t<container_type> value_type;
    typedef const_iterator_t<container_type> iterator;

    typedef conditional_t<
        is_reference_wrapper_v<value_type>,
        value_type,
        std::reference_wrapper<const value_type>
    > reference_type;

    static iterator begin(const container_type &t)
    {
        return t.begin();
    }

    static iterator end(const container_type &t)
    {
        return t.end();
    }
};


/** \brief Calculate cartesian product with zero data copies.
 */
template <
    typename BidirIter,
    typename Function
>
void product_impl(BidirIter first,
    BidirIter last,
    Function function)
{
    typedef ReferenceHelper<iterator_value_t<BidirIter>> Helper;
    typedef iterator_t<Helper> Iterator;
    typedef reference_t<Helper> Reference;

    // sanity check
    if (first == last) {
        return;
    }

    // fill vector for function calls
    const size_t size = last - first;
    std::vector<Iterator> iterators;
    std::vector<Reference> values;
    iterators.reserve(size);
    values.reserve(size);
    std::for_each(first, last, [&](const auto &value) {
        iterators.emplace_back(--Helper::begin(value));
        values.emplace_back(*Helper::begin(value));
    });

    // iterate over all elements
    auto it = first;
    while (it >= first) {
        const size_t k = it - first;
        if (it == last) {
            function(values);
            --it;
        } else if (iterators[k] != --Helper::end(*it)) {
            values[k] = *++iterators[k];
            ++it;
        } else {
            iterators[k] = --Helper::begin(*it);
            --it;
        }
    }
}


// LINEAR - LINEAR
// ---------------


/** \brief Type detection for linear collection of linear collections.
 */
template <typename T>
struct LinearLinearHelper
{
    typedef value_t<T> first_type;
    typedef value_t<first_type> second_type;
    typedef std::reference_wrapper<const second_type> reference_type;
    typedef std::vector<reference_type> list_type;
    typedef std::vector<list_type> matrix_type;
};

/** \brief Product wrapper for linear collection of linear collections.
 */
template <
    typename Container,
    typename Function
>
void linearLinear(const Container &container,
    Function function)
{
    typedef LinearLinearHelper<Container> Helper;
    typedef typename Helper::list_type List;
    typedef typename Helper::matrix_type Matrix;

    Matrix matrix;
    matrix.reserve(container.size());
    for (const auto &outer: container) {
        List list;
        list.reserve(outer.size());
        for (const auto &inner: outer) {
            list.emplace_back(std::cref(inner));
        }
        matrix.emplace_back(std::move(list));
    }

    product_impl(matrix.begin(), matrix.end(), function);
}


// LINEAR - ASSOCIATIVE
// --------------------

/** \brief Type detection for linear collection of associative collections.
 */
template <typename T>
struct LinearAssociativeHelper
{
    typedef value_t<T> first_type;
    typedef mapped_t<first_type> second_type;
    typedef std::reference_wrapper<const second_type> reference_type;
    typedef std::vector<reference_type> list_type;
    typedef std::vector<list_type> matrix_type;
};


/** \brief Product wrapper for linear collection of associative collections.
 */
template <
    typename Container,
    typename Function
>
void linearAssociative(const Container &container,
    Function function)
{
    typedef LinearAssociativeHelper<Container> Helper;
    typedef typename Helper::list_type List;
    typedef typename Helper::matrix_type Matrix;

    Matrix matrix;
    matrix.reserve(container.size());
    for (const auto &outer: container) {
        List list;
        list.reserve(outer.size());
        for (const auto &inner: outer) {
            list.emplace_back(std::cref(inner.second));
        }
        matrix.emplace_back(std::move(list));
    }

    product_impl(matrix.begin(), matrix.end(), function);
}


// ASSOCIATIVE - LINEAR
// --------------------


/** \brief Type detection for associative collection of linear collections.
 */
template <typename T>
struct AssociativeLinearHelper
{
    typedef mapped_t<T> first_type;
    typedef value_t<first_type> second_type;
    typedef std::reference_wrapper<const second_type> reference_type;
    typedef std::vector<reference_type> list_type;
    typedef std::vector<list_type> matrix_type;
};


/** \brief Product wrapper for associative collection of linear collections.
 */
template <
    typename Container,
    typename Function
>
void associativeLinear(const Container &container,
    Function function)
{
    typedef AssociativeLinearHelper<Container> Helper;
    typedef list_t<Helper> List;
    typedef matrix_t<Helper> Matrix;

    Matrix matrix;
    matrix.reserve(container.size());
    for (const auto &outer: container) {
        List list;
        list.reserve(outer.second.size());
        for (const auto &inner: outer.second) {
            list.emplace_back(std::cref(inner));
        }
        matrix.emplace_back(std::move(list));
    }

    product_impl(matrix.begin(), matrix.end(), function);
}


// ASSOCIATIVE - ASSOCIATIVE
// -------------------------


/** \brief Type detection for associative collection of associative collections.
 */
template <typename T>
struct AssociativeAssociativeHelper
{
    typedef mapped_t<T> first_type;
    typedef mapped_t<first_type> second_type;
    typedef std::reference_wrapper<const second_type> reference_type;
    typedef std::vector<reference_type> list_type;
    typedef std::vector<list_type> matrix_type;
};


/** \brief Product wrapper for associative collection of associative collections.
 */
template <
    typename Container,
    typename Function
>
void associativeAssociative(const Container &container,
    Function function)
{
    typedef AssociativeAssociativeHelper<Container> Helper;
    typedef typename Helper::list_type List;
    typedef typename Helper::matrix_type Matrix;

    Matrix matrix;
    matrix.reserve(container.size());
    for (const auto &outer: container) {
        List list;
        list.reserve(outer.second.size());
        for (const auto &inner: outer.second) {
            list.emplace_back(std::cref(inner.second));
        }
        matrix.emplace_back(std::move(list));
    }

    product_impl(matrix.begin(), matrix.end(), function);
}


// SFINAE
// ------

/** \brief Associative-like type-wrapper.
 */
template <typename T>
struct AssociativeLike
{
    typedef value_t<T> mapped_type;
};


/** \brief Base class to check if it contains associative containers.
 */
template <typename T>
struct IsAssociativeImpl
{
    static constexpr bool outer = has_mapped_type_v<T>;
    typedef mapped_t<conditional_t<outer, T, AssociativeLike<T>>> U;
    static constexpr bool inner = has_mapped_type_v<U>;
};


/** \brief Check if linear collection of linear collections.
 */
template <typename T>
struct IsLinearLinear
{
    enum {
        value = !(IsAssociativeImpl<T>::outer || IsAssociativeImpl<T>::inner)
    };
};


/** \brief Check if linear collection of associative collections.
 */
template <typename T>
struct IsLinearAssociative
{
    enum {
        value = !IsAssociativeImpl<T>::outer && IsAssociativeImpl<T>::inner
    };
};


/** \brief Check if associative collection of linear collections.
 */
template <typename T>
struct IsAssociativeLinear
{
    enum {
        value = IsAssociativeImpl<T>::outer && !IsAssociativeImpl<T>::inner
    };
};


/** \brief Check if associative collection of associative collections.
 */
template <typename T>
struct IsAssociativeAssociative
{
    enum {
        value = IsAssociativeImpl<T>::outer && IsAssociativeImpl<T>::inner
    };
};


template <typename T>
constexpr bool IsLinearLinearV = IsLinearLinear<T>::value;

template <typename T>
constexpr bool IsLinearAssociativeV = IsLinearAssociative<T>::value;

template <typename T>
constexpr bool IsAssociativeLinearV = IsAssociativeLinear<T>::value;

template <typename T>
constexpr bool IsAssociativeAssociativeV = IsAssociativeAssociative<T>::value;


// PRODUCT
// -------


/** \brief Deduce container type and invoke correct dispatcher.
 */
struct ProducerHelper
{
    template <typename T, typename Function>
    enable_if_t<IsLinearLinearV<T>, void>
    static product(const T &t,
        Function function)
    {
        return detail::linearLinear(t, function);
    }

    template <typename T, typename Function>
    enable_if_t<IsLinearAssociativeV<T>, void>
    static product(const T &t,
        Function function)
    {
        return detail::linearAssociative(t, function);
    }

    template <typename T, typename Function>
    enable_if_t<IsAssociativeLinearV<T>, void>
    static product(const T &t,
        Function function)
    {
        return detail::associativeLinear(t, function);
    }

    template <typename T, typename Function>
    enable_if_t<IsAssociativeAssociativeV<T>, void>
    static product(const T &t,
        Function function)
    {
        return detail::associativeAssociative(t, function);
    }
};


/** \brief Invoke the appropriate product function.
 */
template <
    typename Container,
    typename Function
>
void product(const Container &container,
    Function function)
{
    return ProducerHelper::product(container, function);
}

}   /* detail */


// PRODUCT
// -------


/** \brief Wrapper for implied cartesian product.
 */
template <
    typename Container,
    typename Function
>
void product(const Container &container,
    Function function)
{
    return detail::product(container, function);
}

#endif  // OPEN_SPIEL_GAMES_CARTESIAN_H
