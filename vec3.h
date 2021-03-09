#pragma once
#include <cmath>
#include <ostream>
#include <type_traits>

template <typename T> class vec3 {
public:
    T x, y, z;

    vec3() : x(0), y(0), z(0) {}
    vec3(T a) : x(a), y(a), z(a) {}
    vec3(T vx, T vy, T zy) : x(vx), y(vy), z(zy) {}

    vec3 operator+(const vec3& rhs) const;
    vec3& operator+=(const vec3& rhs);
    vec3 operator-(const vec3& rhs) const;
    vec3& operator-=(const vec3& rhs);

    template <typename V> vec3 operator*(const V& rhs) const;
    template <typename V> vec3& operator*=(const V& rhs);
    template <typename V> vec3 operator/(const V& rhs) const;
    template <typename V> vec3& operator/=(const V& rhs);

    template <typename V> vec3 operator*(const vec3<V>& rhs) const;
    template <typename V> vec3& operator*=(const vec3<V>& rhs);
    template <typename V> vec3 operator/(const vec3<V>& rhs) const;
    template <typename V> vec3& operator/=(const vec3<V>& rhs);

    bool operator==(const vec3& rhs) const;
    bool operator!=(const vec3& rhs) const;
    bool operator<(const vec3& rhs) const;

    template <typename V> explicit operator vec3<V>() const;

    inline T& operator[](size_t d)
    {
        return reinterpret_cast<double*>(this)[d];
    }
    inline const T& operator[](size_t d) const
    {
        return reinterpret_cast<const double*>(this)[d];
    }

    T length2() const;
    double length() const;

    template <typename T1 = T,
              typename
              = typename std::enable_if<std::is_same<T1, double>::value>::type>
    vec3<double> normalized() const;

    T dot(const vec3&) const;
    vec3 cross(const vec3&) const;
};

using vec3f = vec3<double>;
using vec3i = vec3<int>;

template <typename T>
template <typename T1, typename>
vec3f vec3<T>::normalized() const
{
    double r = sqrt(this->length2());
    return vec3f(x / r, y / r, z / r);
}

template <typename T>
T vec3<T>::dot(const vec3& rhs) const
{
    return x * rhs.x + y * rhs.y + z * rhs.z;
}

template <typename T> vec3<T> vec3<T>::cross(const vec3& rhs) const
{
    return vec3{(y * rhs.z) - (z * rhs.y), (z * rhs.x) - (x * rhs.z),
                (x * rhs.y) - (y * rhs.x)};
}

template <typename T> vec3<T> vec3<T>::operator-(const vec3<T>& rhs) const
{
    return vec3<T>(x - rhs.x, y - rhs.y, z - rhs.z);
}

template <typename T> vec3<T> vec3<T>::operator+(const vec3<T>& rhs) const
{
    return vec3<T>(x + rhs.x, y + rhs.y, z + rhs.z);
}

template <typename T> vec3<T>& vec3<T>::operator-=(const vec3<T>& rhs)
{
    vec3<T> r = (*this) - rhs;
    (*this) = r;
    return *this;
}

template <typename T>
template <typename V>
vec3<T> vec3<T>::operator*(const V& rhs) const
{
    return vec3<T>(x * rhs, y * rhs, z * rhs);
}

template <typename T>
template <typename V>
vec3<T>& vec3<T>::operator*=(const V& rhs)
{
    x *= rhs;
    y *= rhs;
    z *= rhs;
    return *this;
}

template <typename T>
template <typename V>
vec3<T> vec3<T>::operator/(const V& rhs) const
{
    return vec3<T>(x / rhs, y / rhs, z / rhs);
}

template <typename T>
template <typename V>
vec3<T>& vec3<T>::operator/=(const V& rhs)
{
    x /= rhs;
    y /= rhs;
    z /= rhs;
    return *this;
}

template <typename T>
template <typename V>
vec3<T> vec3<T>::operator*(const vec3<V>& rhs) const
{
    return vec3<T>(x * rhs.x, y * rhs.y, z * rhs.z);
}

template <typename T>
template <typename V>
vec3<T>& vec3<T>::operator*=(const vec3<V>& rhs)
{
    x *= rhs.x;
    y *= rhs.y;
    z *= rhs.z;
    return *this;
}

template <typename T>
template <typename V>
vec3<T> vec3<T>::operator/(const vec3<V>& rhs) const
{
    return vec3<T>(x / rhs.x, y / rhs.y, z / rhs.z);
}

template <typename T>
template <typename V>
vec3<T>& vec3<T>::operator/=(const vec3<V>& rhs)
{
    x /= rhs.x;
    y /= rhs.y;
    z /= rhs.z;
    return *this;
}

template <typename T> vec3<T>& vec3<T>::operator+=(const vec3<T>& rhs)
{
    (*this) = (*this) + rhs;
    return *this;
}

template <typename T> bool vec3<T>::operator==(const vec3<T>& rhs) const
{
    return (rhs.x == x) && (rhs.y == y) && (rhs.z == z);
}

template <typename T> bool vec3<T>::operator!=(const vec3<T>& rhs) const
{
    return !(*this == rhs);
}

// template <typename T> bool vec3<T>::operator<(const vec3<T>& rhs) const
// {
//     if (x > rhs.x)
//         return false;
//     else if (x < rhs.x)
//         return true;
//     else if (y < rhs.y)
//         return true;
//     else
//         return false;
// }

template <typename T> template <typename V> vec3<T>::operator vec3<V>() const
{
    return vec3<V>{static_cast<V>(x), static_cast<V>(y), static_cast<V>(z)};
}

template <typename T> T vec3<T>::length2() const {
    return x * x + y * y + z * z;
}

template <typename T> double vec3<T>::length() const
{
    return sqrt(this->length2());
}

template <typename T>
std::ostream& operator<<(std::ostream& stream, const vec3<T>& V)
{
    return stream << "<" << V.x << ", " << V.y << ", " << V.z << ">";
}
