#pragma once
#include <cmath>
#include <ostream>
#include <type_traits>

constexpr double pi = 3.141592653;
using u_int = unsigned int;

template <typename T>
class vec2 {
public:
    T x, y;

    vec2() : x(0), y(0) {}
    vec2(T vx, T vy) : x(vx), y(vy) {}

    vec2 operator+(const vec2 &rhs) const;
    vec2 &operator+=(const vec2 &rhs);
    vec2 operator-(const vec2 &rhs) const;
    vec2 &operator-=(const vec2 &rhs);

    template <typename V> vec2 operator*(const V& rhs) const;
    template <typename V> vec2& operator*=(const V& rhs);

    template <typename V> vec2 operator*(const vec2<V>& rhs) const;
    template <typename V> vec2& operator*=(const vec2<V>& rhs);
    template <typename V> vec2 operator/(const vec2<V>& rhs) const;
    template <typename V> vec2& operator/=(const vec2<V>& rhs);

    bool operator==(const vec2 &rhs) const;
    bool operator!=(const vec2 &rhs) const;
    bool operator<(const vec2 &rhs) const;

    template <typename V>
    explicit operator vec2<V>() const;

    T length2() const;
    double length() const;

    template <typename T1 = T,
              typename = typename std::enable_if<
                  std::is_same<T1, double>::value>::type>
    vec2<double> normalized() const;

    T dot(const vec2&) const;

    template <typename T1 = T,
              typename = typename std::enable_if<
                  std::is_same<T1, double>::value>::type>
    vec2<double> rotated(double a) const;
};

using vec2f = vec2<double>;
using vec2ui = vec2<u_int>;
using vec2i = vec2<int>;

template <typename T>
template <typename T1, typename>
vec2f vec2<T>::normalized() const
{
    double r = sqrt(this->length2());
    return vec2f(x / r, y / r);
}

template <typename T>
T vec2<T>::dot(const vec2& rhs) const
{
    return x * rhs.x + y * rhs.y ;
}

template <typename T>
template <typename T1, typename>
vec2f vec2<T>::rotated(const double a) const
{
    return {x * cos(a) - y * sin(a), x * sin(a) + y * cos(a)};
}

template <typename T>
vec2<T> vec2<T>::operator-(const vec2<T> &rhs) const
{
    return vec2<T>(x - rhs.x, y - rhs.y);
}

template <typename T>
vec2<T> vec2<T>::operator+(const vec2<T> &rhs) const
{
    return vec2<T>(x + rhs.x, y + rhs.y);
}

template <typename T>
vec2<T> &vec2<T>::operator-=(const vec2<T> &rhs)
{
    vec2<T> r = (*this) - rhs;
    (*this) = r;
    return *this;
}

template <typename T>
template <typename V>
vec2<T> vec2<T>::operator*(const V &rhs) const
{
    return vec2<T>(x * rhs, y * rhs);
}

template <typename T>
template <typename V>
vec2<T> &vec2<T>::operator*=(const V &rhs)
{
    x *= rhs;
    y *= rhs;
    return *this;
}

template <typename T>
template <typename V>
vec2<T> vec2<T>::operator*(const vec2<V>& rhs) const
{
    return vec2<T>(x * rhs.x, y * rhs.y);
}

template <typename T>
template <typename V>
vec2<T>& vec2<T>::operator*=(const vec2<V>& rhs)
{
    x *= rhs.x;
    y *= rhs.y;
    return *this;
}

template <typename T>
template <typename V>
vec2<T> vec2<T>::operator/(const vec2<V>& rhs) const
{
    return vec2<T>(x / rhs.x, y / rhs.y);
}

template <typename T>
template <typename V>
vec2<T>& vec2<T>::operator/=(const vec2<V>& rhs)
{
    x /= rhs.x;
    y /= rhs.y;
    return *this;
}

template <typename T>
vec2<T> &vec2<T>::operator+=(const vec2<T> &rhs)
{
    (*this) = (*this) + rhs;
    return *this;
}

template <typename T>
bool vec2<T>::operator==(const vec2<T> &rhs) const
{
    return (rhs.x == x) && (rhs.y == y);
}

template <typename T>
bool vec2<T>::operator!=(const vec2<T> &rhs) const
{
    return !(*this == rhs);
}

template <typename T>
bool vec2<T>::operator<(const vec2<T> &rhs) const
{
    if (x > rhs.x)
        return false;
    else if (x < rhs.x)
        return true;
    else if (y < rhs.y)
        return true;
    else
        return false;
}

template <typename T>
template <typename V>
vec2<T>::operator vec2<V>() const
{
    return vec2<V>{static_cast<V>(x), static_cast<V>(y)};
}

template <typename T>
T vec2<T>::length2() const
{
    return x * x + y * y;
}

template <typename T>
double vec2<T>::length() const
{
    return sqrt(this->length2());
}

struct colour {
    float r, g, b, a;
};

template <typename T>
std::ostream &operator<<(std::ostream &stream, const vec2<T> &V)
{
    return stream << "<" << V.x << ", " << V.y << ">";
}

template <class T, class V>
bool box_bound(T box_min, T box_max, V point)
{
    return point.x > box_min.x && point.y > box_min.y &&
           point.x < box_max.x && point.y < box_max.y;
}
