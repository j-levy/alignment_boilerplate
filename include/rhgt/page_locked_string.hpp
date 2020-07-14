#pragma once

#include <string>

namespace rhgt {

class PageLockedString {
  public:
    PageLockedString(size_t capacity);

    ~PageLockedString();

    PageLockedString& operator+=(const std::string &o);

    char* data()      const;
    size_t size()     const;
    size_t size_left()const;
    bool empty()      const;
    bool full()       const;
    std::string str() const;
    size_t capacity() const;
    void clear();

  private:
    char *const _str = nullptr;
    const size_t _capacity = 0;
    size_t _size = 0;
};

}