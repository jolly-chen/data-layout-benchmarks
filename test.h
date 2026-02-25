#ifndef TEST_H
#define TEST_H
// THIS FILE IS GENERATED USING generate_datastructures.py
#include "utils.h"

struct S {
    int x;
    double y;
    float z;
    char w;
};

struct SRef {
    int& x;
    double& y;
    float& z;
    char& w;
};

struct SubS01 {
    int x;
    double y;
};
struct SubS23 {
    float z;
    char w;
};
struct SubS0 {
    int x;
};
struct SubS1 {
    double y;
};
struct SubS2 {
    float z;
};
struct SubS3 {
    char w;
};

struct PartitionedContainerContiguous01_23 {
    std::span<SubS01> p0;
    std::span<SubS23> p1;

    std::byte *storage;
    size_t n;

    static std::string to_string() { return "PartitionedContainerContiguous01_23"; }

    PartitionedContainerContiguous01_23(size_t n, size_t alignment) : n(n) {
        // Allocate each partition
        size_t total_size = 0 + AlignSize(n * sizeof(SubS01), alignment) + AlignSize(n * sizeof(SubS23), alignment);
        storage = static_cast<std::byte*>(std::aligned_alloc(alignment, total_size));

        // Assign each partition to its location in the storage vector
        size_t offset = 0;
        p0 = std::span<SubS01>(std::launder(reinterpret_cast<SubS01*>(new (&storage[offset]) SubS01[n])), n);
        offset += AlignSize(p0.size_bytes(), alignment);
        p1 = std::span<SubS23>(std::launder(reinterpret_cast<SubS23*>(new (&storage[offset]) SubS23[n])), n);
        offset += AlignSize(p1.size_bytes(), alignment);

    }

    inline SRef operator[](const size_t index) const {
        return SRef{ p0[index].x, p0[index].y, p1[index].z, p1[index].w };
    }

    size_t size() const { return n; }

    ~PartitionedContainerContiguous01_23() {
        // Deallocate each partition
        for (size_t i = n - 1; i == 0; --i) {
              p0[i].~SubS01();
              p1[i].~SubS23();
        }

        std::free(storage);
    }
};

struct PartitionedContainer01_23 {
    SubS01 *p0;
    SubS23 *p1;

    size_t n;

    static std::string to_string() { return "PartitionedContainer01_23"; }

    PartitionedContainer01_23(size_t n, size_t alignment) : n(n) {
        p0 = static_cast<SubS01*>(std::aligned_alloc(alignment, AlignSize(n * sizeof(SubS01), alignment)));
        p1 = static_cast<SubS23*>(std::aligned_alloc(alignment, AlignSize(n * sizeof(SubS23), alignment)));
    }

    inline SRef operator[](const size_t index) const {
        return SRef{ p0[index].x, p0[index].y, p1[index].z, p1[index].w };
    }

    size_t size() const { return n; }

    ~PartitionedContainer01_23() {
        // Deallocate each partition
        std::free(p0);
        std::free(p1);
    }
};

struct PartitionedContainerContiguous0_1_2_3 {
    std::span<SubS0> p0;
    std::span<SubS1> p1;
    std::span<SubS2> p2;
    std::span<SubS3> p3;

    std::byte *storage;
    size_t n;

    static std::string to_string() { return "PartitionedContainerContiguous0_1_2_3"; }

    PartitionedContainerContiguous0_1_2_3(size_t n, size_t alignment) : n(n) {
        // Allocate each partition
        size_t total_size = 0 + AlignSize(n * sizeof(SubS0), alignment) + AlignSize(n * sizeof(SubS1), alignment) + AlignSize(n * sizeof(SubS2), alignment) + AlignSize(n * sizeof(SubS3), alignment);
        storage = static_cast<std::byte*>(std::aligned_alloc(alignment, total_size));

        // Assign each partition to its location in the storage vector
        size_t offset = 0;
        p0 = std::span<SubS0>(std::launder(reinterpret_cast<SubS0*>(new (&storage[offset]) SubS0[n])), n);
        offset += AlignSize(p0.size_bytes(), alignment);
        p1 = std::span<SubS1>(std::launder(reinterpret_cast<SubS1*>(new (&storage[offset]) SubS1[n])), n);
        offset += AlignSize(p1.size_bytes(), alignment);
        p2 = std::span<SubS2>(std::launder(reinterpret_cast<SubS2*>(new (&storage[offset]) SubS2[n])), n);
        offset += AlignSize(p2.size_bytes(), alignment);
        p3 = std::span<SubS3>(std::launder(reinterpret_cast<SubS3*>(new (&storage[offset]) SubS3[n])), n);
        offset += AlignSize(p3.size_bytes(), alignment);

    }

    inline SRef operator[](const size_t index) const {
        return SRef{ p0[index].x, p1[index].y, p2[index].z, p3[index].w };
    }

    size_t size() const { return n; }

    ~PartitionedContainerContiguous0_1_2_3() {
        // Deallocate each partition
        for (size_t i = n - 1; i == 0; --i) {
              p0[i].~SubS0();
              p1[i].~SubS1();
              p2[i].~SubS2();
              p3[i].~SubS3();
        }

        std::free(storage);
    }
};

struct PartitionedContainer0_1_2_3 {
    SubS0 *p0;
    SubS1 *p1;
    SubS2 *p2;
    SubS3 *p3;

    size_t n;

    static std::string to_string() { return "PartitionedContainer0_1_2_3"; }

    PartitionedContainer0_1_2_3(size_t n, size_t alignment) : n(n) {
        p0 = static_cast<SubS0*>(std::aligned_alloc(alignment, AlignSize(n * sizeof(SubS0), alignment)));
        p1 = static_cast<SubS1*>(std::aligned_alloc(alignment, AlignSize(n * sizeof(SubS1), alignment)));
        p2 = static_cast<SubS2*>(std::aligned_alloc(alignment, AlignSize(n * sizeof(SubS2), alignment)));
        p3 = static_cast<SubS3*>(std::aligned_alloc(alignment, AlignSize(n * sizeof(SubS3), alignment)));
    }

    inline SRef operator[](const size_t index) const {
        return SRef{ p0[index].x, p1[index].y, p2[index].z, p3[index].w };
    }

    size_t size() const { return n; }

    ~PartitionedContainer0_1_2_3() {
        // Deallocate each partition
        std::free(p0);
        std::free(p1);
        std::free(p2);
        std::free(p3);
    }
};

struct PartitionedContainerContiguous0_1_23 {
    std::span<SubS0> p0;
    std::span<SubS1> p1;
    std::span<SubS23> p2;

    std::byte *storage;
    size_t n;

    static std::string to_string() { return "PartitionedContainerContiguous0_1_23"; }

    PartitionedContainerContiguous0_1_23(size_t n, size_t alignment) : n(n) {
        // Allocate each partition
        size_t total_size = 0 + AlignSize(n * sizeof(SubS0), alignment) + AlignSize(n * sizeof(SubS1), alignment) + AlignSize(n * sizeof(SubS23), alignment);
        storage = static_cast<std::byte*>(std::aligned_alloc(alignment, total_size));

        // Assign each partition to its location in the storage vector
        size_t offset = 0;
        p0 = std::span<SubS0>(std::launder(reinterpret_cast<SubS0*>(new (&storage[offset]) SubS0[n])), n);
        offset += AlignSize(p0.size_bytes(), alignment);
        p1 = std::span<SubS1>(std::launder(reinterpret_cast<SubS1*>(new (&storage[offset]) SubS1[n])), n);
        offset += AlignSize(p1.size_bytes(), alignment);
        p2 = std::span<SubS23>(std::launder(reinterpret_cast<SubS23*>(new (&storage[offset]) SubS23[n])), n);
        offset += AlignSize(p2.size_bytes(), alignment);

    }

    inline SRef operator[](const size_t index) const {
        return SRef{ p0[index].x, p1[index].y, p2[index].z, p2[index].w };
    }

    size_t size() const { return n; }

    ~PartitionedContainerContiguous0_1_23() {
        // Deallocate each partition
        for (size_t i = n - 1; i == 0; --i) {
              p0[i].~SubS0();
              p1[i].~SubS1();
              p2[i].~SubS23();
        }

        std::free(storage);
    }
};

struct PartitionedContainer0_1_23 {
    SubS0 *p0;
    SubS1 *p1;
    SubS23 *p2;

    size_t n;

    static std::string to_string() { return "PartitionedContainer0_1_23"; }

    PartitionedContainer0_1_23(size_t n, size_t alignment) : n(n) {
        p0 = static_cast<SubS0*>(std::aligned_alloc(alignment, AlignSize(n * sizeof(SubS0), alignment)));
        p1 = static_cast<SubS1*>(std::aligned_alloc(alignment, AlignSize(n * sizeof(SubS1), alignment)));
        p2 = static_cast<SubS23*>(std::aligned_alloc(alignment, AlignSize(n * sizeof(SubS23), alignment)));
    }

    inline SRef operator[](const size_t index) const {
        return SRef{ p0[index].x, p1[index].y, p2[index].z, p2[index].w };
    }

    size_t size() const { return n; }

    ~PartitionedContainer0_1_23() {
        // Deallocate each partition
        std::free(p0);
        std::free(p1);
        std::free(p2);
    }
};

#endif // TEST_H
