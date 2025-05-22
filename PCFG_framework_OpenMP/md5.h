#include <iostream>
#include <string>
#include <cstring>
#include <arm_neon.h>
#include<vector>

using namespace std;

// 定义了Byte，便于使用
typedef unsigned char Byte;
// 定义了32比特
typedef unsigned int bit32;

// MD5的一系列参数。参数是固定的，其实你不需要看懂这些
#define s11 7
#define s12 12
#define s13 17
#define s14 22
#define s21 5
#define s22 9
#define s23 14
#define s24 20
#define s31 4
#define s32 11
#define s33 16
#define s34 23
#define s41 6
#define s42 10
#define s43 15
#define s44 21

/**
 * @Basic MD5 functions.
 *
 * @param there bit32.
 *
 * @return one bit32.
 */
// 定义了一系列MD5中的具体函数
// 这四个计算函数是需要你进行SIMD并行化的
// 可以看到，FGHI四个函数都涉及一系列位运算，在数据上是对齐的，非常容易实现SIMD的并行化

inline uint32x4_t F_neon(uint32x4_t x, uint32x4_t y, uint32x4_t z) {
  uint32x4_t xy = vandq_u32(x, y); 
  uint32x4_t not_x = vmvnq_u32(x); 
  uint32x4_t notx_z = vandq_u32(not_x, z); 
  uint32x4_t result = vorrq_u32(xy, notx_z); 
  return result;
}
inline uint32x4_t G_neon(uint32x4_t x, uint32x4_t y, uint32x4_t z) {
  uint32x4_t xz = vandq_u32(x, z); // 按位与
  uint32x4_t not_z = vmvnq_u32(z);  // 按位取反
  uint32x4_t notz_y = vandq_u32(y, not_z);
  uint32x4_t result = vorrq_u32(xz, notz_y); // 按位或
  return result;
}
inline uint32x4_t H_neon(uint32x4_t x, uint32x4_t y, uint32x4_t z) {
  uint32x4_t xeory = veorq_u32(x, y);
  uint32x4_t result = veorq_u32(xeory, z);
  return result;
}
inline uint32x4_t I_neon(uint32x4_t x, uint32x4_t y, uint32x4_t z) {
  uint32x4_t not_z = vmvnq_u32(z);
  uint32x4_t xor_notz = vorrq_u32(x, not_z);
  uint32x4_t result = veorq_u32(y, xor_notz);
  return result;
}

/**
 * @Rotate Left.
 *
 * @param {num} the raw number.
 *
 * @param {n} rotate left n.
 *
 * @return the number after rotated left.
 */
// 定义了一系列MD5中的具体函数
// 这五个计算函数（ROTATELEFT/FF/GG/HH/II）和之前的FGHI一样，都是需要你进行SIMD并行化的
// 但是你需要注意的是#define的功能及其效果，可以发现这里的FGHI是没有返回值的，为什么呢？你可以查询#define的含义和用法

#define ROTATELEFT_neon(a, s) vorrq_u32(vshlq_n_u32(a, s), vshrq_n_u32(a, 32 - s))

#define FF_neon(a, b, c, d, x, s, ac) \
{ \
  uint32x4_t temp1 = vaddq_u32(F_neon(b, c, d), x); \
  uint32x4_t temp2 = vaddq_u32(temp1, vdupq_n_u32(ac)); \
  a = vaddq_u32(a, temp2); \
  a = ROTATELEFT_neon(a, s); \
  a = vaddq_u32(a, b); \
}

#define GG_neon(a, b, c, d, x, s, ac) \
{ \
  uint32x4_t temp1 = vaddq_u32(G_neon(b, c, d), x); \
  uint32x4_t temp2 = vaddq_u32(temp1, vdupq_n_u32(ac)); \
  a = vaddq_u32(a, temp2); \
  a = ROTATELEFT_neon(a, s); \
  a = vaddq_u32(a, b); \
}

#define HH_neon(a, b, c, d, x, s, ac) \
{ \
  uint32x4_t temp1 = vaddq_u32(H_neon(b, c, d), x); \
  uint32x4_t temp2 = vaddq_u32(temp1, vdupq_n_u32(ac)); \
  a = vaddq_u32(a, temp2); \
  a = ROTATELEFT_neon(a, s); \
  a = vaddq_u32(a, b); \
}

#define II_neon(a, b, c, d, x, s, ac) \
{ \
  uint32x4_t temp1 = vaddq_u32(I_neon(b, c, d), x); \
  uint32x4_t temp2 = vaddq_u32(temp1, vdupq_n_u32(ac)); \
  a = vaddq_u32(a, temp2); \
  a = ROTATELEFT_neon(a, s); \
  a = vaddq_u32(a, b); \
}

void MD5Hash_neon(vector<string> inputs, uint32_t states[4][4]);
