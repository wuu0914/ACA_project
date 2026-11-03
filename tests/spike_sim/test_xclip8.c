#include <stdint.h>

#include "xclip8.h"

int main(void) {
  const int32_t inputs[] = {
      INT32_MIN, -129, -128, -1, 0, 127, 128, INT32_MAX,
  };
  const int64_t expected[] = {
      -128, -128, -128, -1, 0, 127, 127, 127,
  };

  for (unsigned int i = 0; i < sizeof(inputs) / sizeof(inputs[0]); ++i) {
    if (xclip8(inputs[i]) != expected[i]) return 1;
  }

  return 0;
}
