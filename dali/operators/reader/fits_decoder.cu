#include <iostream>
#include "dali/operators/reader/fits_decoder.cuh"

namespace dali {

__global__ void rice_decompress(unsigned char **compressed_data, void *uncompressed_data,
                                int bytepix, int blocksize, long tiles, long maxtilelen,
                                const int *tile_size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  const int nonzero_count[256] = {
      0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
      5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
      6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
      8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
      8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
      8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
      8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};

  if (bytepix == 1) {
    for (long tile = index; tile < tiles; tile += stride) {
      int i, imax;
      int k;
      int nbits, nzero, fs;
      unsigned int b, diff, lastpix;
      int fsmax, fsbits, bbits;
      int shift;
      long beg;

      fsbits = 3;
      fsmax = 6;

      bbits = 1 << fsbits;

      lastpix = compressed_data[tile][0];
      compressed_data[tile] += 1;
      shift = 1;

      b = *compressed_data[tile]++;
      shift++;

      beg = tile * maxtilelen;
      nbits = 8;
      for (i = 0; i < tile_size[tile];) {
        nbits -= fsbits;
        while (nbits < 0) {
          b = (b << 8) | (*compressed_data[tile]++);
          shift++;
          nbits += 8;
        }
        fs = (b >> nbits) - 1;

        b &= (1 << nbits) - 1;
        imax = i + blocksize;
        if (imax > tile_size[tile])
          imax = tile_size[tile];
        if (fs < 0) {
          for (; i < imax; i++)
            ((unsigned char *)uncompressed_data)[beg + i] = lastpix;
        } else if (fs == fsmax) {
          for (; i < imax; i++) {
            k = bbits - nbits;
            diff = b << k;
            for (k -= 8; k >= 0; k -= 8) {
              b = *compressed_data[tile]++;
              shift++;
              diff |= b << k;
            }
            if (nbits > 0) {
              b = *compressed_data[tile]++;
              shift++;
              diff |= b >> (-k);
              b &= (1 << nbits) - 1;
            } else {
              b = 0;
            }

            if ((diff & 1) == 0) {
              diff = diff >> 1;
            } else {
              diff = ~(diff >> 1);
            }
            ((unsigned char *)uncompressed_data)[beg + i] = diff + lastpix;
            lastpix = ((unsigned char *)uncompressed_data)[beg + i];
          }
        } else {
          for (; i < imax; i++) {
            while (b == 0) {
              nbits += 8;
              b = *compressed_data[tile]++;
              shift++;
            }
            nzero = nbits - nonzero_count[b];
            nbits -= nzero + 1;
            b ^= 1 << nbits;
            nbits -= fs;
            while (nbits < 0) {
              b = (b << 8) | (*compressed_data[tile]++);
              shift++;
              nbits += 8;
            }
            diff = (nzero << fs) | (b >> nbits);
            b &= (1 << nbits) - 1;

            if ((diff & 1) == 0) {
              diff = diff >> 1;
            } else {
              diff = ~(diff >> 1);
            }
            ((unsigned char *)uncompressed_data)[beg + i] = diff + lastpix;
            lastpix = ((unsigned char *)uncompressed_data)[beg + i];
          }
        }
      }
      compressed_data[tile] -= shift;
    }
  } else if (bytepix == 2) {
    for (long tile = index; tile < tiles; tile += stride) {
      int i, imax, k;
      int nbits, nzero, fs;
      unsigned char bytevalue;
      unsigned int b, diff, lastpix;
      int fsmax, fsbits, bbits;
      int shift;
      long beg;

      fsbits = 4;
      fsmax = 14;

      bbits = 1 << fsbits;

      lastpix = 0;
      bytevalue = compressed_data[tile][0];
      lastpix = lastpix | (bytevalue << 8);
      bytevalue = compressed_data[tile][1];
      lastpix = lastpix | bytevalue;

      compressed_data[tile] += 2;
      shift = 2;

      b = *compressed_data[tile]++;
      shift++;

      beg = tile * maxtilelen;
      nbits = 8;
      for (i = 0; i < tile_size[tile];) {
        nbits -= fsbits;
        while (nbits < 0) {
          b = (b << 8) | (*compressed_data[tile]++);
          shift++;
          nbits += 8;
        }
        fs = (b >> nbits) - 1;

        b &= (1 << nbits) - 1;
        imax = i + blocksize;
        if (imax > tile_size[tile])
          imax = tile_size[tile];
        if (fs < 0) {
          for (; i < imax; i++)
            ((unsigned short *)uncompressed_data)[beg + i] = lastpix;
        } else if (fs == fsmax) {
          for (; i < imax; i++) {
            k = bbits - nbits;
            diff = b << k;
            for (k -= 8; k >= 0; k -= 8) {
              b = *compressed_data[tile]++;
              shift++;
              diff |= b << k;
            }
            if (nbits > 0) {
              b = *compressed_data[tile]++;
              shift++;
              diff |= b >> (-k);
              b &= (1 << nbits) - 1;
            } else {
              b = 0;
            }

            if ((diff & 1) == 0) {
              diff = diff >> 1;
            } else {
              diff = ~(diff >> 1);
            }
            ((unsigned short *)uncompressed_data)[beg + i] = diff + lastpix;
            lastpix = ((unsigned short *)uncompressed_data)[beg + i];
          }
        } else {
          for (; i < imax; i++) {
            while (b == 0) {
              nbits += 8;
              b = *compressed_data[tile]++;
              shift++;
            }
            nzero = nbits - nonzero_count[b];
            nbits -= nzero + 1;
            b ^= 1 << nbits;
            nbits -= fs;
            while (nbits < 0) {
              b = (b << 8) | (*compressed_data[tile]++);
              shift++;
              nbits += 8;
            }
            diff = (nzero << fs) | (b >> nbits);
            b &= (1 << nbits) - 1;

            if ((diff & 1) == 0) {
              diff = diff >> 1;
            } else {
              diff = ~(diff >> 1);
            }
            ((unsigned short *)uncompressed_data)[beg + i] = diff + lastpix;
            lastpix = ((unsigned short *)uncompressed_data)[beg + i];
          }
        }
      }
      compressed_data[tile] -= shift;
    }
  } else {
    for (long tile = index; tile < tiles; tile += stride) {
      int i, imax, k;
      int nbits, nzero, fs;
      unsigned char bytevalue;
      unsigned int b, diff, lastpix;
      int fsmax, fsbits, bbits;
      int shift;
      long beg;

      fsbits = 5;
      fsmax = 25;

      bbits = 1 << fsbits;

      lastpix = 0;
      bytevalue = compressed_data[tile][0];
      lastpix = lastpix | (bytevalue << 24);
      bytevalue = compressed_data[tile][1];
      lastpix = lastpix | (bytevalue << 16);
      bytevalue = compressed_data[tile][2];
      lastpix = lastpix | (bytevalue << 8);
      bytevalue = compressed_data[tile][3];
      lastpix = lastpix | bytevalue;

      compressed_data[tile] += 4;
      shift = 4;

      b = *compressed_data[tile]++;
      shift++;

      beg = tile * maxtilelen;
      nbits = 8;
      for (i = 0; i < tile_size[tile];) {
        nbits -= fsbits;
        while (nbits < 0) {
          b = (b << 8) | (*compressed_data[tile]++);
          shift++;
          nbits += 8;
        }
        fs = (b >> nbits) - 1;

        b &= (1 << nbits) - 1;
        imax = i + blocksize;
        if (imax > tile_size[tile])
          imax = tile_size[tile];
        if (fs < 0) {
          for (; i < imax; i++)
            ((unsigned int *)uncompressed_data)[beg + i] = lastpix;
        } else if (fs == fsmax) {
          for (; i < imax; i++) {
            k = bbits - nbits;
            diff = b << k;
            for (k -= 8; k >= 0; k -= 8) {
              b = *compressed_data[tile]++;
              shift++;
              diff |= b << k;
            }
            if (nbits > 0) {
              b = *compressed_data[tile]++;
              shift++;
              diff |= b >> (-k);
              b &= (1 << nbits) - 1;
            } else {
              b = 0;
            }

            if ((diff & 1) == 0) {
              diff = diff >> 1;
            } else {
              diff = ~(diff >> 1);
            }
            ((unsigned int *)uncompressed_data)[beg + i] = diff + lastpix;
            lastpix = ((unsigned int *)uncompressed_data)[beg + i];
          }
        } else {
          for (; i < imax; i++) {
            while (b == 0) {
              nbits += 8;
              b = *compressed_data[tile]++;
              shift++;
            }
            nzero = nbits - nonzero_count[b];
            nbits -= nzero + 1;
            b ^= 1 << nbits;
            nbits -= fs;
            while (nbits < 0) {
              b = (b << 8) | (*compressed_data[tile]++);
              shift++;
              nbits += 8;
            }
            diff = (nzero << fs) | (b >> nbits);
            b &= (1 << nbits) - 1;

            if ((diff & 1) == 0) {
              diff = diff >> 1;
            } else {
              diff = ~(diff >> 1);
            }
            ((unsigned int *)uncompressed_data)[beg + i] = diff + lastpix;
            lastpix = ((unsigned int *)uncompressed_data)[beg + i];
          }
        }
      }
      compressed_data[tile] -= shift;
    }
  }
}

}  // namespace dali
