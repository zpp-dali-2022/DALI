// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "dali/util/fits.h"
#include <fitsio.h>
#include <fitsio2.h>
#include <string>
#include <vector>
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/data/views.h"

namespace dali {
namespace fits {

std::string GetFitsErrorMessage(int status) {
  std::string status_str;
  status_str.resize(FLEN_STATUS);

  fits_get_errstatus(status, &status_str[0]); /* get the error description */

  return status_str;
}

void HandleFitsError(int status) {
  if (status) {
    DALI_FAIL(GetFitsErrorMessage(status));
  }
}

void FITS_CALL(int status) {
  return HandleFitsError(status);
}

int ImgTypeToDatatypeCode(int img_type) {
  switch (img_type) {
    case SBYTE_IMG:
      return TSBYTE;
    case BYTE_IMG:
      return TBYTE;
    case SHORT_IMG:
      return TSHORT;
    case USHORT_IMG:
      return TUSHORT;
    case LONG_IMG:
      return TINT;
    case ULONG_IMG:
      return TUINT;
    case LONGLONG_IMG:
      return TLONGLONG;
    case ULONGLONG_IMG:
      return TULONGLONG;
    case FLOAT_IMG:
      return TFLOAT;
    case DOUBLE_IMG:
      return TDOUBLE;
    default:
      DALI_FAIL("Unknown BITPIX value! Refer to the CFITSIO documentation.");
  }
}

const TypeInfo &TypeFromFitsDatatypeCode(int datatype) {
  switch (datatype) {
    case TSBYTE:
      return TypeTable::GetTypeInfo<int8_t>();
    case TBYTE:
      return TypeTable::GetTypeInfo<uint8_t>();
    case TSHORT:
      return TypeTable::GetTypeInfo<int16_t>();
    case TUSHORT:
      return TypeTable::GetTypeInfo<uint16_t>();
    case TINT:
      return TypeTable::GetTypeInfo<int32_t>();
    case TUINT:
      return TypeTable::GetTypeInfo<uint32_t>();
    case TLONGLONG:
      return TypeTable::GetTypeInfo<int64_t>();
    case TULONGLONG:
      return TypeTable::GetTypeInfo<uint64_t>();
    case TFLOAT:
      return TypeTable::GetTypeInfo<float>();
    case TDOUBLE:
      return TypeTable::GetTypeInfo<double>();
    default:
      DALI_FAIL("Unknown datatype code value! Refer to the CFITSIO documentation.");
  }
}
std::vector<int32_t> readTileSizes(fitsfile *fptr, int32_t n_dims) {
  std::vector<int32_t> tileSizes(n_dims, 1);
  int32_t status = 0;

  for (int32_t i = 0; i < n_dims; i++) {
    std::string keyword = "ZTILE" + std::to_string(i + 1);
    FITS_CALL(fits_read_key(fptr, TLONG, keyword.c_str(), &tileSizes[i], NULL, &status));
    DALI_ENFORCE(tileSizes[i] > 0, "All ZTILE{i} values must be greater than 0!");
  }

  return tileSizes;
}


void ParseHeader(HeaderData &parsed_header, fitsfile *src) {
  int32_t hdu_type, img_type, n_dims, status = 0;

  FITS_CALL(fits_get_hdu_type(src, &hdu_type, &status));
  bool is_image = (hdu_type == IMAGE_HDU);
  DALI_ENFORCE(is_image, "Only IMAGE_HDUs are supported!");

  FITS_CALL(fits_get_img_equivtype(src, &img_type, &status)); /* get IMG_TYPE code value */
  FITS_CALL(fits_get_img_dim(src, &n_dims, &status));         /* get NAXIS value */

  DALI_ENFORCE(n_dims > 0, "NAXIS (image dimensions) value for each HDU has to be greater than 0!");
  std::vector<int64_t> dims(n_dims, 0); /* create vector for storing img dims*/

  FITS_CALL(fits_get_img_size(src, n_dims, &dims[0], &status)); /* get NAXISn values */

  parsed_header.hdu_type = hdu_type;
  parsed_header.datatype_code = ImgTypeToDatatypeCode(img_type);
  parsed_header.type_info = &TypeFromFitsDatatypeCode(parsed_header.datatype_code);
  parsed_header.compressed = (fits_is_compressed_image(src, &status) == 1);

  if (parsed_header.compressed) {
    parsed_header.tile_sizes = readTileSizes(src, n_dims);
    parsed_header.tiles =
        std::accumulate(parsed_header.tile_sizes.begin(), parsed_header.tile_sizes.end(), 1,
                        std::multiplies<int32_t>());
    parsed_header.blocksize = (src->Fptr)->rice_blocksize;
    parsed_header.bytepix = (src->Fptr)->rice_bytepix;
    parsed_header.maxtilelen = (src->Fptr)->maxtilelen;
    parsed_header.zbitpix = (src->Fptr)->zbitpix;
    parsed_header.rice_blocksize = (src->Fptr)->rice_blocksize;


    DALI_ENFORCE(parsed_header.blocksize > 0, "RICE_BLOCKSIZE value must be greater than 0!");
    DALI_ENFORCE(parsed_header.bytepix > 0, "RICE_BYTEPIX value must be greater than 0!");
  }

  for (auto it = dims.rbegin(); it != dims.rend(); ++it) {
    parsed_header.shape.shape.push_back(static_cast<int64_t>(*it));
  }
}

DALIDataType HeaderData::type() const {
  return type_info ? type_info->id() : DALI_NO_TYPE;
}

size_t HeaderData::size() const {
  return volume(shape);
}

size_t HeaderData::nbytes() const {
  return type_info ? type_info->size() * size() : 0_uz;
}

unsigned char **extract_compressed_data(fitsfile *fptr, int *status) {
  long tiles;
  long *row_sizes;
  int *tile_sizes;
  LONGLONG fpixel[MAX_COMPRESS_DIM], lpixel[MAX_COMPRESS_DIM];
  long inc[MAX_COMPRESS_DIM];
  long ztile[MAX_COMPRESS_DIM] = {1, 1, 1, 1, 1, 1};

  fits_read_key(fptr, TLONG, "ZTILE1", &ztile[0], nullptr, status);
  fits_read_key(fptr, TLONG, "ZTILE2", &ztile[1], nullptr, status);
  fits_read_key(fptr, TLONG, "ZTILE3", &ztile[2], nullptr, status);
  fits_read_key(fptr, TLONG, "ZTILE4", &ztile[3], nullptr, status);
  fits_read_key(fptr, TLONG, "ZTILE5", &ztile[4], nullptr, status);
  fits_read_key(fptr, TLONG, "ZTILE6", &ztile[5], nullptr, status);
  *status = 0;

  tiles = ztile[0] * ztile[1] * ztile[2] * ztile[3] * ztile[4] * ztile[5];
  row_sizes = (long *)malloc(tiles * sizeof(long));
  tile_sizes = (int *)malloc(tiles * sizeof(int));
  auto **data = (unsigned char **)malloc(tiles * sizeof(unsigned char *));
  int size = 0;

  for (int i = 0; i < (fptr->Fptr)->zndim; ++i) {
    fpixel[i] = 1;
    lpixel[i] = (fptr->Fptr)->znaxis[i];
    inc[i] = 1;
  }

  long naxis[MAX_COMPRESS_DIM] = {1, 1, 1, 1, 1, 1};
  long tiledim[MAX_COMPRESS_DIM] = {1, 1, 1, 1, 1, 1};
  long tilesize[MAX_COMPRESS_DIM] = {1, 1, 1, 1, 1, 1};
  long ftile[MAX_COMPRESS_DIM] = {1, 1, 1, 1, 1, 1};
  long ltile[MAX_COMPRESS_DIM] = {1, 1, 1, 1, 1, 1};
  long rowdim[MAX_COMPRESS_DIM] = {1, 1, 1, 1, 1, 1};
  long tfpixel[MAX_COMPRESS_DIM], tlpixel[MAX_COMPRESS_DIM];
  long offset[MAX_COMPRESS_DIM], thistilesize[MAX_COMPRESS_DIM];
  long mfpixel[MAX_COMPRESS_DIM], mlpixel[MAX_COMPRESS_DIM];
  long minc[MAX_COMPRESS_DIM];
  long i5, i4, i3, i2, i1, i0, irow;
  long ntemp;
  int ndim;

  ndim = (fptr->Fptr)->zndim;
  ntemp = 1;
  for (int i = 0; i < ndim; ++i) {
    if (fpixel[i] <= lpixel[i]) {
      mfpixel[i] = (long)fpixel[i];
      mlpixel[i] = (long)lpixel[i];
      minc[i] = inc[i];
    } else {
      mfpixel[i] = (long)lpixel[i];
      mlpixel[i] = (long)fpixel[i];
      minc[i] = -inc[i];
    }

    naxis[i] = (fptr->Fptr)->znaxis[i];
    tilesize[i] = (fptr->Fptr)->tilesize[i];
    tiledim[i] = (naxis[i] - 1) / tilesize[i] + 1;
    ftile[i] = (mfpixel[i] - 1) / tilesize[i] + 1;
    ltile[i] = minvalue((mlpixel[i] - 1) / tilesize[i] + 1, tiledim[i]);
    rowdim[i] = ntemp;
    ntemp *= tiledim[i];
  }

  for (i5 = ftile[5]; i5 <= ltile[5]; ++i5) {
    tfpixel[5] = (i5 - 1) * tilesize[5] + 1;
    tlpixel[5] = minvalue(tfpixel[5] + tilesize[5] - 1, naxis[5]);
    thistilesize[5] = tlpixel[5] - tfpixel[5] + 1;
    offset[5] = (i5 - 1) * rowdim[5];
    for (i4 = ftile[4]; i4 <= ltile[4]; ++i4) {
      tfpixel[4] = (i4 - 1) * tilesize[4] + 1;
      tlpixel[4] = minvalue(tfpixel[4] + tilesize[4] - 1, naxis[4]);
      thistilesize[4] = thistilesize[5] * (tlpixel[4] - tfpixel[4] + 1);
      offset[4] = (i4 - 1) * rowdim[4] + offset[5];
      for (i3 = ftile[3]; i3 <= ltile[3]; ++i3) {
        tfpixel[3] = (i3 - 1) * tilesize[3] + 1;
        tlpixel[3] = minvalue(tfpixel[3] + tilesize[3] - 1, naxis[3]);
        thistilesize[3] = thistilesize[4] * (tlpixel[3] - tfpixel[3] + 1);
        offset[3] = (i3 - 1) * rowdim[3] + offset[4];
        for (i2 = ftile[2]; i2 <= ltile[2]; ++i2) {
          tfpixel[2] = (i2 - 1) * tilesize[2] + 1;
          tlpixel[2] = minvalue(tfpixel[2] + tilesize[2] - 1, naxis[2]);
          thistilesize[2] = thistilesize[3] * (tlpixel[2] - tfpixel[2] + 1);
          offset[2] = (i2 - 1) * rowdim[2] + offset[3];
          for (i1 = ftile[1]; i1 <= ltile[1]; ++i1) {
            tfpixel[1] = (i1 - 1) * tilesize[1] + 1;
            tlpixel[1] = minvalue(tfpixel[1] + tilesize[1] - 1, naxis[1]);
            thistilesize[1] = thistilesize[2] * (tlpixel[1] - tfpixel[1] + 1);
            offset[1] = (i1 - 1) * rowdim[1] + offset[2];
            for (i0 = ftile[0]; i0 <= ltile[0]; ++i0) {
              tfpixel[0] = (i0 - 1) * tilesize[0] + 1;
              tlpixel[0] = minvalue(tfpixel[0] + tilesize[0] - 1, naxis[0]);
              thistilesize[0] = thistilesize[1] * (tlpixel[0] - tfpixel[0] + 1);
              irow = i0 + offset[1];

              LONGLONG nelemll = 0, noffset = 0;
              ffgdesll(fptr, (fptr->Fptr)->cn_compressed, irow, &nelemll, &noffset, status);

              row_sizes[size] = (long)nelemll;
              tile_sizes[size] = thistilesize[0];
              data[size] = (unsigned char *)malloc((long)nelemll);
              unsigned char charnull = 0;
              fits_read_col(fptr, TBYTE, (fptr->Fptr)->cn_compressed, irow, 1, (long)nelemll,
                            &charnull, data[size], nullptr, status);
              ++size;
            }
          }
        }
      }
    }
  }
  if(row_sizes)
    free(row_sizes);

  if(tile_sizes)
    free(tile_sizes);

  return data;
}

}  // namespace fits
}  // namespace dali
