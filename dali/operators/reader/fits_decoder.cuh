namespace dali {

__global__ void rice_decompress(unsigned char **compressed_data, void *uncompressed_data,
                                int bytepix, int blocksize, long tiles, long maxtilelen,
                                const int *tile_size);

}  // namespace dali
