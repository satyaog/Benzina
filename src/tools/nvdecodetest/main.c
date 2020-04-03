/* Includes */
#define _GNU_SOURCE
#define __HAVE_FLOAT128 0
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <linux/limits.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <libavcodec/avcodec.h>
#include <libavutil/pixdesc.h>

#include "cuviddec.h"
#include "nvcuvid.h"

#include "benzina/iso/bmff-intops.h"

#include "benzina/itu/h26x.h"
#include "benzina/itu/h265.h"


/* Data Structure Definitions */
struct UNIVERSE;
typedef struct UNIVERSE UNIVERSE;
struct UNIVERSE{
    /* Arguments */
    struct{
        const char* path;
        int         device;
    } args;
    
    /* Argument Parsing */
    int                   fileH265Fd;
    struct stat           fileH265Stat;
    const uint8_t*        fileH265Data;
    
    /* FFmpeg */
    enum AVCodecID        codecID;
    AVCodec*              codec;
    AVCodecContext*       codecCtx;
    AVPacket*             packet;
    AVFrame*              frame;
    
    
    /* NVDECODE */
    cudaStream_t          stream;
    CUvideodecoder        decoder;
    CUVIDDECODECREATEINFO decoderInfo;
    CUvideoparser         parser;
    CUVIDPARSERPARAMS     parserParams;
    
    /* Processing */
    long                  numDecodedImages;
    long                  numMappedImages;
    
    /* Decoded Frame */
    uint8_t*              nvdecFramePtr;
};



/* Static Function Prototypes */



/* Static Function Definitions */

/**
 * @brief Sequence Callback
 */

static int   nvdtSequenceCb(UNIVERSE* u, CUVIDEOFORMAT *format){
    CUresult result;
    
    
    /* Get our cues from the CUVIDEOFORMAT struct */
    memset(&u->decoderInfo, 0, sizeof(u->decoderInfo));
    u->decoderInfo.ulWidth             = format->coded_width;
    u->decoderInfo.ulHeight            = format->coded_height;
    u->decoderInfo.ulNumDecodeSurfaces = u->parserParams.ulMaxNumDecodeSurfaces;
    u->decoderInfo.CodecType           = format->codec;
    u->decoderInfo.ChromaFormat        = format->chroma_format;
    u->decoderInfo.ulCreationFlags     = cudaVideoCreate_PreferCUVID;
    u->decoderInfo.bitDepthMinus8      = format->bit_depth_luma_minus8;
    u->decoderInfo.ulIntraDecodeOnly   = 1;
    u->decoderInfo.ulMaxWidth          = u->decoderInfo.ulWidth;
    u->decoderInfo.ulMaxHeight         = u->decoderInfo.ulHeight;
    u->decoderInfo.display_area.left   = 0;
    u->decoderInfo.display_area.top    = 0;
    u->decoderInfo.display_area.right  = u->decoderInfo.ulWidth;
    u->decoderInfo.display_area.bottom = u->decoderInfo.ulHeight;
    u->decoderInfo.OutputFormat        = format->bit_depth_luma_minus8 > 0 ?
                                         cudaVideoSurfaceFormat_P016 :
                                         cudaVideoSurfaceFormat_NV12;
    u->decoderInfo.DeinterlaceMode     = cudaVideoDeinterlaceMode_Weave;
    u->decoderInfo.ulTargetWidth       = u->decoderInfo.ulWidth;
    u->decoderInfo.ulTargetHeight      = u->decoderInfo.ulHeight;
    u->decoderInfo.ulNumOutputSurfaces = 4;
    u->decoderInfo.vidLock             = NULL;
    u->decoderInfo.target_rect.left    = 0;
    u->decoderInfo.target_rect.top     = 0;
    u->decoderInfo.target_rect.right   = u->decoderInfo.ulTargetWidth;
    u->decoderInfo.target_rect.bottom  = u->decoderInfo.ulTargetHeight;
    
    printf("nvdtSequenceCb -- \n"
           "  ulWidth = %lu\n"
           "  ulHeight = %lu\n"
           "  ulNumDecodeSurfaces = %lu\n"
           "  CodecType = %u\n"
           "  ChromaFormat = %u\n"
           "  ulCreationFlags = %lu\n"
           "  bitDepthMinus8 = %lu\n"
           "  ulIntraDecodeOnly = %lu\n"
           "  ulMaxWidth = %lu\n"
           "  ulMaxHeight = %lu\n"
           "  display_area.left = %u\n"
           "  display_area.top = %u\n"
           "  display_area.right = %u\n"
           "  display_area.bottom = %u\n"
           "  OutputFormat = %u\n"
           "  DeinterlaceMode = %u\n"
           "  ulTargetWidth = %lu\n"
           "  ulTargetHeight = %lu\n"
           "  ulNumOutputSurfaces = %lu\n"
//           "  vidLock = %u\n"
           "  target_rect.left = %u\n"
           "  target_rect.top = %u\n"
           "  target_rect.right = %u\n"
           "  target_rect.bottom = %u\n",
           u->decoderInfo.ulWidth,
           u->decoderInfo.ulHeight,
           u->decoderInfo.ulNumDecodeSurfaces,
           u->decoderInfo.CodecType,
           u->decoderInfo.ChromaFormat,
           u->decoderInfo.ulCreationFlags,
           u->decoderInfo.bitDepthMinus8,
           u->decoderInfo.ulIntraDecodeOnly,
           u->decoderInfo.ulMaxWidth,
           u->decoderInfo.ulMaxHeight,
           u->decoderInfo.display_area.left,
           u->decoderInfo.display_area.top,
           u->decoderInfo.display_area.right,
           u->decoderInfo.display_area.bottom,
           u->decoderInfo.OutputFormat,
           u->decoderInfo.DeinterlaceMode,
           u->decoderInfo.ulTargetWidth,
           u->decoderInfo.ulTargetHeight,
           u->decoderInfo.ulNumOutputSurfaces,
//           u->decoderInfo.vidLock,
           u->decoderInfo.target_rect.left,
           u->decoderInfo.target_rect.top,
           u->decoderInfo.target_rect.right,
           u->decoderInfo.target_rect.bottom);
    
    
    /* Print Image Size */
    if(!u->decoder){
        fprintf(stdout, "Dataset Coded Image Size:        %4lux%4lu\n",
                u->decoderInfo.ulWidth, u->decoderInfo.ulHeight);
        fflush (stdout);
    }
    
    
    /* Initialize decoder only once */
    if(!u->decoder){
        result = cuvidCreateDecoder(&u->decoder, &u->decoderInfo);
        if(result != CUDA_SUCCESS){
            fprintf(stdout, "Failed to create NVDEC decoder (%d)!\n", (int)result);
            fflush (stdout);
            return 0;
        }
    }else{
        //cuvidDestroyDecoder(u->decoder);
        //return 1;
    }
    
    
    /* Exit */
    return 1;
}

/**
 * @brief Decode Callback
 */

static int   nvdtDecodeCb  (UNIVERSE* u, CUVIDPICPARAMS* picParams){
    /**
     * Notes:
     * 
     * In our particular situation, each image is a single-slice IDR frame.
     * Therefore, the following is true:
     * 
     *   - picParams->pBitstreamData       points to the 00 00 01 start code of the
     *                                     VCL NAL unit (Here it will always be
     *                                     type 5 - IDR). The first three bytes are
     *                                     therefore always 00 00 01.
     *   - picParams->nBitstreamDataLen    is always equal to the length of
     *                                     the above-mentioned NAL unit, from the
     *                                     beginning of its start code to the
     *                                     beginning of the next start code.
     *   - picParams->nNumSlices           is always equal to 1.
     *   - picParams->pSliceDataOffsets[0] is always equal to 0.
     * 
     * Additionally, the CurrPicIdx is dynamically determined. Its value is an
     * incrementing counter modulo u->decoderInfo.ulNumDecodeSurfaces.
     * 
     * All bytes of the structure beyond the first 968 (which corresponds to
     * offsetof(CUVIDPICPARAMS, CodecSpecific.h264.fmo)) should be exactly 0.
     */
    
    CUVIDPICPARAMS* pP = picParams;
    CUVIDHEVCPICPARAMS* hevcPP = (CUVIDHEVCPICPARAMS*)&pP->CodecSpecific;

    printf("SPS -- nvdec \n"
           "  pic_width_in_luma_samples = %u\n"
           "  pic_height_in_luma_samples = %u\n"
           "  log2_min_luma_coding_block_size_minus3 = %u\n"
           "  log2_diff_max_min_luma_coding_block_size = %u\n"
           "  log2_min_transform_block_size_minus2 = %u\n"
           "  log2_diff_max_min_transform_block_size = %u\n"
           "  pcm_enabled_flag = %u\n"
           "  log2_min_pcm_luma_coding_block_size_minus3 = %u\n"
           "  log2_diff_max_min_pcm_luma_coding_block_size = %u\n"
           "  pcm_sample_bit_depth_luma_minus1 = %u\n"
           "  pcm_sample_bit_depth_chroma_minus1 = %u\n"
           "  pcm_loop_filter_disabled_flag = %u\n"
           "  strong_intra_smoothing_enabled_flag = %u\n"
           "  max_transform_hierarchy_depth_intra = %u\n"
           "  max_transform_hierarchy_depth_inter = %u\n"
           "  amp_enabled_flag = %u\n"
           "  separate_colour_plane_flag = %u\n"
           "  log2_max_pic_order_cnt_lsb_minus4 = %u\n"
           "  num_short_term_ref_pic_sets = %u\n"
           "  long_term_ref_pics_present_flag = %u\n"
           "  num_long_term_ref_pics_sps = %u\n"
           "  sps_temporal_mvp_enabled_flag = %u\n"
           "  sample_adaptive_offset_enabled_flag = %u\n"
           "  scaling_list_enable_flag = %u\n"
           "  IrapPicFlag = %u\n"
           "  IdrPicFlag = %u\n"
           "  bit_depth_luma_minus8 = %u\n"
           "  bit_depth_chroma_minus8 = %u\n"
           "  scalingList4x4[*] = %llu\n"
           "  scalingList8x8[*] = %llu\n"
           "  scalingList16x16[*] = %llu\n"
           "  scalingList32x32[*] = %llu\n"
           "  scalingListDC16x16[*] = %u\n"
           "  scalingListDC32x32[*] = %u\n"
           "  scalingList4x4[*] = %#10x\n"
           "  scalingList8x8[*] = %#10x\n"
           "  scalingList16x16[*] = %#10x\n"
           "  scalingList32x32[*] = %#10x\n"
           "  scalingListDC16x16[*] = %#6x\n"
           "  scalingListDC32x32[*] = %#4x\n",
           hevcPP->pic_width_in_luma_samples,
           hevcPP->pic_height_in_luma_samples,
           hevcPP->log2_min_luma_coding_block_size_minus3,
           hevcPP->log2_diff_max_min_luma_coding_block_size,
           hevcPP->log2_min_transform_block_size_minus2,
           hevcPP->log2_diff_max_min_transform_block_size,
           hevcPP->pcm_enabled_flag,
           hevcPP->log2_min_pcm_luma_coding_block_size_minus3,
           hevcPP->log2_diff_max_min_pcm_luma_coding_block_size,
           hevcPP->pcm_sample_bit_depth_luma_minus1,
           hevcPP->pcm_sample_bit_depth_chroma_minus1,
           hevcPP->pcm_loop_filter_disabled_flag,
           hevcPP->strong_intra_smoothing_enabled_flag,
           hevcPP->max_transform_hierarchy_depth_intra,
           hevcPP->max_transform_hierarchy_depth_inter,
           hevcPP->amp_enabled_flag,
           hevcPP->separate_colour_plane_flag,
           hevcPP->log2_max_pic_order_cnt_lsb_minus4,
           hevcPP->num_short_term_ref_pic_sets,
           hevcPP->long_term_ref_pics_present_flag,
           hevcPP->num_long_term_ref_pics_sps,
           hevcPP->sps_temporal_mvp_enabled_flag,
           hevcPP->sample_adaptive_offset_enabled_flag,
           hevcPP->scaling_list_enable_flag,
           hevcPP->IrapPicFlag,
           hevcPP->IdrPicFlag,
           hevcPP->bit_depth_luma_minus8,
           hevcPP->bit_depth_chroma_minus8,
           benz_iso_bmff_as_u64((const uint8_t*)hevcPP->ScalingList4x4),
           benz_iso_bmff_as_u64((const uint8_t*)hevcPP->ScalingList8x8),
           benz_iso_bmff_as_u64((const uint8_t*)hevcPP->ScalingList16x16),
           benz_iso_bmff_as_u64((const uint8_t*)hevcPP->ScalingList32x32),
           benz_iso_bmff_as_u32((const uint8_t*)hevcPP->ScalingListDCCoeff16x16),
           benz_iso_bmff_as_u16((const uint8_t*)hevcPP->ScalingListDCCoeff32x32),
           benz_iso_bmff_as_u64((const uint8_t*)hevcPP->ScalingList4x4),
           benz_iso_bmff_as_u64((const uint8_t*)hevcPP->ScalingList8x8),
           benz_iso_bmff_as_u64((const uint8_t*)hevcPP->ScalingList16x16),
           benz_iso_bmff_as_u64((const uint8_t*)hevcPP->ScalingList32x32),
           benz_iso_bmff_as_u32((const uint8_t*)hevcPP->ScalingListDCCoeff16x16),
           benz_iso_bmff_as_u16((const uint8_t*)hevcPP->ScalingListDCCoeff32x32));

    printf("  ScalingList4x4 -- \n");
    for (int i = 0; i < 6; ++i)
    {
        printf("    ");
        for (int j = 0; j < 16; ++j)
        {
            printf("%#3u ", hevcPP->ScalingList4x4[i][j]);
        }
        printf("\n");
    }

    printf("  ScalingList8x8 -- \n");
    for (int i = 0; i < 6; ++i)
    {
        printf("    ");
        for (int j = 0; j < 64; ++j)
        {
            printf("%#3u ", hevcPP->ScalingList8x8[i][j]);
        }
        printf("\n");
    }

    printf("  ScalingList16x16 -- \n");
    for (int i = 0; i < 6; ++i)
    {
        printf("    ");
        for (int j = 0; j < 64; ++j)
        {
            printf("%#3u ", hevcPP->ScalingList16x16[i][j]);
        }
        printf("\n");
    }

    printf("  ScalingList32x32 -- \n");
    for (int i = 0; i < 2; ++i)
    {
        printf("    ");
        for (int j = 0; j < 64; ++j)
        {
            printf("%#3u ", hevcPP->ScalingList32x32[i][j]);
        }
        printf("\n");
    }

    printf("  ScalingListDCCoeff16x16 -- \n");
    printf("    ");
    for (int i = 0; i < 6; ++i)
    {
        printf("%#3u ", hevcPP->ScalingListDCCoeff16x16[i]);
    }
    printf("\n");

    printf("  ScalingListDCCoeff32x32 -- \n");
    printf("    ");
    for (int i = 0; i < 2; ++i)
    {
        printf("%#3u ", hevcPP->ScalingListDCCoeff32x32[i]);
    }
    printf("\n");

    printf("PPS -- nvdec \n"
           "  dependent_slice_segments_enabled_flag = %u\n"
           "  slice_segment_header_extension_present_flag = %u\n"
           "  sign_data_hiding_enabled_flag = %u\n"
           "  cu_qp_delta_enabled_flag = %u\n"
           "  diff_cu_qp_delta_depth = %u\n"
           "  init_qp_minus26 = %u\n"
           "  pps_cb_qp_offset = %u\n"
           "  pps_cr_qp_offset = %u\n"
           "  constrained_intra_pred_flag = %u\n"
           "  weighted_pred_flag = %u\n"
           "  weighted_bipred_flag = %u\n"
           "  transform_skip_enabled_flag = %u\n"
           "  transquant_bypass_enabled_flag = %u\n"
           "  entropy_coding_sync_enabled_flag = %u\n"
           "  log2_parallel_merge_level_minus2 = %u\n"
           "  num_extra_slice_header_bits = %u\n"
           "  loop_filter_across_tiles_enabled_flag = %u\n"
           "  loop_filter_across_slices_enabled_flag = %u\n"
           "  output_flag_present_flag = %u\n"
           "  num_ref_idx_l0_default_active_minus1 = %u\n"
           "  num_ref_idx_l1_default_active_minus1 = %u\n"
           "  lists_modification_present_flag = %u\n"
           "  cabac_init_present_flag = %u\n"
           "  pps_slice_chroma_qp_offsets_present_flag = %u\n"
           "  deblocking_filter_override_enabled_flag = %u\n"
           "  pps_deblocking_filter_disabled_flag = %u\n"
           "  pps_beta_offset_div2 = %u\n"
           "  pps_tc_offset_div2 = %u\n"
           "  tiles_enabled_flag = %u\n"
           "  uniform_spacing_flag = %u\n"
           "  num_tile_columns_minus1 = %u\n"
           "  num_tile_rows_minus1 = %u\n"
           "  column_width_minus1[*] = %llu\n"
           "  row_height_minus1[*] = %llu\n"
           "  scalingList4x4[*] = %llu\n"
           "  scalingList8x8[*] = %llu\n"
           "  scalingList16x16[*] = %llu\n"
           "  scalingList32x32[*] = %llu\n"
           "  scalingListDC16x16[*] = %u\n"
           "  scalingListDC32x32[*] = %u\n",
           hevcPP->dependent_slice_segments_enabled_flag,
           hevcPP->slice_segment_header_extension_present_flag,
           hevcPP->sign_data_hiding_enabled_flag,
           hevcPP->cu_qp_delta_enabled_flag,
           hevcPP->diff_cu_qp_delta_depth,
           hevcPP->init_qp_minus26,
           hevcPP->pps_cb_qp_offset,
           hevcPP->pps_cr_qp_offset,
           hevcPP->constrained_intra_pred_flag,
           hevcPP->weighted_pred_flag,
           hevcPP->weighted_bipred_flag,
           hevcPP->transform_skip_enabled_flag,
           hevcPP->transquant_bypass_enabled_flag,
           hevcPP->entropy_coding_sync_enabled_flag,
           hevcPP->log2_parallel_merge_level_minus2,
           hevcPP->num_extra_slice_header_bits,
           hevcPP->loop_filter_across_tiles_enabled_flag,
           hevcPP->loop_filter_across_slices_enabled_flag,
           hevcPP->output_flag_present_flag,
           hevcPP->num_ref_idx_l0_default_active_minus1,
           hevcPP->num_ref_idx_l1_default_active_minus1,
           hevcPP->lists_modification_present_flag,
           hevcPP->cabac_init_present_flag,
           hevcPP->pps_slice_chroma_qp_offsets_present_flag,
           hevcPP->deblocking_filter_override_enabled_flag,
           hevcPP->pps_deblocking_filter_disabled_flag,
           hevcPP->pps_beta_offset_div2,
           hevcPP->pps_tc_offset_div2,
           hevcPP->tiles_enabled_flag,
           hevcPP->uniform_spacing_flag,
           hevcPP->num_tile_columns_minus1,
           hevcPP->num_tile_rows_minus1,
           benz_iso_bmff_as_u64((const uint8_t*)hevcPP->column_width_minus1),
           benz_iso_bmff_as_u64((const uint8_t*)hevcPP->row_height_minus1),
           benz_iso_bmff_as_u64((const uint8_t*)hevcPP->ScalingList4x4),
           benz_iso_bmff_as_u64((const uint8_t*)hevcPP->ScalingList8x8),
           benz_iso_bmff_as_u64((const uint8_t*)hevcPP->ScalingList16x16),
           benz_iso_bmff_as_u64((const uint8_t*)hevcPP->ScalingList32x32),
           benz_iso_bmff_as_u32((const uint8_t*)hevcPP->ScalingListDCCoeff16x16),
           benz_iso_bmff_as_u32((const uint8_t*)hevcPP->ScalingListDCCoeff32x32));

    printf("nvdecodeFeederThrdCore -- \n"
           "picParams -- \n"
           "  PicWidthInMbs = %u\n"
           "  FrameHeightInMbs = %u\n"
           "  CurrPicIdx = %u\n"
           "  field_pic_flag = %u\n"
           "  bottom_field_flag = %u\n"
           "  second_field = %u\n"
           "  nBitstreamDataLen = %u\n"
           "  pBitstreamData = %lu\n"
           "  pBitstreamData = %lli\n"
           "  nNumSlices = %u\n"
           "  pSliceDataOffsets = %u\n"
           "  ref_pic_flag = %u\n"
           "  intra_pic_flag = %u\n",
           pP->PicWidthInMbs,
           pP->FrameHeightInMbs,
           pP->CurrPicIdx,
           pP->field_pic_flag,
           pP->bottom_field_flag,
           pP->second_field,
           pP->nBitstreamDataLen,
           pP->pBitstreamData,
           ((const int64_t*)pP->pBitstreamData)[0],
           pP->nNumSlices,
           pP->pSliceDataOffsets[0],
           pP->ref_pic_flag,
           pP->intra_pic_flag);

    if(cuvidDecodePicture(u->decoder, picParams) != CUDA_SUCCESS){
        fprintf(stdout, "Could not decode picture!\n");
        return -1;
    }
    u->numDecodedImages++;
    
    return 1;
}

/**
 * @brief Display Callback
 */

static int   nvdtDisplayCb (UNIVERSE* u, CUVIDPARSERDISPINFO* dispInfo){
    CUresult              result   = CUDA_SUCCESS;
    cudaError_t           err      = cudaSuccess;
    CUVIDPROCPARAMS       VPP      = {0};
    unsigned long long    devPtr   = 0;
    unsigned              devPitch = 0;
    
    
    /* Map frame */
    VPP.progressive_frame = dispInfo->progressive_frame;
    VPP.second_field      = dispInfo->repeat_first_field + 1;
    VPP.top_field_first   = dispInfo->top_field_first;
    VPP.unpaired_field    = dispInfo->repeat_first_field < 0;
    VPP.output_stream     = u->stream;
    result = cuvidMapVideoFrame(u->decoder, dispInfo->picture_index,
                                &devPtr, &devPitch, &VPP);
    if(result != CUDA_SUCCESS){
        return 0;
    }
    
    /* Increment count of mapped frames */
    if(u->numMappedImages++ == 0){
        u->nvdecFramePtr = calloc(u->decoderInfo.ulTargetHeight*3/2,
                                  u->decoderInfo.ulTargetWidth);
        if(!u->nvdecFramePtr){
            fprintf(stdout, "Failed to allocate memory for frame!\n");
            exit(1);
        }
        err = cudaMemcpy2DAsync(u->nvdecFramePtr,
                                u->decoderInfo.ulTargetWidth,
                                (const void*)devPtr, devPitch,
                                u->decoderInfo.ulTargetWidth,
                                u->decoderInfo.ulTargetHeight*3/2,
                                cudaMemcpyDeviceToHost,
                                u->stream);
        if(err != cudaSuccess){
            fprintf(stdout, "Could not read out frame from memory (%d)!\n", (int)err);
            exit(1);
        }
    }
    
    /* Unmap frame */
    result = cuvidUnmapVideoFrame(u->decoder, devPtr);
    if(result != CUDA_SUCCESS){
        return 0;
    }
    
    /* Exit */
    return 1;
}

/**
 * @brief Init FFmpeg
 * 
 * Implies creating an H.264 decoder.
 */

static int   nvdtInitFFmpeg(UNIVERSE* u){
    int ret;
    
    avcodec_register_all();
    u->codecID  = AV_CODEC_ID_HEVC;
    u->codec    = avcodec_find_decoder  (u->codecID);
    u->codecCtx = avcodec_alloc_context3(u->codec);
    if(!u->codecCtx){
        fprintf(stdout, "Could not allocate FFmpeg decoder context!\n");
        return -1;
    }
    ret         = avcodec_open2         (u->codecCtx,
                                         u->codec,
                                         NULL);

    printf("nvdtInitFFmpeg -- \n"
           "codecID = %u\n"
           "codec = %u\n"
           "avcodec_open2 ret = %i\n",
           u->codecID,
           u->codec,
           ret);

    if(ret < 0){
        fprintf(stdout, "Could not open FFmpeg decoder context!\n");
        return -1;
    }
    u->packet   = av_packet_alloc();
    if(!u->packet){
        fprintf(stdout, "Failed to allocate packet object!\n");
        return -1;
    }
    u->frame    = av_frame_alloc();
    if(!u->frame){
        fprintf(stdout, "Error allocating frame!\n");
        return -1;
    }
    
    return 0;
}

/**
 * @brief Init CUDA & NVCUVID
 * 
 * Reference:
 * 
 * https://devtalk.nvidia.com/default/topic/417734/problem-using-nvcuvid-library-for-video-decoding/
 */

static int   nvdtInitCUDA(UNIVERSE* u){
    CUresult  result;
    
#if 1
    /* CUDA Runtime API */
    if(cudaSetDevice(u->args.device) != cudaSuccess){
        fprintf(stdout, "Could not set GPU device %d!\n", u->args.device);
        return -1;
    }
    cudaDeviceSynchronize();
#else
    CUdevice  cuDev;
    CUcontext cuCtx;
    
    /* CUDA Driver API */
    if(cuInit(0)                 != CUDA_SUCCESS){
        printf("Could not initialize CUDA runtime!\n");
        return -1;
    }
    
    if(cuDeviceGet(&cuDev, 0)    != CUDA_SUCCESS){
        printf("Could not retrieve handle for GPU device 0!\n");
        return -1;
    }
    
    if(cuDevicePrimaryCtxRetain(&cuCtx, cuDev) != CUDA_SUCCESS){
        printf("Failed to retain primary context!\n");
        return -1;
    }
    
    if(cuCtxSetCurrent(cuCtx)    != CUDA_SUCCESS){
        printf("Failed to bind context!\n");
        return -1;
    }
#endif
    
    if(cudaStreamCreateWithFlags(&u->stream, cudaStreamNonBlocking) != cudaSuccess){
        printf("Failed to create CUDA stream!\n");
        return -1;
    }
    
    memset(&u->parserParams, 0, sizeof(u->parserParams));
    u->parserParams.CodecType              = cudaVideoCodec_HEVC;
    u->parserParams.ulMaxNumDecodeSurfaces = 20;
    u->parserParams.ulClockRate            = 0;
    u->parserParams.ulErrorThreshold       = 0;
    u->parserParams.ulMaxDisplayDelay      = 4;
    u->parserParams.pUserData              = u;
    u->parserParams.pfnSequenceCallback    = (PFNVIDSEQUENCECALLBACK)nvdtSequenceCb;
    u->parserParams.pfnDecodePicture       = (PFNVIDDECODECALLBACK)  nvdtDecodeCb;
    u->parserParams.pfnDisplayPicture      = (PFNVIDDISPLAYCALLBACK) nvdtDisplayCb;
    result = cuvidCreateVideoParser(&u->parser, &u->parserParams);
    if(result != CUDA_SUCCESS){
        printf("Failed to create CUVID video parser (%d)!\n", (int)result);
        return -1;
    }
    
    return 0;
}

/**
 * @brief Init memory-map of dataset.
 */

static int   nvdtInitMmap(UNIVERSE* u){
    if      ((u->fileH265Fd = open(u->args.path, O_RDONLY|O_CLOEXEC)) < 0){
        printf("Cannot open() file %s ...\n", u->args.path);
        exit(-1);
    }else if(fstat(u->fileH265Fd, &u->fileH265Stat) < 0){
        printf("Cannot stat() file %s ...\n", u->args.path);
        exit(-1);
    }
    
    u->fileH265Data = (const uint8_t *)mmap(NULL,
                                            u->fileH265Stat.st_size,
                                            PROT_READ,
                                            MAP_SHARED,
                                            u->fileH265Fd,
                                            0);
    if(u->fileH265Data == MAP_FAILED){
        printf("Cannot mmap dataset file %s!\n", u->args.path);
        goto exit_mmap;
    }
    
    if(madvise((void*)u->fileH265Data, u->fileH265Stat.st_size, MADV_DONTDUMP) < 0){
        printf("Cannot madvise memory range of dataset!\n");
        goto exit_madvise;
    }
    
    printf("Processing file %s ...\n", u->args.path);
    return 0;
    
    
exit_madvise:
exit_mmap:
    return -1;
}

void print_nalus(const uint8_t* record, int record_size)
{
    printf("print_nalus -- \n");

    const int annex_begin_flag = 0x000001;
    int flag = 0;
    int pos = 0;
    int begin = 0;

    for (pos = 0; pos < record_size; ++pos)
    {
        flag = 0;
        flag |= record[pos];
        flag <<= 8;
        flag |= record[pos+1];
        flag <<= 8;
        flag |= record[pos+2];
        if (flag == annex_begin_flag)
        {
//            printf("%#4u: %#04x\n"
//                   "%#4u: %#04x\n"
//                   "%#4u: %#04x\n"
//                   "%#4u: %#04x\n",
//                   pos - 1, record[pos - 1],
//                   pos, record[pos],
//                   pos + 1, record[pos + 1],
//                   pos + 2, record[pos + 2]);

            pos += 3;

            printf("  nalUnitLength = %u\n",
                   pos - 3 - begin);

            begin = pos;
            flag = 0;

//            printf("%#4u: %#04x\n"
//                   "%#4u: %#04x\n",
//                   pos, record[pos],
//                   pos + 1, record[pos + 1]);

            uint32_t header = (uint32_t)benz_iso_bmff_as_u16(record + pos);

//            printf("header: %#06x\n",
//                   header);

            printf("NALU Header -- \n"
                   "  pos = %u\n"
                   "  length = %u\n"
                   "  nal_unit_type = %llu\n"
                   "  nuh_layer_id = %llu\n"
                   "  nuh_temporal_id_plus1 = %llu\n",
                   pos,
                   record_size,
                   (header >> 9) & 0x3F,            // 01111110 00000000
                   (header >> 3) & 0x3F,            // 00000001 11111000
                   header & 0x3F);                  // 00000000 00000111

            pos += 2;
        }
    }

    printf("  nalUnitLength = %u\n",
           pos - begin);
}

/**
 * @brief Run
 */

static int   nvdtRun(UNIVERSE* u){
    CUVIDSOURCEDATAPACKET packet;
    CUresult              result = CUDA_SUCCESS;
    int                   ret    = 0, match = 0;
    int                   i, j;
    int                   w, h;

    /* Initialize */
    if(nvdtInitMmap(u) != 0){
        fprintf(stdout, "Failed to initialize memory map!\n");
        return -1;
    }
    if(nvdtInitCUDA(u) != 0){
        fprintf(stdout, "Failed to initialize CUDA!\n");
        return -1;
    }
    if(nvdtInitFFmpeg(u) != 0){
        fprintf(stdout, "Failed to initialize FFmpeg!\n");
        return -1;
    }

    fprintf(stdout, "Dataset File size:         %15lu\n", u->fileH265Stat.st_size);
    fflush (stdout);

//    print_nalus(u->fileH265Data, u->fileH265Stat.st_size);

    uint8_t buffer[100000] = {0};
    int buf_len = 0;
    const uint8_t* record = &u->fileH265Data[79421];

    uint32_t lengthSizeMinusOne = (uint32_t)(record[21] & 3);                                      // 21 : 00000011
    uint32_t numOfArrays = (uint32_t)record[22];
    record += 23;                                                                                  // 22

    printf("lengthSizeMinusOne = %u\n"
           "numOfArrays = %u\n",
           lengthSizeMinusOne,
           numOfArrays);

    benz_putbe8(buffer + buf_len, 0x00); // 0x00 + annexb begin flag
    ++buf_len;

    for (int i=0; i < numOfArrays; i++)
    {
        // In our case, array_completeness will always == 1
//        uint32_t array_completeness = (uint32_t)(record[0] >> 7);                                  // 0 : 10000000
        // bits(1) reserved = 0;
        uint32_t NAL_unit_type = (uint32_t)(record[0] & 0x3f);                                     // 0 : 00111111
        uint32_t numNalus = (uint32_t)benz_iso_bmff_as_u16(record + 1);                            // 1 (1-2)
        record += 3;                         // 1 (1-2)

        printf("nvdecodeFeederThrdCore -- \n"
//               "array_completeness = %u\n"
               "NAL_unit_type = %u\n"
               "numNalus = %u\n",
//               array_completeness,
               NAL_unit_type,
               numNalus);

        uint8_t sps_id = 0;
        for (int j=0; j < numNalus; j++)
        {
            const uint8_t* nalBuf = NULL;
            int32_t nalBufSize = 0;

            uint32_t nalUnitLength = (uint32_t)benz_iso_bmff_as_u16(record);
            record += 2;

            // annexb begin flag
            benz_putbe8(buffer + buf_len, 0x00);
            benz_putbe16(buffer + buf_len + 1, 0x0001);
            buf_len += 3;

            memcpy(buffer + buf_len, record, nalUnitLength);
            buf_len += nalUnitLength;

            uint32_t header = (uint32_t)benz_iso_bmff_as_u16(record);

            printf("nvdecodeFeederThrdCore -- \n"
                   "NALU Header -- \n"
                   "  nalUnitLength = %u\n"
                   "  nal_unit_type = %u\n"
                   "  nuh_layer_id = %u\n"
                   "  nuh_temporal_id_plus1 = %u\n",
                   nalUnitLength,
                   (header >> 9) & 0x3f,            // 01111110 00000000
                   (header >> 3) & 0x3f,            // 00000001 11111000
                   header & 0x07);                  // 00000000 00000111

            switch (NAL_unit_type) {
            case 32: //VPS_NUT
                break;
            case 33: //SPS_NUT
                break;
            case 34: //PPS_NUT
                break;
            }
            record += nalUnitLength;
        }
    }

    // annexb begin flag
    benz_putbe8(buffer + buf_len, 0x00);
    benz_putbe16(buffer + buf_len + 1, 0x0001);
    buf_len += 3;

    memcpy(buffer + buf_len, &u->fileH265Data[68] + 4, 78845 - 4);
    buf_len += 78845 - 4;

    print_nalus(buffer, buf_len);

    /* Feed entire dataset in one go to NVDECODE. */

    printf("nvdtRun -- \n"
           "cuvidParseVideoData\n");

    packet.flags        = 0;
    packet.payload_size = buf_len;
    packet.payload      = buffer;
    packet.timestamp    = 0;
    result = cuvidParseVideoData(u->parser, &packet);
    if(result != CUDA_SUCCESS){
        return -1;
    }
    packet.flags        = CUVID_PKT_ENDOFSTREAM;
    packet.payload_size = 0;
    packet.payload      = NULL;
    packet.timestamp    = 0;
    result = cuvidParseVideoData(u->parser, &packet);
    if(result != CUDA_SUCCESS){
        return -1;
    }
    cudaDeviceSynchronize();
    cuvidDestroyVideoParser(u->parser);
    cuvidDestroyDecoder    (u->decoder);
    cudaDeviceSynchronize();

    /* Feed entire dataset in one go to FFmpeg. */
    u->packet->data = (void*)buffer;
    u->packet->size = (int)  buf_len;

    printf("nvdtRun -- \n"
           "avcodec_send_packet\n");

    ret = avcodec_send_packet  (u->codecCtx, u->packet);
    if(ret != 0){
        fprintf(stdout, "Error pushing packet! (%d)\n", ret);
        return ret;
    }

    printf("nvdtRun -- \n"
           "avcodec_send_packet NULL\n");

    ret = avcodec_send_packet  (u->codecCtx, NULL);
    if(ret != 0){
        fprintf(stdout, "Error flushing decoder! (%d)\n", ret);
        return ret;
    }

    printf("nvdtRun -- \n"
           "avcodec_receive_frame\n");

    ret = avcodec_receive_frame(u->codecCtx, u->frame);
    if(ret != 0){
        fprintf(stdout, "Error pulling frame! (%d)\n",  ret);
        return ret;
    }
    w = u->decoderInfo.ulTargetWidth;
    h = u->decoderInfo.ulTargetHeight;

    /* Final Check and Printouts */
    fprintf(stdout, "# of decoded images:       %15ld\n", u->numDecodedImages);
    fprintf(stdout, "# of mapped  images:       %15ld\n", u->numMappedImages);
    for(i=0;i<h;i+=2){
        for(j=0;j<w;j+=2){
            uint8_t ny0 = u->nvdecFramePtr [0*h*w+(i+0)*w+(j+0)];
            uint8_t ny1 = u->nvdecFramePtr [0*h*w+(i+0)*w+(j+1)];
            uint8_t ny2 = u->nvdecFramePtr [0*h*w+(i+1)*w+(j+0)];
            uint8_t ny3 = u->nvdecFramePtr [0*h*w+(i+1)*w+(j+1)];
            uint8_t ncb = u->nvdecFramePtr [1*h*w+(i/2)*w+(j+0)];
            uint8_t ncr = u->nvdecFramePtr [1*h*w+(i/2)*w+(j+1)];
            uint8_t fy0 = u->frame->data[0][(i+0)*u->frame->linesize[0]+(j+0)];
            uint8_t fy1 = u->frame->data[0][(i+0)*u->frame->linesize[0]+(j+1)];
            uint8_t fy2 = u->frame->data[0][(i+1)*u->frame->linesize[0]+(j+0)];
            uint8_t fy3 = u->frame->data[0][(i+1)*u->frame->linesize[0]+(j+1)];
            uint8_t fcb = u->frame->data[1][(i/2)*u->frame->linesize[1]+(j/2)];
            uint8_t fcr = u->frame->data[2][(i/2)*u->frame->linesize[2]+(j/2)];
            if(ny0 != fy0 || ny1 != fy1 || ny2 != fy2 || ny3 != fy3){
                fprintf(stdout, "ERROR: NVDEC-decoded image luma does not bitwise match FFmpeg!\n");
                return 1;
            }
            if(ncb != fcb || ncr != fcr){
                fprintf(stdout, "ERROR: NVDEC-decoded image chroma does not bitwise match FFmpeg!\n");
                fprintf(stdout, "ERROR: CbCr NVDEC %d,%d != %d,%d FFmpeg\n", ncb, ncr, fcb, fcr);
                return 1;
            }
        }
    }
    fprintf(stdout, "SUCCESS: NVDEC-decoded image bitwise matches FFmpeg!\n");

    /* Exit. */
    return match ? 0 : 1;
}



/**
 * Main
 */

int   main(int argc, char* argv[]){
    static UNIVERSE U = {0}, *u = &U;
    int i;
    
    /**
     * Argument parsing
     */
    
    u->args.device =    0;
    u->args.path   = NULL;
    u->fileH265Fd  =   -1;
    for(i=0;i<argc; i++){
        if(strcmp(argv[i], "--device") == 0){
            u->args.device = strtol(argv[++i]?argv[i]:"0", NULL, 0);
        }else{
            u->args.path   = argv[i];
        }
    }
    
    /**
     * Argument validation
     */
    
    if(!u->args.path){
        printf("No PATH/TO/FILE.h264 argument provided!\n");
        exit(-1);
    }
    
    /**
     * Run
     */
    
    return nvdtRun(u);
}

