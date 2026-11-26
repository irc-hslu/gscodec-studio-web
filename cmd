python compress_ply_sequence.py x265_compression_rp3 \
  --ply-dir /mnt/e/Transmixr/UEGSDemo/Content/gps/001CL8K14100HK/ply \
  --data-dir /path/to/colmap_seq \
  --frame-num 100 \
  --result-dir results/my_exp \
  --test-view-id 0 1 2




python compress_ply_sequence.py x265_compression_rp3 --ply-dir /mnt/d/LiveStream/GSCodec_Studio/001CL8K14100HK/ply --data-dir /mnt/d/LiveStream/GSCodec_Studio/colmap_seq --frame-num 100 --gop-size 16 --result-dir /mnt/d/LiveStream/GSCodec_Studio/results/001CL8K14100HK_test \
  --data-factor 1 \
  --no-normalize-world-space \
  --test-view-id 0

python compress_ply_sequence.py x265_compression_rp3 \
  --ply-dir /mnt/d/LiveStream/GSCodec_Studio/ply/ \
  --frame-num 1 \
  --gop-size 16 \
  --result-dir results/PortalCam_MellifontAbbey_1_test \
  --skip-eval

  python compress_ply_sequence.py x265_compression_rp3 \
  --ply-dir /mnt/d/LiveStream/GSCodec_Studio/ply/ \
  --frame-num 1 \
  --gop-size 16 \
  --result-dir results/PortalCam_MellifontAbbey_1_test \
  --skip-eval


python compress_ply_sequence.py x265_compression_rp3 \
  --ply-dir /mnt/d/LiveStream/GSCodec_Studio/MellifontAbbey_1_SH_fast/ply-result \
  --frame-num 1 \
  --gop-size 16 \
  --result-dir results/MellifontAbbey_1_SH_fast \
  --skip-eval
  

python codec_ply_sequence.py x265_compression_rp3   --ply-dir /mnt/d/LiveStream/GSCodec_Studio/MellifontAbbey_1_SH_fast/ply-result  --ply-filename /mnt/d/LiveStream/GSCodec_Studio/MellifontAbbey_1_SH_fast/ply-result/point_cloud/iteration_100/point_cloud.ply   --frame-num 1   --result-dir results/MellifontAbbey_1_SH_fast_rp3   --decode-only   --skip-eval

python codec_ply_sequence.py x265_compression_rp2   --ply-dir /mnt/d/LiveStream/GSCodec_Studio/MellifontAbbey_1_SH_fast/ply-result  --ply-filename /mnt/d/LiveStream/GSCodec_Studio/MellifontAbbey_1_SH_fast/ply-result/point_cloud/iteration_100/point_cloud.ply   --frame-num 1   --result-dir results/MellifontAbbey_1_SH_fast_rp2   --decode-only   --skip-eval

python codec_ply_sequence.py x265_compression_rp1   --ply-dir /mnt/d/LiveStream/GSCodec_Studio/MellifontAbbey_1_SH_fast/ply-result  --ply-filename /mnt/d/LiveStream/GSCodec_Studio/MellifontAbbey_1_SH_fast/ply-result/point_cloud/iteration_100/point_cloud.ply   --frame-num 1   --result-dir results/MellifontAbbey_1_SH_fast_rp1   --decode-only   --skip-eval

 python codec_ply_sequence.py x265_compression_rp3   --ply-dir /mnt/d/LiveStream/GSCodec_Studio/ply   --ply-filename /mnt/d/LiveStream/GSCodec_Studio/ply/PortalCam_MellifontAbbey_1.ply   --frame-num 1   --result-dir results/PortalCam_MellifontAbbey_1_test   --decode-only   --skip-eval


python compress_ply_sequence.py x265_compression_rp3 \
  --ply-dir /mnt/d/LiveStream/GSCodec_Studio/ply/ \
  --frame-num 1 \
  --gop-size 16 \
  --result-dir results/MellifontAbbey_1_SH_full_rp3 --skip-eval

  