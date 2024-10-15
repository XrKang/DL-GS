cd /data/DLGS/NeRFStereo/DualCameraSynthetic/0011/wide_trainTest_10_SR
# Stereo-fusion-based Initialization
colmap feature_extractor --database_path pc.db --image_path wide_trainForPC --ImageReader.camera_model PINHOLE
colmap exhaustive_matcher --database_path pc.db
mkdir sparse
colmap mapper --database_path pc.db --image_path wide_trainForPC --output_path sparse
mkdir dense
colmap image_undistorter --image_path wide_trainForPC --input_path sparse/0 --output_path dense --output_type COLMAP
colmap patch_match_stereo --workspace_path dense --workspace_format COLMAP --PatchMatchStereo.geom_consistency true
colmap stereo_fusion --workspace_path dense --workspace_format COLMAP --input_type geometric --output_path dense/fused.ply
# colmap poisson_mesher --input_path dense/fused.ply --output_path dense/meshed-poisson.ply
# colmap delaunay_mesher --input_path dense --output_path dense/meshed-delaunay.ply

rm -rf sparse
# Camera Pose Estimation
colmap feature_extractor --database_path pose.db --image_path images --ImageReader.camera_model PINHOLE
colmap exhaustive_matcher --database_path pose.db
mkdir sparse
colmap mapper --database_path pose.db --image_path images --output_path sparse

python /data/DLGS/ours/llff/imgs2poses.py -s "/data/DLGS/NeRFStereo/DualCameraSynthetic/0011/wide_trainTest_10_SR"
