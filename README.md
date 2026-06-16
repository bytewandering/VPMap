VPMap

This repository contains the official implementation of VPMap, a cross-dimensional vascular image mapping framework for 3D cerebrovascular surgical navigation.

Paper: Towards Precise Guidance: A Novel Cross-Dimensional Mapping Framework for 3D Cerebrovascular Surgical Navigation

Overview

High-precision mapping between two-dimensional DSA images and three-dimensional vascular images is essential for 3D cerebrovascular surgical navigation. However, this task is challenging due to segmentation errors, initial pose variations, centerline misalignment, and projection-induced vascular overlap.

VPMap is designed to improve the robustness and accuracy of 2D/3D vascular image mapping by integrating vascular anatomy information and projective priors. The framework consists of three main components:

VASeg: Vascular Anatomy-aware Segmentation for 2D vessel extraction.
Vessel Skeletonization: Skeleton extraction and topology construction for 2D/3D vessels.
PPHReg: Projective Prior-enhanced Hierarchical Registration for robust 2D/3D vessel registration.
Framework

The overall pipeline includes:

2D vessel segmentation from DSA images.
Vessel skeleton extraction and topology construction.
Hierarchical 2D/3D vessel matching.
Projective-prior-weighted pose optimization.
3D vascular guidance based on the calibrated 2D/3D mapping relationship.
Installation
conda create -n vpmap python=3.10
conda activate vpmap
pip install -r requirements.txt
