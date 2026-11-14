# Attribution: NVIDIA SimReady assets

The 3D assets in `hf/` come from **NVIDIA SimReady Assets**, published by NVIDIA Corporation on Hugging Face. So do the meshes, MJCF and GLB files made from them in `meshes/`, `mjcf/` and `glb/`.

- Dataset: https://huggingface.co/datasets/nvidia/simready-assets
- Revision used: `7879e0e50423290300ac20ecd6e727325f680b60` (lastModified 2026-11-04)
- License: **Creative Commons Attribution 4.0 International (CC-BY-4.0)**. The full text is in each `hf/<asset>/.metadata/LICENSE.txt`.
- Changes made:
  - Textures (.png) were not downloaded.
  - Geometry was triangulated and exported to OBJ and GLB with flat colours.
  - MJCF bodies were generated from the geometry.
  - Fence collision was simplified to boxes (see README.md).

| Asset | SimReady package_id |
|---|---|
| barrier_fence_metal_a02 | simready.hf.nvidia.simready-assets.barrier_fence_metal_a02.2026.10.08_220144422240 |
| construction_wheelbarrow_metal_a01 | simready.hf.nvidia.simready-assets.construction_wheelbarrow_metal_a01.2026.10.08_234101244534 |
| crate_plywood_b01 | simready.hf.nvidia.simready-assets.crate_plywood_b01.2026.10.09_002951927985 |
| drum_plastic_blue_a01 | simready.hf.nvidia.simready-assets.drum_plastic_blue_a01.2026.10.09_203953425620 |
| drum_steel_a01 | simready.hf.nvidia.simready-assets.drum_steel_a01.2026.10.09_011312912512 |
| food_produce_corn_b01 | simready.hf.nvidia.simready-assets.food_produce_corn_b01.2026.10.10_185858015618 |
| pallet_a01 | simready.hf.nvidia.simready-assets.pallet_a01.2026.10.09_143330045639 |

Suggested credit line: "3D props: NVIDIA SimReady Assets (huggingface.co/datasets/nvidia/simready-assets), CC-BY-4.0, modified."
