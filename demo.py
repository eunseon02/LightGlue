# demo.py

from pathlib import Path
import torch

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from lightglue import viz2d


def main():
    torch.set_grad_enabled(False)

    # Set path to images inside the cloned repo
    images = Path("assets")
    image0_path = images / "DSC_0411.JPG"
    image1_path = images / "DSC_0410.JPG"

    # Load and preprocess images
    image0 = load_image(image0_path)
    image1 = load_image(image1_path)

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize SuperPoint and LightGlue
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    # Extract features
    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))

    # Match features
    matches01 = matcher({"image0": feats0, "image1": feats1})

    # Remove batch dimension
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    # Get matched keypoints
    kpts0, kpts1 = feats0["keypoints"], feats1["keypoints"]
    matches = matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    # Visualize matches
    axes = viz2d.plot_images([image0, image1])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)

    # Visualize pruned keypoints
    kpc0 = viz2d.cm_prune(matches01["prune0"])
    kpc1 = viz2d.cm_prune(matches01["prune1"])
    viz2d.plot_images([image0, image1])
    viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)


if __name__ == "__main__":
    main()
