# PoseCNN_pytorch
This is an implementation of PoseCNN for 6D pose estimation on PROPSP dataset
## Dependencies

The project requires the following Python libraries and versions:

| Package       | Version    | Description                                         |
|---------------|------------|-----------------------------------------------------|
| `matplotlib`  | `3.7.2`    | For plotting and visualization.                     |
| `numpy`       | `1.24.3`   | Fundamental package for numerical computations.     |
| `Pillow`      | `11.0.0`   | Library for working with image processing tasks.    |
| `pyrender`    | `0.1.45`   | Rendering 3D scenes for visualization.              |
| `torch`       | `2.3.1`    | PyTorch library for deep learning.                  |
| `torchvision` | `0.18.1`   | PyTorch's library for vision-related tasks.         |
| `tqdm`        | `4.66.4`   | For creating progress bars in scripts.              |
| `trimesh`     | `4.4.3`    | For loading and working with 3D triangular meshes.  |

### Installing Dependencies

You can install the required dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt

```
## Visualization

The following section demonstrates the process visually using four images from the `PoseCNN_pytorch/image` directory. The images are displayed in a receding manner, switching every second.

<div align="center">
    <img id="receding-image" src="PoseCNN_pytorch/image/image1.png" alt="PoseCNN Visualization" width="600px">
</div>

### Images
- `6d1.png`
- `6d2.png`
- `6d3.png`
- `6d4.png`

### Animation Script
The images automatically switch every second using the following JavaScript snippet:

```html
<script>
  const images = [
    "PoseCNN_pytorch/image/6d1.png",
    "PoseCNN_pytorch/image/6d2.png",
    "PoseCNN_pytorch/image/6d3.png",
    "PoseCNN_pytorch/image/6d4.png",
  ];
  let index = 0;
  setInterval(() => {
    document.getElementById("receding-image").src = images[index];
    index = (index + 1) % images.length;
  }, 1000);
</script>

