// Project title
export const title = "RealFusion: 360° Reconstruction of Any Object from a Single Image"

// Short version of the abstract
export const description = "We propose a novel method for single-image 3D reconstruction which generates a sparse point cloud via a conditional denoising diffusion process with a geometrically-consistent conditioning process which we call projection conditioning."

// Abstract
export const abstract = "We consider the problem of reconstructing a full 360° photographic model of an object from a single image of it. We do so by fitting a neural radiance field to the image, but find this problem to be severely ill-posed. We thus take an off-the-self conditional image generator based on diffusion and engineer a prompt that encourages it to ``dream up'' novel views of the object. Using an approach inspired by DreamFields and DreamFusion, we fuse the given input view, the conditional prior, and other regularizers in a final, consistent reconstruction. We demonstrate state-of-the-art reconstruction results on benchmark images when compared to prior methods for monocular 3D reconstruction of objects. Qualitatively, our reconstructions provide a faithful match of the input view and a plausible extrapolation of its appearance and 3D shape, including to the side of the object not visible in the image."

// Institutions
export const institutions = {
  1: "VGG Group, Oxford University",
}

// Authors
export const authors = [
  {
    'name': 'Luke Melas-Kyriazi',
    'institutions': [1],
    'url': "https://github.com/lukemelas/"
  },
  {
    'name': 'Christian Rupprecht',
    'institutions': [1],
    'url': "https://chrirupp.github.io/"
  },
  {
    'name': 'Iro Laina',
    'institutions': [1],
    'url': "http://campar.in.tum.de/Main/IroLaina"
  },
  {
    'name': 'Andrea Vedaldi',
    'institutions': [1],
    'url': "https://www.robots.ox.ac.uk/~vedaldi/"
  }
]

// Links
export const links = {
  'paper': "https://arxiv.org/abs/2302.10663", // "https://arxiv.org/abs/2002.00733",
  'github': "https://github.com/lukemelas/realfusion"
}

// Acknowledgements
export const acknowledgements = "L.M.K. is supported by the Rhodes Trust. A.V. and C.R. are supported by ERC-UNION-CoG-101001212. C.R. is also supported by VisualAI EP/T028572/1."

// Citation
export const citationId = "melaskyriazi2023realfusion"
export const citationAuthors = "Luke Melas-Kyriazi and Christian Rupprecht and Iro Laina and Andrea Vedaldi"
export const citationYear = "2023"
export const citationBooktitle = "Arxiv"

// Video
export const video_url = "https://www.youtube.com/embed/EExD5zaxctE"  // placeholder: ScMzIvxBSi4