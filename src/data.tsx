// Project title
export const title = "Projection-Conditioned Point Cloud Diffusion for Single-Image 3D Reconstruction"

// Short version of the abstract
export const description = "We propose a novel method for single-image 3D reconstruction which generates a sparse point cloud via a conditional denoising diffusion process with a geometrically-consistent conditioning process which we call projection conditioning."

// Abstract
export const abstract = "Reconstructing the 3D shape of an object from a single RGB image is a long-standing and highly challenging problem in computer vision. In this paper, we propose a novel method for single-image 3D reconstruction which generates a sparse point cloud via a conditional denoising diffusion process. Our method takes as input a single RGB image along with its camera pose and gradually denoises a set of 3D points, whose positions are initially sampled randomly from a three-dimensional Gaussian distribution, into the shape of an object. The key to our method is a geometrically-consistent conditioning process which we call projection conditioning: at each step in the diffusion process, we project local image features onto the partially-denoised point cloud from the given camera pose. This projection conditioning process enables us to generate high-resolution sparse geometries that are well-aligned with the input image, and can additionally be used to predict point colors after shape reconstruction. Moreover, due to the probabilistic nature of the diffusion process, our method is naturally capable of generating multiple different shapes consistent with a single input image. In contrast to prior work, our approach not only performs well on synthetic benchmarks, but also gives large qualitative improvements on complex real-world data."

// Institutions
export const institutions = {
  1: "Oxford University",
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
  // {
  //   'name': 'Iro Laina',
  //   'institutions': [1],
  //   'url': "http://campar.in.tum.de/Main/IroLaina"
  // },
  {
    'name': 'Andrea Vedaldi',
    'institutions': [1],
    'url': "https://www.robots.ox.ac.uk/~vedaldi/"
  }
]

// Links
export const links = {
  'paper': "#", // "https://arxiv.org/abs/2002.00733",
  'github': "https://github.com/lukemelas/projection-conditioned-point-cloud-diffusion"
}

// Acknowledgements
export const acknowledgements = "We thank xyz for abc..."

// Citation
export const citationId = "melaskyriazi2023projection"
export const citationAuthors = "Luke Melas-Kyriazi and Christian Rupprecht and Iro Laina and Andrea Vedaldi"
export const citationYear = "2023"
export const citationBooktitle = "Arxiv"

// Video
export const video_url = "https://www.youtube.com/embed/ScMzIvxBSi4"