# consistentdmd.github.io

Project page for **Consistent Distribution Matching for Data-Free Diffusion Distillation (CDMD)**,
served by GitHub Pages at <https://consistentdmd.github.io/>.

## Layout

- `index.html`: the page
- `static/css/index.css`, `static/js/index.js`: styles and small interactions (gallery, table toggle, lightbox, mobile menu)
- `static/images/`: figures and samples converted from the paper
- `static/videos/`: teaser animation and its poster frame
- `tools/cdmd_teaser.py`: Manim scene that renders the teaser animation

## Updating

- **Header links:** Paper, arXiv, alphaXiv, Code, and Checkpoints (Hugging Face) are `<a class="link-block">` buttons in `index.html`.
  Icons come from Font Awesome / Academicons; alphaXiv uses `static/images/alphaxiv.png` and Hugging Face the 🤗 emoji.
- **BibTeX:** update the entry in `index.html` once the arXiv ID is available.
- **Teaser video:** with [Manim Community](https://www.manim.community/) installed, run
  `manim -qh --fps 30 tools/cdmd_teaser.py CDMDTeaser`, then re-encode the output to `static/videos/cdmd_teaser.mp4`.

This page was built using the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template),
which was adopted from the [Nerfies](https://nerfies.github.io) project page.
Licensed under [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/).
