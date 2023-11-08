# HackaLOD 2023

[HackaLOD](https://netwerkdigitaalerfgoed.nl/hackalod/) Gouda 2023.

Intro to [teams](https://www.instagram.com/p/CzTNa4JpvDk/)

---

To run this on your computer, first do:

`docker build -t fizzysearch .`

And then you can run it with something like this:

`docker run --rm -it -v $(pwd):/data -e X_ENDPOINT=https://someplace/sparql -e DATA_PATH=/data -p 8000:8000 fizzysearch`

At the moment you still need the data files for the semantic search in the form of pickle files to also be present in the cwd.

## TODO

- [ ] The data files are generated and cached automatically at the first run of the system.

- [ ] Add Github action to build dockerfile

- [ ] Refactor core functionality to separate Python package on PyPI
